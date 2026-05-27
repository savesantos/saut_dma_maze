"""Hardware IR-strip driver for the AlphaBot2 (3 central channels only).

This is the real-robot counterpart of ``ir_driver_gazebo``. It reads the
Waveshare TRSensors 5-channel reflectance array directly over the bit-banged
TLC1543 SPI bus (BCM pins from the course-staff reference
``TRSensors.py``) and consumes **only the three central channels**.

Why 3 of 5: the maze line-following policy only needs ``/line_pose``
(small lateral offset), ``/intersection`` (perpendicular crossing) and
``/line_lost`` (no tape under any central channel). Three sensors are
enough for all three signals and reduce the false-positive rate of the
outer pair, which on the AlphaBot2 frequently see the chassis shadow and
floor seams during turns.

Published topics (match ``ir_driver_gazebo`` so the rest of the stack is
agnostic to the source):

- ``/line_pose``   (``std_msgs/Float32``) -- normalised lateral position
  in ``[-1, +1]`` or NaN when no line is under the central trio.
- ``/intersection`` (``std_msgs/Empty``)  -- one-shot per perpendicular
  crossing (debounced).
- ``/line_lost``    (``std_msgs/Empty``)  -- emitted once the central
  trio has been dark for ``line_lost_grace_s``.

This node does **not** publish ``/goal_marker_seen``. On hardware that
signal is produced by ``fiducial_localizer`` from the camera stream.

Calibration: on startup the node blocks for ``calibration_seconds``
while the operator manually sweeps the chassis across the line so each
of the three central channels sees both white floor and black tape.
Per-channel min/max are then locked in. Run this with the motors off.
"""

from __future__ import annotations

import time
from typing import List, Optional, Tuple

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from std_msgs.msg import Empty, Float32


# BCM pin map for the TLC1543 ADC on the Waveshare TRSensors board
# (verbatim from AlphaBot2-Demo/RaspberryPi/AlphaBot2/python/TRSensors.py).
_CS = 5
_CLOCK = 25
_ADDRESS = 24
_DATA_OUT = 23

# The TRSensors board exposes 5 reflectance channels.
_NUM_SENSORS = 5
# Indices of the three central channels (left-of-centre, centre,
# right-of-centre). The two outermost channels are intentionally ignored.
_CENTER_IDX: Tuple[int, int, int] = (1, 2, 3)


def _read_tlc1543(gpio, num_channels: int = _NUM_SENSORS) -> List[int]:
    """Bit-bang one full conversion cycle of the TLC1543 ADC.

    Returns ``num_channels`` 10-bit values in channel order, each scaled
    by ``>> 2`` so the dynamic range is 0..255 (matches the reference
    code's behaviour: ``value[j] >>= 2``).
    """
    raw = [0] * (num_channels + 1)
    for j in range(num_channels + 1):
        gpio.output(_CS, gpio.LOW)
        # First 8 clocks: send the 4-bit channel address (MSB first),
        # then four don't-care clocks while latching the MSB nibble.
        for i in range(8):
            if i < 4:
                bit_high = ((j >> (3 - i)) & 0x01) != 0
                gpio.output(_ADDRESS, gpio.HIGH if bit_high else gpio.LOW)
            else:
                gpio.output(_ADDRESS, gpio.LOW)
            raw[j] <<= 1
            if gpio.input(_DATA_OUT):
                raw[j] |= 0x01
            gpio.output(_CLOCK, gpio.HIGH)
            gpio.output(_CLOCK, gpio.LOW)
        # Four more clocks for the LSB nibble.
        for _ in range(4):
            raw[j] <<= 1
            if gpio.input(_DATA_OUT):
                raw[j] |= 0x01
            gpio.output(_CLOCK, gpio.HIGH)
            gpio.output(_CLOCK, gpio.LOW)
        time.sleep(1e-4)
        gpio.output(_CS, gpio.HIGH)
    # Drop the first read (channel-select pipeline fill) and scale.
    return [raw[i] >> 2 for i in range(1, num_channels + 1)]


def _calibrated(raw: List[int],
                cmin: List[int],
                cmax: List[int]) -> List[int]:
    """Scale each channel into ``[0, 1000]`` using locked-in calibration."""
    out = [0] * len(raw)
    for i, v in enumerate(raw):
        denom = cmax[i] - cmin[i]
        if denom <= 0:
            out[i] = 0
            continue
        scaled = int((v - cmin[i]) * 1000 / denom)
        if scaled < 0:
            scaled = 0
        elif scaled > 1000:
            scaled = 1000
        out[i] = scaled
    return out


class IRDriverHardware(Node):
    """Publish ``/line_pose``, ``/intersection``, ``/line_lost`` on hardware."""

    def __init__(self) -> None:
        """Set up GPIO, calibrate the trio, then start the read timer."""
        super().__init__('ir_driver_hardware')

        # --- topics ---
        self.declare_parameter('line_pose_topic', '/line_pose')
        self.declare_parameter('intersection_topic', '/intersection')
        self.declare_parameter('line_lost_topic', '/line_lost')
        # --- rates ---
        self.declare_parameter('publish_rate_hz', 50.0)
        # --- calibration ---
        self.declare_parameter('calibration_seconds', 6.0)
        self.declare_parameter('calibration_samples_per_burst', 10)
        # Override calibration entirely (skip the warm-up sweep). Empty
        # list means "calibrate live". When provided, both lists must
        # have length 3 in channel order (left, centre, right).
        self.declare_parameter('calibrated_min', [0, 0, 0])
        self.declare_parameter('calibrated_max', [0, 0, 0])
        # Auto-spin during the calibration window: publish alternating
        # in-place yaw on cmd_vel so the three central channels sweep
        # over the line without the operator having to nudge the bot.
        # Disable (set false) if the robot is on a stand or you want to
        # do the sweep by hand.
        self.declare_parameter('calibration_auto_spin', True)
        self.declare_parameter('calibration_cmd_vel_topic',
                               '/alphabot2/cmd_vel')
        # Yaw magnitude (rad/s) and half-period (s) of the wiggle.
        # Defaults tuned for the real AlphaBot2: 0.6 rad/s + 0.7 s was below
        # the motor deadband / wheel inertia, so the chassis only jerked.
        # 1.8 rad/s with a 1.5 s half-period gives a clean ~120 deg sweep
        # before reversing, so all three central channels see both the
        # white floor and the black tape.
        self.declare_parameter('calibration_spin_yaw', 1.8)
        self.declare_parameter('calibration_spin_half_period_s', 1.5)
        # --- detection thresholds (calibrated units 0..1000) ---
        # A central channel is "on the black tape" when its calibrated
        # reading exceeds this value.
        self.declare_parameter('on_line_threshold', 400)
        # Used to decide that the whole trio is dark (no line under the
        # robot). Below this for every central channel triggers loss.
        self.declare_parameter('lost_threshold', 150)
        # A perpendicular crossing pulses the outer pair *and* the centre
        # at once; require the outer pair to both exceed this.
        self.declare_parameter('intersection_threshold', 700)
        # --- timing ---
        self.declare_parameter('line_lost_grace_s', 0.3)
        # Suppress repeat /intersection events until the robot has
        # cleared the crossing (at least one outer-centre channel falls
        # below ``on_line_threshold``).
        self.declare_parameter('intersection_min_gap_s', 0.4)
        # --- sign convention ---
        # +1 means: positive /line_pose <=> line is under the *right*
        # central channel (i.e. drift to the left). Flip to -1 if the
        # action_executor's correction direction is inverted on the real
        # robot after wiring.
        self.declare_parameter('sensor_sign', 1)

        self._pose_pub = self.create_publisher(
            Float32, str(self.get_parameter('line_pose_topic').value), 10)
        self._cross_pub = self.create_publisher(
            Empty, str(self.get_parameter('intersection_topic').value), 10)
        self._lost_pub = self.create_publisher(
            Empty, str(self.get_parameter('line_lost_topic').value), 10)

        self._on_line_th = int(self.get_parameter('on_line_threshold').value)
        self._lost_th = int(self.get_parameter('lost_threshold').value)
        self._cross_th = int(
            self.get_parameter('intersection_threshold').value)
        self._lost_grace = float(
            self.get_parameter('line_lost_grace_s').value)
        self._cross_gap = float(
            self.get_parameter('intersection_min_gap_s').value)
        self._sign = float(int(self.get_parameter('sensor_sign').value))

        # Defer the RPi.GPIO import so this module remains importable on
        # the dev laptop (lint, pytest) where the package is unavailable.
        try:
            import RPi.GPIO as GPIO  # noqa: N814
        except ImportError as exc:
            raise RuntimeError(
                'ir_driver_hardware requires RPi.GPIO; install it on the '
                'AlphaBot2 (apt: python3-rpi.gpio) or run ir_driver_gazebo '
                'instead for simulation.') from exc
        self._gpio = GPIO
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)
        GPIO.setup(_CLOCK, GPIO.OUT)
        GPIO.setup(_ADDRESS, GPIO.OUT)
        GPIO.setup(_CS, GPIO.OUT)
        GPIO.setup(_DATA_OUT, GPIO.IN, GPIO.PUD_UP)

        # cmd_vel publisher used optionally during calibration to spin
        # the robot in place. Created unconditionally so it survives
        # parameter toggles at runtime, but only written to when
        # calibration_auto_spin is true.
        self._cmd_vel_pub = self.create_publisher(
            Twist,
            str(self.get_parameter('calibration_cmd_vel_topic').value),
            10)

        # Decide calibration source.
        preset_min = [int(v) for v in (
            self.get_parameter('calibrated_min').value or [])]
        preset_max = [int(v) for v in (
            self.get_parameter('calibrated_max').value or [])]
        if (len(preset_min) == 3 and len(preset_max) == 3
                and any(preset_max)):
            self._cmin = preset_min
            self._cmax = preset_max
            self.get_logger().info(
                f'Using preset calibration min={self._cmin} max={self._cmax}')
        else:
            self._cmin, self._cmax = self._live_calibrate(
                float(self.get_parameter('calibration_seconds').value),
                int(self.get_parameter(
                    'calibration_samples_per_burst').value),
                bool(self.get_parameter(
                    'calibration_auto_spin').value),
                float(self.get_parameter(
                    'calibration_spin_yaw').value),
                float(self.get_parameter(
                    'calibration_spin_half_period_s').value))

        self._lost_since: Optional[float] = None
        self._lost_published = False
        self._last_intersection_t: float = -1e9
        self._in_intersection = False

        rate = float(self.get_parameter('publish_rate_hz').value)
        self._timer = self.create_timer(1.0 / max(rate, 1e-3), self._tick)
        self.get_logger().info(
            f'IRDriverHardware ready: using channels {_CENTER_IDX} at '
            f'{rate:.1f} Hz')

    # ------------------------------------------------------------ helpers

    def _now_s(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _read_center(self) -> List[int]:
        """Return the three central raw readings in [left, centre, right]."""
        all_raw = _read_tlc1543(self._gpio, _NUM_SENSORS)
        return [all_raw[i] for i in _CENTER_IDX]

    def _live_calibrate(self, seconds: float,
                        samples_per_burst: int,
                        auto_spin: bool = False,
                        spin_yaw: float = 0.6,
                        spin_half_period_s: float = 0.7,
                        ) -> Tuple[List[int], List[int]]:
        """Block while sweeping the chassis over the line.

        Tracks per-channel min and max raw readings (3 central channels
        only). When ``auto_spin`` is True the node publishes alternating
        in-place yaw on cmd_vel so the trio sweeps the line on its own;
        otherwise the operator must nudge the bot by hand.
        """
        if auto_spin:
            self.get_logger().warning(
                f'Calibrating IR for {seconds:.1f}s -- spinning in place '
                f'(yaw={spin_yaw:.2f} rad/s, half-period='
                f'{spin_half_period_s:.2f}s). Keep the bot centred on the '
                f'line.')
        else:
            self.get_logger().warning(
                f'Calibrating IR for {seconds:.1f}s -- sweep the '
                f'AlphaBot2 over the line now (motors must be OFF).')
        # Seed with the first reading so the min/max are real numbers,
        # not 0/1023 placeholders that would never tighten.
        first = self._read_center()
        cmin = list(first)
        cmax = list(first)
        deadline = time.monotonic() + max(seconds, 0.0)
        bursts = 0
        spin_started = time.monotonic()
        half_p = max(float(spin_half_period_s), 0.05)
        try:
            while time.monotonic() < deadline:
                if auto_spin:
                    # Square-wave yaw: +yaw for half_p seconds, then
                    # -yaw for half_p seconds, repeat. Linear x stays 0.
                    phase = int(
                        (time.monotonic() - spin_started) // half_p)
                    sign = 1.0 if (phase % 2 == 0) else -1.0
                    msg = Twist()
                    msg.angular.z = sign * float(spin_yaw)
                    self._cmd_vel_pub.publish(msg)
                for _ in range(max(samples_per_burst, 1)):
                    vals = self._read_center()
                    for i, v in enumerate(vals):
                        if v < cmin[i]:
                            cmin[i] = v
                        if v > cmax[i]:
                            cmax[i] = v
                bursts += 1
        finally:
            if auto_spin:
                # Always brake at the end of calibration, even on error.
                stop = Twist()
                self._cmd_vel_pub.publish(stop)
        self.get_logger().info(
            f'Calibration done after {bursts} bursts: '
            f'min={cmin} max={cmax}')
        # Sanity-check: if any channel never saw a contrast >50 raw
        # units the operator likely forgot to sweep. Warn loudly but
        # keep going so the operator can iterate without a restart.
        for i, (lo, hi) in enumerate(zip(cmin, cmax)):
            if hi - lo < 50:
                self.get_logger().error(
                    f'Channel {i} contrast only {hi - lo} raw units; '
                    f'sweep again or set calibrated_min/max via params.')
        return cmin, cmax

    # ------------------------------------------------------------ main loop

    def _tick(self) -> None:
        raw = self._read_center()
        cal = _calibrated(raw, self._cmin, self._cmax)
        v_left, v_centre, v_right = cal

        now = self._now_s()

        # 1) Line lost: all three central channels below lost threshold.
        if (v_left < self._lost_th and v_centre < self._lost_th
                and v_right < self._lost_th):
            self._pose_pub.publish(Float32(data=float('nan')))
            if self._lost_since is None:
                self._lost_since = now
            elif (not self._lost_published
                    and now - self._lost_since >= self._lost_grace):
                self._lost_pub.publish(Empty())
                self._lost_published = True
            self._in_intersection = False
            return

        self._lost_since = None
        self._lost_published = False

        # 2) Line pose: weighted average across the three central channels
        # in {-1, 0, +1}, normalised by total intensity. Sensor sign lets
        # the integrator flip handedness without a code change.
        # Only mix in channels above on_line_threshold so faint noise on
        # an otherwise-dark sensor doesn't bias the centre estimate.
        weights = (-1.0, 0.0, 1.0)
        num = 0.0
        den = 0.0
        for w, v in zip(weights, cal):
            if v >= self._on_line_th or v >= self._lost_th:
                num += w * v
                den += v
        if den > 0:
            pose = self._sign * (num / den)
            # Clamp for safety.
            if pose > 1.0:
                pose = 1.0
            elif pose < -1.0:
                pose = -1.0
            self._pose_pub.publish(Float32(data=float(pose)))
        else:
            self._pose_pub.publish(Float32(data=float('nan')))

        # 3) Intersection: both outer-central channels are strongly on
        # the line at the same time (perpendicular tape spans the trio).
        # Rising-edge with debounce so a single crossing emits one event.
        on_crossing = (v_left >= self._cross_th
                       and v_right >= self._cross_th)
        if on_crossing:
            if (not self._in_intersection
                    and now - self._last_intersection_t >= self._cross_gap):
                self._cross_pub.publish(Empty())
                self._last_intersection_t = now
            self._in_intersection = True
        else:
            # Require one outer-central channel to fall below the
            # *on-line* threshold before re-arming, not just below the
            # cross threshold; this matches "robot has cleared the
            # crossing tape entirely".
            if (v_left < self._on_line_th
                    or v_right < self._on_line_th):
                self._in_intersection = False

    # ------------------------------------------------------------ shutdown

    def destroy_node(self) -> bool:
        """Release GPIO before shutting down the node."""
        try:
            if getattr(self, '_gpio', None) is not None:
                self._gpio.cleanup()
        except Exception:  # noqa: BLE001
            # GPIO cleanup failure is non-fatal; log and continue.
            self.get_logger().warning('GPIO cleanup raised; ignoring.')
        return super().destroy_node()


def main(args: Optional[list] = None) -> None:
    """Entry point: ``ros2 run maze_mdp ir_driver_hardware``."""
    rclpy.init(args=args)
    node = IRDriverHardware()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except rclpy.executors.ExternalShutdownException:
        # rclpy context was shut down from outside (e.g. SIGINT delivered to
        # the whole process group, or a second instance of this node grabbed
        # the same name). Treat as a normal exit.
        pass
    finally:
        try:
            node.destroy_node()
        except Exception:  # noqa: BLE001
            pass
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()

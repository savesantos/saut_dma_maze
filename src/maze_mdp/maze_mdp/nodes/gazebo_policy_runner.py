"""HW-mirrored Gazebo policy runner.

Counterpart of :mod:`maze_mdp.hardware.line_follow_policy` that runs in
Gazebo Classic. Same control logic, same control flow, same logging
vocabulary so we can validate the Pi-side deployment in sim:

- PID line-follow on ``/line_pose`` between intersections (analog of
  ``TR.readLine()`` ``position - 2000`` proportional control).
- On every ``/intersection`` event consult the loaded policy
  (``argmax`` of ``Q``, or stored ``pi``) and dispatch one of
  ``FORWARD`` / ``TURN_LEFT`` / ``TURN_RIGHT``.
- ``FORWARD`` is a short open-loop forward burst (mirror of HW's
  ``go_forward``) so the robot leaves the cross before the PID resumes.
- Turns are open-loop spin-in-place (mirror of HW's ``turn_right`` /
  ``turn_left``) followed by **camera-based alignment** (rotate until
  the line in the front-camera frame is vertical) and an IR-gated
  forward creep until ``/line_pose`` is valid again.

The robot's discrete cell estimate is maintained locally (same code as
the Pi script), so this node replaces ``policy_runner`` +
``action_executor`` + ``cell_tracker`` in the ``mirror_hw:=true``
launch path.

For debuggability the node also subscribes to ``/virtual_odometry``
(the ground-truth Gazebo pose, not available on hardware) and logs a
god-view ``(row, col, heading)`` estimate at every state transition
and at a fixed throttled rate. That way the chat-side reviewer can
verify whether the robot is actually where the discrete estimate
claims it is.
"""

from __future__ import annotations

import math
from enum import Enum
from pathlib import Path
from typing import Optional

import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Empty, Float32

from maze_msgs.msg import MazeGrid

from maze_mdp.hardware.camera_align import (
    estimate_line_from_frame,
)
from maze_mdp.policy import greedy_policy, load_policy


# Mirrors maze_mdp.mdp.Action / Heading.
FORWARD, TURN_LEFT, TURN_RIGHT = 0, 1, 2
# heading -> (dr, dc) for N, E, S, W
_HEADING_DELTA = ((-1, 0), (0, 1), (1, 0), (0, -1))
_HEADING_NAMES = ('N', 'E', 'S', 'W')
_ACTION_NAMES = {FORWARD: 'FORWARD', TURN_LEFT: 'TURN_LEFT',
                 TURN_RIGHT: 'TURN_RIGHT'}


class _State(Enum):
    """High-level FSM state. Names mirror the HW script's comments."""

    WAIT = 'wait'
    LINE_FOLLOW = 'line_follow'
    FORWARD_BURST = 'forward_burst'
    CENTER_ON_CROSS = 'center_on_cross'
    TURN = 'turn'
    ALIGN = 'align'
    CREEP = 'creep'
    DONE = 'done'


def _latched_qos() -> QoSProfile:
    return QoSProfile(
        depth=1,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
        history=HistoryPolicy.KEEP_LAST,
    )


def _decode_image(msg: Image) -> Optional[np.ndarray]:
    """Decode a ROS Image into an HxWx3 RGB or HxW gray numpy array."""
    enc = msg.encoding
    h = int(msg.height)
    w = int(msg.width)
    buf = np.frombuffer(bytes(msg.data), dtype=np.uint8)
    if enc in ('rgb8', 'bgr8'):
        if buf.size != h * w * 3:
            return None
        arr = buf.reshape(h, w, 3)
        if enc == 'bgr8':
            arr = arr[..., ::-1]
        return arr
    if enc == 'mono8':
        if buf.size != h * w:
            return None
        return buf.reshape(h, w)
    return None


def _yaw_from_quat(qx: float, qy: float, qz: float, qw: float) -> float:
    """ZYX yaw from a unit quaternion."""
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)


def _yaw_to_heading(yaw: float) -> int:
    """Snap a yaw (rad) to a cardinal heading {N=0, E=1, S=2, W=3}.

    Matches the world frame used by ``maze_to_sdf``: ``yaw=0`` -> East,
    ``yaw=+pi/2`` -> North, ``yaw=-pi/2`` -> South, ``yaw=+/-pi`` -> West.
    """
    y = math.atan2(math.sin(yaw), math.cos(yaw))
    # Closest of {pi/2 -> N, 0 -> E, -pi/2 -> S, +/-pi -> W}.
    candidates = ((0, math.pi / 2), (1, 0.0), (2, -math.pi / 2),
                  (3, math.pi), (3, -math.pi))
    return min(candidates, key=lambda kv: abs(math.atan2(
        math.sin(y - kv[1]), math.cos(y - kv[1]))))[0]


class GazeboPolicyRunner(Node):
    """Hardware-mirror controller for the Gazebo AlphaBot2."""

    def __init__(self) -> None:
        super().__init__('gazebo_policy_runner')

        # --- parameters ---
        self.declare_parameter('policy_path', '')
        self.declare_parameter('rows', 0)
        self.declare_parameter('cols', 0)
        self.declare_parameter('cell_size', 0.20)
        self.declare_parameter('start_row', 0)
        self.declare_parameter('start_col', 0)
        self.declare_parameter('start_heading', 1)
        # Topics.
        self.declare_parameter('cmd_topic', '/alphabot2/cmd_vel')
        self.declare_parameter('line_topic', '/line_pose')
        self.declare_parameter('cross_topic', '/intersection')
        self.declare_parameter('marker_topic', '/goal_marker_seen')
        self.declare_parameter('image_topic', '/image/raw')
        self.declare_parameter('odom_topic', '/virtual_odometry')
        self.declare_parameter('maze_topic', '/maze')
        # Line-follow PID. Matches the HW reference but scaled for
        # /line_pose (meters) instead of TRSensors raw (0..4000).
        self.declare_parameter('forward_speed', 0.10)
        self.declare_parameter('kp', 1.2)
        self.declare_parameter('kd', 1.0)
        self.declare_parameter('ki', 0.0)
        self.declare_parameter('omega_clamp', 2.5)
        # Forward burst (HW go_forward equivalent): drive forward briefly
        # to leave the intersection so the next intersection event isn't
        # a re-trigger of the same cross.
        self.declare_parameter('forward_burst_s', 0.6)
        # Centre-on-cross: when policy is TURN, drive forward to the
        # cell centre before spinning so the rotation happens over the
        # cross instead of cutting a diagonal. The state exits when the
        # along-heading signed distance to the cell centre crosses
        # ``center_tol_m`` (sim, via /virtual_odometry) or after the
        # ``center_drive_max_s`` safety timeout.
        self.declare_parameter('center_drive_max_s', 1.5)
        self.declare_parameter('center_tol_m', 0.015)
        self.declare_parameter('center_speed', 0.08)
        # Open-loop turn.
        self.declare_parameter('turn_speed', 1.5)
        self.declare_parameter('turn_open_loop_s', 1.10)
        # Camera alignment.
        self.declare_parameter('camera_align', True)
        self.declare_parameter('align_omega', 0.8)
        self.declare_parameter('align_theta_tol_deg', 4.0)
        self.declare_parameter('align_stable_frames', 3)
        self.declare_parameter('align_timeout_s', 2.0)
        self.declare_parameter('creep_speed', 0.06)
        self.declare_parameter('creep_timeout_s', 2.0)
        # Misc.
        self.declare_parameter('exit_on_goal', True)
        self.declare_parameter('control_rate_hz', 30.0)
        self.declare_parameter('god_log_period_s', 1.0)
        self.declare_parameter('camera_pitch_rad', 0.7854)  # 45 deg down

        gp = self.get_parameter
        policy_path = gp('policy_path').get_parameter_value().string_value
        if not policy_path:
            raise RuntimeError('policy_path is required.')
        self._rows = int(gp('rows').value)
        self._cols = int(gp('cols').value)
        self._cell_size = float(gp('cell_size').value)
        if self._rows <= 0 or self._cols <= 0:
            raise RuntimeError('rows/cols must be positive.')

        self._row = int(gp('start_row').value)
        self._col = int(gp('start_col').value)
        self._heading = int(gp('start_heading').value)

        # --- policy ---
        bundle = load_policy(policy_path)
        if 'pi' in bundle:
            self._pi = np.asarray(bundle['pi'], dtype=np.int64)
        elif 'Q' in bundle:
            self._pi = greedy_policy(np.asarray(bundle['Q']))
        else:
            raise RuntimeError(f'No pi or Q in {policy_path}')
        expected = self._rows * self._cols * 4
        if self._pi.size != expected:
            raise RuntimeError(
                f'policy size {self._pi.size} != rows*cols*4 = {expected}')
        self.get_logger().info(
            f'Loaded policy: {self._pi.size} states from {policy_path}')

        # --- gains ---
        self._fwd_speed = float(gp('forward_speed').value)
        self._kp = float(gp('kp').value)
        self._kd = float(gp('kd').value)
        self._ki = float(gp('ki').value)
        self._omega_clamp = float(gp('omega_clamp').value)
        self._forward_burst_s = float(gp('forward_burst_s').value)
        self._center_drive_max_s = float(gp('center_drive_max_s').value)
        self._center_tol_m = float(gp('center_tol_m').value)
        self._center_speed = float(gp('center_speed').value)
        self._turn_speed = float(gp('turn_speed').value)
        self._turn_open_loop_s = float(gp('turn_open_loop_s').value)
        self._camera_align = bool(gp('camera_align').value)
        self._align_omega = float(gp('align_omega').value)
        self._theta_tol = math.radians(float(gp('align_theta_tol_deg').value))
        self._align_stable_frames = int(gp('align_stable_frames').value)
        self._align_timeout_s = float(gp('align_timeout_s').value)
        self._creep_speed = float(gp('creep_speed').value)
        self._creep_timeout_s = float(gp('creep_timeout_s').value)
        self._exit_on_goal = bool(gp('exit_on_goal').value)
        self._god_log_period_s = float(gp('god_log_period_s').value)
        # theta_offset: the 45-deg-down mount means even on a perfectly
        # centered line the column does not change with row at the same
        # rate as in a top-down view. The fit's slope at a centered line
        # is still ~0 because the line is symmetric L/R along the image
        # vertical axis, but mounting asymmetries can be subtracted here.
        # Default 0; tune from --camera-align-debug on real hardware.
        self._theta_offset = 0.0

        # --- pubs / subs ---
        cmd_topic = gp('cmd_topic').get_parameter_value().string_value
        self._cmd_pub = self.create_publisher(Twist, cmd_topic, 10)
        self.create_subscription(
            MazeGrid, gp('maze_topic').get_parameter_value().string_value,
            self._on_maze, _latched_qos())
        self.create_subscription(
            Float32, gp('line_topic').get_parameter_value().string_value,
            self._on_line, 30)
        self.create_subscription(
            Empty, gp('cross_topic').get_parameter_value().string_value,
            self._on_cross, 10)
        self.create_subscription(
            Bool, gp('marker_topic').get_parameter_value().string_value,
            self._on_marker, 10)
        self.create_subscription(
            Image, gp('image_topic').get_parameter_value().string_value,
            self._on_image, 5)
        self.create_subscription(
            Odometry, gp('odom_topic').get_parameter_value().string_value,
            self._on_odom, 10)

        # --- runtime state ---
        self._state = _State.WAIT
        self._state_start_t = self._now_s()
        self._goal: Optional[tuple[int, int]] = None
        self._line_pose = float('nan')
        self._line_valid = False
        self._prev_line = 0.0
        self._integral = 0.0
        self._latest_image: Optional[np.ndarray] = None
        # Cell-advance is now anchored on /intersection events from
        # ir_driver_gazebo. The first policy lookup happens at the
        # start cell (no increment); every subsequent /intersection in
        # LINE_FOLLOW means "arrived at a new cell" and advances the
        # estimate by HEADING_DELTA[heading] *before* the policy lookup.
        # ``_start_processed`` flips True the first time we run the
        # policy at the start cell (either via a real /intersection or
        # the synthetic one fired from ``_on_maze``).
        self._start_processed = False
        # Camera-align bookkeeping.
        self._align_stable = 0
        self._goal_reached = False
        # Pending heading change deferred until CENTER_ON_CROSS finishes.
        self._pending_heading_delta = 0
        # God-view (truth) pose.
        self._true_xy: Optional[tuple[float, float]] = None
        self._true_yaw: float = 0.0
        self._true_cell: Optional[tuple[int, int, int]] = None
        self._last_god_log_t = self._now_s()

        rate_hz = float(gp('control_rate_hz').value)
        self._tick_period = 1.0 / max(rate_hz, 1.0)
        self._timer = self.create_timer(self._tick_period, self._tick)

        self.get_logger().info(
            f'start cell=({self._row},{self._col},'
            f'{_HEADING_NAMES[self._heading]}) '
            f'rows={self._rows} cols={self._cols} '
            f'cell_size={self._cell_size}m forward={self._fwd_speed} m/s '
            f'turn={self._turn_speed} rad/s')

    # ---------------------------------------------------------- utils
    def _now_s(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _state_elapsed(self) -> float:
        return self._now_s() - self._state_start_t

    def _transition(self, new_state: _State, reason: str = '') -> None:
        if new_state == self._state:
            return
        gv = self._format_god_view()
        self.get_logger().info(
            f'FSM {self._state.value} -> {new_state.value} '
            f'(estim=({self._row},{self._col},{_HEADING_NAMES[self._heading]})'
            f' {gv} t={self._state_elapsed():.2f}s {reason})')
        self._state = new_state
        self._state_start_t = self._now_s()
        self._align_stable = 0
        if new_state == _State.LINE_FOLLOW:
            self._integral = 0.0
            self._prev_line = 0.0
        if new_state == _State.DONE:
            self._stop()

    def _format_god_view(self) -> str:
        if self._true_xy is None:
            return 'truth=?'
        x, y = self._true_xy
        if self._true_cell is None:
            return f'truth=xy({x:+.3f},{y:+.3f})'
        r, c, h = self._true_cell
        return (f'truth=({r},{c},{_HEADING_NAMES[h]}) '
                f'xy({x:+.3f},{y:+.3f}) yaw={math.degrees(self._true_yaw):+.1f}')

    def _publish(self, lin: float, ang: float) -> None:
        msg = Twist()
        msg.linear.x = float(lin)
        msg.angular.z = float(ang)
        self._cmd_pub.publish(msg)

    def _stop(self) -> None:
        self._publish(0.0, 0.0)

    # ---------------------------------------------------------- callbacks
    def _on_maze(self, msg: MazeGrid) -> None:
        cells = np.asarray(msg.cells, dtype=np.int8).reshape(
            int(msg.rows), int(msg.cols))
        goals = np.argwhere(cells == 2)
        if goals.size == 0:
            self.get_logger().warn('Maze has no goal cell yet.')
            return
        self._goal = (int(goals[0, 0]), int(goals[0, 1]))
        self.get_logger().info(f'Maze received: goal={self._goal}')
        if self._state == _State.WAIT:
            self._transition(_State.LINE_FOLLOW, 'maze ready')
            # Synthesize the start-cell /intersection: ir_driver may have
            # already fired it before our subscription was active (the
            # runner is launched via TimerAction). Calling the policy
            # here guarantees we *always* consult the policy at the
            # start cell.
            if not self._start_processed:
                self.get_logger().info(
                    'synthesised start-cell intersection at '
                    f'({self._row},{self._col},'
                    f'{_HEADING_NAMES[self._heading]})')
                self._start_processed = True
                self._handle_intersection()

    def _on_line(self, msg: Float32) -> None:
        val = float(msg.data)
        if math.isnan(val):
            self._line_valid = False
        else:
            self._line_valid = True
            self._line_pose = val

    def _on_cross(self, _msg: Empty) -> None:
        if self._state != _State.LINE_FOLLOW:
            # Ignore mid-action intersection events (turn / creep can pass
            # over the cross again briefly without it being a new cell).
            return
        if not self._start_processed:
            # First /intersection received in LINE_FOLLOW: this is the
            # start cell. No cell increment (estim already matches
            # start cell), just consult the policy.
            self._start_processed = True
            self.get_logger().info(
                'first /intersection at start cell -> consult policy')
        else:
            # Every subsequent /intersection in LINE_FOLLOW means we
            # have arrived at a new cell. Advance the estimate
            # *before* the policy lookup so the lookup happens at the
            # cell we are physically in.
            dr, dc = _HEADING_DELTA[self._heading]
            self._row += dr
            self._col += dc
            self.get_logger().info(
                f'/intersection -> advance estim to '
                f'({self._row},{self._col},'
                f'{_HEADING_NAMES[self._heading]})')
        self._handle_intersection()

    def _on_marker(self, msg: Bool) -> None:
        # Just for logging; goal detection in mirror_hw mode is via the
        # internal cell estimate, like on hardware.
        if msg.data:
            self.get_logger().info('goal marker visible (truth-side sensor)')

    def _on_image(self, msg: Image) -> None:
        arr = _decode_image(msg)
        if arr is not None:
            self._latest_image = arr

    def _on_odom(self, msg: Odometry) -> None:
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        self._true_xy = (float(p.x), float(p.y))
        self._true_yaw = _yaw_from_quat(q.x, q.y, q.z, q.w)
        # World frame: x = col*cell_size, y = -row*cell_size.
        c_est = round(self._true_xy[0] / self._cell_size)
        r_est = round(-self._true_xy[1] / self._cell_size)
        h_est = _yaw_to_heading(self._true_yaw)
        self._true_cell = (int(r_est), int(c_est), int(h_est))
        # Throttled god-view print.
        now = self._now_s()
        if now - self._last_god_log_t >= self._god_log_period_s:
            self._last_god_log_t = now
            self.get_logger().info(
                f'[god-view] {self._format_god_view()} '
                f'state={self._state.value} '
                f'line_pose={"%.3f" % self._line_pose if self._line_valid else "nan"}')

    # ---------------------------------------------------------- intersection handler
    def _handle_intersection(self) -> None:
        assert self._goal is not None
        if (self._row, self._col) == self._goal:
            self.get_logger().info(
                f'reached goal at ({self._row},{self._col})')
            self._goal_reached = True
            self._transition(_State.DONE, 'goal')
            return
        s = (self._row * self._cols + self._col) * 4 + self._heading
        if not 0 <= s < self._pi.size:
            self.get_logger().warn(f'state index {s} out of range')
            self._transition(_State.DONE, 'bad state')
            return
        action = int(self._pi[s])
        self.get_logger().info(
            f'POLICY cell=({self._row},{self._col},'
            f'{_HEADING_NAMES[self._heading]}) -> {_ACTION_NAMES[action]}')
        if action == FORWARD:
            self._turn_sign = 0.0
            self._transition(_State.FORWARD_BURST, 'policy=FORWARD')
        elif action == TURN_LEFT:
            # Defer heading update until the spin actually completes;
            # CENTER_ON_CROSS drives forward to the cell centre first.
            self._pending_heading_delta = -1
            self._turn_sign = +1.0   # left = CCW = +omega
            self._transition(_State.CENTER_ON_CROSS, 'policy=TURN_LEFT')
        elif action == TURN_RIGHT:
            self._pending_heading_delta = +1
            self._turn_sign = -1.0   # right = CW = -omega
            self._transition(_State.CENTER_ON_CROSS, 'policy=TURN_RIGHT')
        else:
            self.get_logger().warn(f'unknown action {action}; stopping')
            self._transition(_State.DONE, 'unknown action')

    # ---------------------------------------------------------- tick
    def _tick(self) -> None:
        if self._state == _State.WAIT:
            return
        if self._state == _State.LINE_FOLLOW:
            self._do_line_follow()
            return
        if self._state == _State.FORWARD_BURST:
            self._do_forward_burst()
            return
        if self._state == _State.CENTER_ON_CROSS:
            self._do_center_on_cross()
            return
        if self._state == _State.TURN:
            self._do_turn()
            return
        if self._state == _State.ALIGN:
            self._do_align()
            return
        if self._state == _State.CREEP:
            self._do_creep()
            return
        if self._state == _State.DONE:
            self._stop()
            return

    # ---------------------------------------------------------- behaviors
    def _do_line_follow(self) -> None:
        if not self._line_valid:
            # Line lost between intersections: keep going straight at
            # half speed; the next intersection (or a recovered
            # /line_pose) will get us back on track. Matches HW
            # behaviour where momentarily missing the line just lets the
            # previous PID output coast.
            self._publish(self._fwd_speed * 0.5, 0.0)
            return
        e = self._line_pose
        de = (e - self._prev_line) / self._tick_period
        self._integral += e * self._tick_period
        # +e (line to the right of strip) -> need to turn right -> negative ω.
        omega = -(self._kp * e + self._kd * de + self._ki * self._integral)
        omega = float(np.clip(omega, -self._omega_clamp, self._omega_clamp))
        self._prev_line = e
        self._publish(self._fwd_speed, omega)

    def _do_forward_burst(self) -> None:
        """Drive forward briefly to leave the current cross.

        Unlike the HW go_forward primitive, the cell estimate is *not*
        advanced here. Cell advancement is anchored on the next
        ``/intersection`` event (see :py:meth:`_on_cross`). The burst
        only exists to push the robot off the current cross so that
        the next rising-edge ``/intersection`` corresponds to the
        physically next cell.
        """
        if self._state_elapsed() >= self._forward_burst_s:
            self.get_logger().info(
                f'FORWARD_BURST done (estim unchanged at '
                f'({self._row},{self._col},'
                f'{_HEADING_NAMES[self._heading]}); await /intersection)')
            self._transition(_State.LINE_FOLLOW, 'forward burst done')
            return
        self._publish(self._fwd_speed, 0.0)

    def _do_center_on_cross(self) -> None:
        """Drive forward to the cell centre before spinning.

        ``/intersection`` fires the moment the robot enters the
        half-cell radius around the cell centre, so a TURN action
        executed immediately would spin ~10 cm short of the cross and
        cut a diagonal. This state drives forward at ``center_speed``
        until the along-heading signed distance from robot to cell
        centre is below ``center_tol_m`` (sim, via /virtual_odometry),
        or until the ``center_drive_max_s`` safety timeout fires (HW
        fallback / sim with stale odom).
        """
        cs = self._cell_size
        cx = self._col * cs
        cy = -self._row * cs
        ahead: Optional[float] = None
        if self._true_xy is not None:
            dx = cx - self._true_xy[0]
            dy = cy - self._true_xy[1]
            dr, dc = _HEADING_DELTA[self._heading]
            # Heading unit vector in world: (dc, -dr).
            ahead = dx * dc + dy * (-dr)
        elapsed = self._state_elapsed()
        if ahead is not None and ahead <= self._center_tol_m:
            self.get_logger().info(
                f'CENTER done at cell ({self._row},{self._col}) '
                f'ahead={ahead:+.3f}m')
            self._commit_pending_turn()
            return
        if elapsed >= self._center_drive_max_s:
            ahead_str = '%.3fm' % ahead if ahead is not None else 'n/a'
            self.get_logger().warn(
                f'CENTER timeout (ahead={ahead_str}); starting turn anyway')
            self._commit_pending_turn()
            return
        self._publish(self._center_speed, 0.0)

    def _commit_pending_turn(self) -> None:
        """Apply the deferred heading change and enter TURN."""
        delta = getattr(self, '_pending_heading_delta', 0)
        self._heading = (self._heading + delta) % 4
        self._pending_heading_delta = 0
        self._stop()
        self._transition(_State.TURN, 'centred -> spin')

    def _do_turn(self) -> None:
        """Mirror HW turn_right/turn_left: open-loop spin in place."""
        if self._state_elapsed() >= self._turn_open_loop_s:
            self._stop()
            self.get_logger().info(
                f'TURN done (yaw target reached open-loop, '
                f'heading -> {_HEADING_NAMES[self._heading]})')
            if self._camera_align and self._latest_image is not None:
                self._transition(_State.ALIGN, 'start camera align')
            else:
                why = ('camera_align disabled' if not self._camera_align
                       else 'no camera frame yet -> skip align')
                self._transition(_State.CREEP, why)
            return
        sign = getattr(self, '_turn_sign', -1.0) or -1.0
        self._publish(0.0, sign * self._turn_speed)

    def _do_align(self) -> None:
        """Rotate-in-place until camera-line angle is below tol."""
        if self._latest_image is None:
            if self._state_elapsed() >= self._align_timeout_s:
                self.get_logger().warn(
                    'ALIGN timeout without any camera frame')
                self._transition(_State.CREEP, 'align timeout (no frame)')
            return
        est = estimate_line_from_frame(
            self._latest_image, theta_offset=self._theta_offset)
        if not est.valid:
            if self._state_elapsed() >= self._align_timeout_s:
                self.get_logger().warn(
                    'ALIGN timeout: line not visible to camera')
                self._transition(_State.CREEP, 'align timeout (no line)')
                return
            # Nudge forward a hair if we cannot see the line.
            self._publish(self._creep_speed, 0.0)
            return
        if abs(est.e_theta) <= self._theta_tol:
            self._align_stable += 1
            if self._align_stable >= self._align_stable_frames:
                self.get_logger().info(
                    f'ALIGN done e_theta={math.degrees(est.e_theta):+.2f}deg '
                    f'e_x={est.e_x:+.1f}px ({self._align_stable} stable)')
                self._transition(_State.CREEP, 'aligned')
                return
            self._stop()
            return
        self._align_stable = 0
        # +e_theta -> line tilts down-right -> robot heading is left of
        # the line -> rotate right (-ω).
        omega = -math.copysign(self._align_omega, est.e_theta)
        if self.get_logger() is not None and self._state_elapsed() < 0.5:
            self.get_logger().info(
                f'ALIGN pulse e_theta={math.degrees(est.e_theta):+.2f}deg '
                f'omega={omega:+.2f}')
        self._publish(0.0, omega)
        if self._state_elapsed() >= self._align_timeout_s:
            self.get_logger().warn(
                f'ALIGN timeout |e_theta|={math.degrees(abs(est.e_theta)):.1f}'
                'deg')
            self._transition(_State.CREEP, 'align timeout')

    def _do_creep(self) -> None:
        """Forward at low speed until ``/line_pose`` is valid again.

        A turn does not change the cell. The cell estimate is *not*
        advanced here; the upcoming ``/intersection`` at the next cell
        will advance it (see :py:meth:`_on_cross`).
        """
        if self._line_valid and abs(self._line_pose) < 0.6:
            self.get_logger().info(
                f'CREEP done (estim unchanged at '
                f'({self._row},{self._col},'
                f'{_HEADING_NAMES[self._heading]}); line_pose='
                f'{self._line_pose:+.3f})')
            self._transition(_State.LINE_FOLLOW, 'creep done')
            return
        if self._state_elapsed() >= self._creep_timeout_s:
            self.get_logger().warn(
                f'CREEP timeout (estim unchanged at '
                f'({self._row},{self._col},'
                f'{_HEADING_NAMES[self._heading]}))')
            self._transition(_State.LINE_FOLLOW, 'creep timeout')
            return
        self._publish(self._creep_speed, 0.0)

    @property
    def done(self) -> bool:
        return self._state == _State.DONE


def main(args: list[str] | None = None) -> None:
    """Entry point for ``ros2 run maze_mdp gazebo_policy_runner``."""
    rclpy.init(args=args)
    node = GazeboPolicyRunner()
    try:
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        node._stop()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

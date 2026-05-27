"""
ROS-free state machine for executing discrete maze actions on the AlphaBot2.

Drives a differential-drive robot along a black-line grid:

- ``FORWARD``: line-follow with a proportional controller until the next
  intersection event (all five IR sensors on a line), then keep driving
  forward for ``pivot_creep_s`` so the wheel axle ends up over the crossing
  before stopping. This *centres* the robot on the cross before handing
  control to the next action.
- ``TURN_LEFT`` / ``TURN_RIGHT``: open-loop in-place spin at ``turn_speed``
  until the forward camera reports a new line that is *almost parallel*
  to the image vertical axis. The completion condition is purely
  geometric: the originating line must first sweep away from vertical
  (``|angle| > align_lost_threshold`` or NaN), then a new line must
  return to within ``align_aligned_threshold`` of vertical for
  ``align_debounce`` consecutive samples. ``turn_max_yaw_rad`` and the
  global ``action_timeout_s`` are the only safety bounds. Assumes the
  robot starts the turn already centred on the crossing (FORWARD's
  post-intersection creep guarantees that).

The executor is intentionally I/O-free: callers feed it events
(``on_line_pose``, ``on_intersection``, ``on_line_lost``,
``on_line_alignment``, ``on_tick``) and read back a :class:`MotorCmd`
plus an optional :class:`ActionResult`.

This keeps the algorithm trivially unit-testable with plain pytest and lets the
same state machine drive both the hardware and the Gazebo wrapper.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from maze_mdp.control.line_pid import LinePID, LinePIDConfig
from maze_mdp.mdp import Action


class _State(Enum):
    IDLE = 'idle'
    DRIVING = 'driving'      # FORWARD: line-follow until /intersection
    CROSSING = 'crossing'    # FORWARD: post-/intersection creep to centre
    TURNING = 'turning'
    APPROACHING = 'approaching'
    DONE = 'done'


# DiscreteActionGoal.DRIVE_UNTIL_MARKER constant; kept here so the
# executor stays ROS-free. Must match the value in the .msg.
DRIVE_UNTIL_MARKER: int = 3


class FailureMode(Enum):
    """Mirrors ``maze_msgs/DiscreteActionResult`` failure_mode constants."""

    NONE = 0
    LINE_LOST = 1
    TIMEOUT = 2
    COLLISION = 3
    ABORTED = 4


@dataclass(frozen=True)
class MotorCmd:
    """Differential-drive command in m/s and rad/s."""

    linear: float = 0.0
    angular: float = 0.0


STOP = MotorCmd(0.0, 0.0)


@dataclass(frozen=True)
class ActionResult:
    """One-shot outcome emitted when an action terminates."""

    goal_id: int
    action: int
    success: bool
    failure_mode: FailureMode


@dataclass(frozen=True)
class ExecutorConfig:
    """Static tuning of the state machine."""

    forward_speed: float = 0.10       # m/s along the line
    turn_speed: float = 0.60          # rad/s for in-place rotation (fast)
    # ---- Line-follow PID ----
    # ``line_p_gain`` is the proportional gain ``Kp`` (kept for backward
    # compatibility with launch files that only tune it). ``line_i_gain``
    # / ``line_d_gain`` default to 0 so the legacy P-only behaviour is
    # reproduced exactly when callers do not override them.
    line_p_gain: float = 0.8          # Kp, rad/s per unit line_pose error
    line_i_gain: float = 0.0          # Ki, rad/s per (pose * s)
    line_d_gain: float = 0.0          # Kd, rad/s per (pose / s)
    # First-order low-pass on the derivative term (seconds). Required on
    # hardware where the IR strip is noisy; ``0.0`` disables the filter.
    line_d_filter_tau: float = 0.05
    # Symmetric clamp on the integrator state and on the controller output.
    line_i_clamp: float = 0.5
    line_omega_clamp: float = 2.5     # rad/s
    action_timeout_s: float = 8.0     # global per-action timeout
    line_lost_timeout_s: float = 0.5  # forward fail after this with no line
    approach_speed: float = 0.08      # m/s while creeping toward the marker

    # ---- Centering creep ----
    # Time spent driving forward *past* the rising edge of /intersection
    # during FORWARD, so the wheel axle ends up over the crossing rather
    # than behind it (the IR strip is mounted forward of the axle).
    # Calibrate per chassis as ``strip_to_axle_distance / forward_speed``.
    pivot_creep_s: float = 0.45

    # ---- Turn (open-loop spin, camera-based completion) ----
    # The forward camera looking at the floor reports the angle (rad)
    # of the dominant black line from image-vertical. The executor
    # spins in place at ``turn_speed`` and finishes the turn as soon
    # as a new line returns to near-vertical:
    #   1. Originating line departs: a frame with ``|angle|`` greater
    #      than ``align_lost_threshold`` (or NaN, meaning the line is
    #      out of view) latches ``_align_lost_seen``. This is what
    #      stops the FSM from accepting the originating line at t=0.
    #   2. New line aligned: after departure, ``|angle|`` inside
    #      ``align_aligned_threshold`` for ``align_debounce``
    #      consecutive samples ends the turn successfully.
    # ``turn_max_yaw_rad`` is the only commanded-yaw bound: if the
    # robot spins past it without ever aligning, the turn fails.
    align_lost_threshold: float = 0.5      # rad, ~28 deg
    align_aligned_threshold: float = 0.15  # rad, ~8.6 deg
    align_debounce: int = 3
    turn_max_yaw_rad: float = 3.50         # hard safety bound


class ActionExecutor:
    """Closed-loop driver for one discrete maze action at a time."""

    def __init__(self, config: ExecutorConfig | None = None) -> None:
        self._cfg = config or ExecutorConfig()
        self._state = _State.IDLE
        self._action: int = -1
        self._goal_id: int = 0
        self._t_since_start: float = 0.0
        self._t_since_line: float = 0.0
        self._t_crossing: float = 0.0
        self._t_last_pose: float = 0.0  # for PID dt
        self._result: Optional[ActionResult] = None
        # Line-follow PID (shared between DRIVING and APPROACHING).
        self._line_pid = LinePID(LinePIDConfig(
            kp=self._cfg.line_p_gain,
            ki=self._cfg.line_i_gain,
            kd=self._cfg.line_d_gain,
            d_filter_tau=self._cfg.line_d_filter_tau,
            i_clamp=self._cfg.line_i_clamp,
            output_clamp=self._cfg.line_omega_clamp,
        ))
        # Turn bookkeeping (single-phase spin, camera-based completion).
        self._turn_direction: int = 0  # -1 left (CCW), +1 right (CW)
        self._yaw_accum: float = 0.0   # |integrated commanded omega|
        # ``_align_lost_seen`` latches True once the originating line
        # has swept out of the image-vertical reference (|angle| past
        # ``align_lost_threshold`` or NaN). Until then, aligned frames
        # are ignored -- they would otherwise immediately re-accept
        # the originating line at t=0. ``_align_streak`` debounces
        # the final acceptance; resets whenever the line drops out
        # or moves back outside the aligned window.
        self._align_lost_seen: bool = False
        self._align_streak: int = 0
        # Most recent angular command emitted by the line-follow PID. Held
        # across brief NaN samples so a transient line drop-out does not
        # zero the correction in progress (the robot was already steering
        # toward the line; keep doing that until the line returns or the
        # action fails on line_lost_timeout).
        self._last_line_omega: float = 0.0

    # ----------------------------------------------------------- public API
    @property
    def state(self) -> _State:
        return self._state

    @property
    def is_active(self) -> bool:
        return self._state in (
            _State.DRIVING, _State.CROSSING,
            _State.TURNING, _State.APPROACHING)

    def take_result(self) -> Optional[ActionResult]:
        """Return and clear any pending one-shot result."""
        r, self._result = self._result, None
        return r

    def start(self, action: int, goal_id: int) -> MotorCmd:
        """Begin executing ``action`` (``Action`` enum int)."""
        if self.is_active:
            # Pre-empt the in-flight action with an ABORTED result.
            self._finish(success=False, failure_mode=FailureMode.ABORTED)
        self._action = int(action)
        self._goal_id = int(goal_id)
        self._t_since_start = 0.0
        self._t_since_line = 0.0
        self._t_crossing = 0.0
        self._t_last_pose = 0.0
        self._line_pid.reset()
        self._yaw_accum = 0.0
        self._align_lost_seen = False
        self._align_streak = 0
        self._last_line_omega = 0.0
        if self._action == int(Action.FORWARD):
            self._state = _State.DRIVING
            return MotorCmd(self._cfg.forward_speed, 0.0)
        if self._action == int(Action.TURN_LEFT):
            self._state = _State.TURNING
            self._turn_direction = -1
            return self._turn_cmd()
        if self._action == int(Action.TURN_RIGHT):
            self._state = _State.TURNING
            self._turn_direction = +1
            return self._turn_cmd()
        if self._action == DRIVE_UNTIL_MARKER:
            self._state = _State.APPROACHING
            return MotorCmd(self._cfg.approach_speed, 0.0)
        raise ValueError(f'unknown action {action}')

    def abort(self) -> MotorCmd:
        """External pre-emption (e.g. e-stop)."""
        if self.is_active:
            self._finish(success=False, failure_mode=FailureMode.ABORTED)
        return STOP

    # ------------------------------------------------------------- events
    def on_line_pose(self, pose: float) -> MotorCmd:
        """
        Latest line position estimate from the IR strip.

        ``pose`` is in [-1, +1] (negative = line is left of centre).
        The driver should publish NaN when no line is visible.
        """
        if self._state == _State.DRIVING:
            # NaN means the IR strip currently sees no line. Do NOT zero
            # the angular command: the robot was almost certainly steering
            # toward the line, and zeroing here would let it coast off
            # tangent until line_lost_timeout fires. Instead hold the most
            # recent PID output -- the controller keeps pulling back the
            # same direction it was last correcting in. ``_t_since_line``
            # is intentionally not reset here so the line_lost grace
            # timeout still arms.
            if pose != pose:  # NaN check
                return MotorCmd(self._cfg.forward_speed,
                                self._last_line_omega)
            self._t_since_line = 0.0
            ang = self._line_pid_step(float(pose))
            self._last_line_omega = ang
            return MotorCmd(self._cfg.forward_speed, ang)

        if self._state == _State.CROSSING:
            # Strip is over the cross during this creep -- pose is
            # ambiguous (all sensors on a line). Drive straight; only the
            # creep timer ends the phase.
            return MotorCmd(self._cfg.forward_speed, 0.0)

        if self._state == _State.TURNING:
            # Turn completion is driven entirely by the forward camera
            # (``on_line_alignment``); the IR strip mid-spin is too
            # noisy and ambiguous to be useful. Keep spinning.
            return self._turn_cmd()

        if self._state == _State.APPROACHING:
            # Use the line-follow controller (with reduced speed) to stay
            # straight while we wait for the goal marker. NaN -> hold the
            # last correction so the robot still curves toward the line.
            self._t_since_line = 0.0
            if pose != pose:
                return MotorCmd(self._cfg.approach_speed,
                                self._last_line_omega)
            ang = self._line_pid_step(float(pose))
            self._last_line_omega = ang
            return MotorCmd(self._cfg.approach_speed, ang)
        return STOP

    def on_intersection(self) -> MotorCmd:
        """All five IR sensors are on a line (crossing reached)."""
        if self._state == _State.DRIVING:
            # Promote to CROSSING: keep driving forward so the axle reaches
            # the cross. Do *not* stop the robot here.
            self._state = _State.CROSSING
            self._t_crossing = 0.0
            return MotorCmd(self._cfg.forward_speed, 0.0)
        # Intersections inside CROSSING, APPROACHING or TURNING are ignored:
        # the goal cell may sit beyond one more intersection (APPROACHING),
        # we are already creeping past one (CROSSING), or the cross is
        # passing under the strip during the spin (TURNING).
        return self._current_cmd()

    def on_marker_seen(self) -> MotorCmd:
        """Goal fiducial detected at final-approach proximity."""
        if self._state == _State.APPROACHING:
            self._finish(success=True, failure_mode=FailureMode.NONE)
            return STOP
        return self._current_cmd()

    def on_line_x_offset(self, x_offset: float) -> MotorCmd:
        """Accept the line-centroid x-offset for compatibility; ignored.

        The simplified turn FSM uses only the line-tilt angle for
        completion. Centroid x-offset is not consulted.
        """
        return self._current_cmd()

    def on_line_alignment(self, angle: float) -> MotorCmd:
        """Camera-derived line-tilt angle (radians) from image-vertical.

        ``angle`` is the dominant ground-line tilt from image-vertical,
        as published by ``line_aligner`` on ``/line_alignment``.
        ``0`` means the line is vertical in the image (heading aligned).
        ``NaN`` means no usable line is in view.

        Active only during ``_State.TURNING``. The completion logic is
        a single-stage debounce gated by one latch:

          1. **Misaligned seen** (``_align_lost_seen``): latches True
             on the first frame with ``|angle| > align_lost_threshold``
             or a NaN. Without this latch, the originating line --
             which is near-vertical at t=0 -- would immediately
             satisfy the aligned predicate and the robot would never
             turn.
          2. **Aligned debounce**: once the misaligned latch is set,
             ``align_debounce`` consecutive in-band samples with
             ``|angle| <= align_aligned_threshold`` complete the turn.
             An explicitly misaligned frame
             (``|angle| > align_aligned_threshold``) resets the streak.
             NaN is treated as a transient detector dropout and holds
             the streak: a single missing frame between two aligned
             frames does not penalise convergence.
        """
        if self._state != _State.TURNING:
            return self._current_cmd()

        is_nan = angle != angle
        abs_a = abs(float(angle)) if not is_nan else 0.0

        if is_nan:
            # Transient detector dropout. Latch the misaligned flag if
            # it has not been set yet (line out of view also counts as
            # the originating line having swept away), but do not
            # touch the debounce streak: a flicker between two aligned
            # frames must not penalise convergence.
            self._align_lost_seen = True
            return self._turn_cmd()

        if abs_a > self._cfg.align_lost_threshold:
            # Originating line (or a new line still far from vertical):
            # arm the FSM and reset the debounce streak.
            self._align_lost_seen = True
            self._align_streak = 0
            return self._turn_cmd()

        # Real measurement, |angle| within align_lost_threshold.
        if not self._align_lost_seen:
            # Still the originating line. Keep spinning.
            return self._turn_cmd()

        if abs_a > self._cfg.align_aligned_threshold:
            # A new line is in view but not yet vertical enough.
            self._align_streak = 0
            return self._turn_cmd()

        self._align_streak += 1
        if self._align_streak >= self._cfg.align_debounce:
            self._finish(success=True, failure_mode=FailureMode.NONE)
            return STOP
        return self._turn_cmd()

    def on_line_lost(self) -> MotorCmd:
        """Signal that no IR sensor currently sees a line."""
        if self._state == _State.DRIVING:
            # Hold the last PID correction (see on_line_pose docstring);
            # line_lost_timeout will eventually fail us.
            return MotorCmd(self._cfg.forward_speed, self._last_line_omega)
        if self._state == _State.CROSSING:
            return MotorCmd(self._cfg.forward_speed, 0.0)
        if self._state == _State.APPROACHING:
            return MotorCmd(self._cfg.approach_speed, self._last_line_omega)
        if self._state == _State.TURNING:
            # Turn completion is camera-only; IR line loss during the
            # spin is uninformative. Keep spinning.
            return self._turn_cmd()
        return self._current_cmd()

    def on_tick(self, dt: float) -> MotorCmd:
        """
        Advance time by ``dt`` seconds; enforce action and line-loss timeouts.

        Should be called at a steady rate by the ROS wrapper.
        """
        if not self.is_active:
            return STOP
        self._t_since_start += dt

        if self._state == _State.DRIVING:
            self._t_since_line += dt
            if (self._cfg.line_lost_timeout_s > 0.0
                    and self._t_since_line
                    >= self._cfg.line_lost_timeout_s):
                self._finish(success=False,
                             failure_mode=FailureMode.LINE_LOST)
                return STOP

        if self._state == _State.CROSSING:
            self._t_crossing += dt
            if self._t_crossing >= self._cfg.pivot_creep_s:
                # Axle is now over the cross; FORWARD complete.
                self._finish(success=True, failure_mode=FailureMode.NONE)
                return STOP
            return MotorCmd(self._cfg.forward_speed, 0.0)

        if self._state == _State.TURNING:
            return self._turn_on_tick(dt)

        # APPROACHING tolerates line loss (we may overshoot the last
        # intersection on the way to the marker); only the global
        # action_timeout aborts it.
        if self._t_since_start >= self._cfg.action_timeout_s:
            self._finish(success=False, failure_mode=FailureMode.TIMEOUT)
            return STOP
        return self._current_cmd()

    # -------------------------------------------------------- turn helpers
    def _turn_on_tick(self, dt: float) -> MotorCmd:
        # Integrate commanded |omega| as a cheap safety bound.
        self._yaw_accum += self._cfg.turn_speed * dt

        if self._yaw_accum >= self._cfg.turn_max_yaw_rad:
            # Spun past the safety bound without ever aligning -> fail.
            self._finish(success=False, failure_mode=FailureMode.LINE_LOST)
            return STOP

        if self._t_since_start >= self._cfg.action_timeout_s:
            self._finish(success=False, failure_mode=FailureMode.TIMEOUT)
            return STOP
        return self._turn_cmd()

    def _turn_cmd(self) -> MotorCmd:
        # turn_direction: -1 left -> +omega; +1 right -> -omega.
        return MotorCmd(0.0, -self._turn_direction * self._cfg.turn_speed)

    # ---------------------------------------------------------- internals
    def _line_pid_step(self, pose: float) -> float:
        """Run one PID update with dt measured from the executor clock."""
        dt = self._t_since_start - self._t_last_pose
        if dt < 0.0:
            dt = 0.0
        self._t_last_pose = self._t_since_start
        return self._line_pid.step(pose, dt)

    def _current_cmd(self) -> MotorCmd:
        if self._state == _State.DRIVING:
            return MotorCmd(self._cfg.forward_speed, 0.0)
        if self._state == _State.CROSSING:
            return MotorCmd(self._cfg.forward_speed, 0.0)
        if self._state == _State.TURNING:
            return self._turn_cmd()
        if self._state == _State.APPROACHING:
            return MotorCmd(self._cfg.approach_speed, 0.0)
        return STOP

    def _finish(self, success: bool, failure_mode: FailureMode) -> None:
        self._result = ActionResult(
            goal_id=self._goal_id,
            action=self._action,
            success=success,
            failure_mode=failure_mode,
        )
        self._state = _State.DONE


__all__ = [
    'ActionExecutor',
    'ActionResult',
    'DRIVE_UNTIL_MARKER',
    'ExecutorConfig',
    'FailureMode',
    'MotorCmd',
    'STOP',
]

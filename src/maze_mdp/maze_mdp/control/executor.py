"""
ROS-free state machine for executing discrete maze actions on the AlphaBot2.

Drives a differential-drive robot along a black-line grid:

- ``FORWARD``: line-follow with a proportional controller until the next
  intersection event (all five IR sensors on a line), then keep driving
  forward for ``pivot_creep_s`` so the wheel axle ends up over the crossing
  before stopping. This *centres* the robot on the cross before handing
  control to the next action.
- ``TURN_LEFT`` / ``TURN_RIGHT``: 3-phase closed-loop spin
  (``LEAVE`` -> ``ACQUIRE`` -> ``LOCK``) using the IR strip plus the
  commanded-yaw integral as a sanity gate. Assumes the robot starts the turn
  already centred on the crossing (FORWARD's post-intersection creep
  guarantees that).

The executor is intentionally I/O-free: callers feed it events
(``on_line_pose``, ``on_intersection``, ``on_line_lost``, ``on_tick``) and read
back a :class:`MotorCmd` plus an optional :class:`ActionResult`.

This keeps the algorithm trivially unit-testable with plain pytest and lets the
same state machine drive both the hardware and the Gazebo wrapper.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from maze_mdp.mdp import Action


class _State(Enum):
    IDLE = 'idle'
    DRIVING = 'driving'      # FORWARD: line-follow until /intersection
    CROSSING = 'crossing'    # FORWARD: post-/intersection creep to centre
    TURNING = 'turning'
    APPROACHING = 'approaching'
    DONE = 'done'


class _TurnPhase(Enum):
    LEAVE = 'leave'        # spin until the strip leaves the originating line
    ACQUIRE = 'acquire'    # spin until the perpendicular line is re-acquired
    LOCK = 'lock'          # slow P-control until centred on the new line


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
    line_p_gain: float = 0.8          # rad/s per unit line_pose error
    action_timeout_s: float = 8.0     # global per-action timeout
    line_lost_timeout_s: float = 0.5  # forward fail after this with no line
    approach_speed: float = 0.08      # m/s while creeping toward the marker

    # ---- Centering creep ----
    # Time spent driving forward *past* the rising edge of /intersection
    # during FORWARD, so the wheel axle ends up over the crossing rather
    # than behind it (the IR strip is mounted forward of the axle).
    # Calibrate per chassis as ``strip_to_axle_distance / forward_speed``.
    pivot_creep_s: float = 0.45

    # ---- Turn FSM tuning ----
    # LEAVE phase exit: ``|pose|`` past which the strip is considered to have
    # left the originating line.
    turn_leave_threshold: float = 0.5
    # ACQUIRE -> LOCK gate: ``|pose|`` re-entering this window after having
    # crossed ``turn_leave_threshold`` (and after enough yaw has accrued).
    turn_acquire_threshold: float = 0.5
    # LOCK phase: angular speed = this * ``turn_speed`` (slow creep).
    turn_lock_speed_factor: float = 0.25
    # LOCK exit: ``|pose|`` window for declaring the turn complete.
    turn_lock_threshold: float = 0.15
    # LOCK exit: number of consecutive samples below ``turn_lock_threshold``
    # required before publishing success (debounce against IR noise).
    turn_lock_debounce: int = 3
    # Yaw-integral gates (using commanded omega only; cheap sanity bound):
    # do not allow LOCK before this much rotation has accumulated.
    turn_min_yaw_rad: float = 1.10   # ~0.7 * pi/2
    # Primary turn completion criterion: succeed as soon as this much
    # commanded rotation has accumulated, even if the IR strip has not
    # locked on a perpendicular line. This is the most reliable signal in
    # simulation (where /line_pose during a spin is synthetic) and a robust
    # backstop on hardware.
    turn_target_yaw_rad: float = math.pi / 2  # exactly 90 degrees
    # Hard-fail the turn (LINE_LOST) once this much rotation has accumulated.
    turn_max_yaw_rad: float = 2.50   # well above turn_target_yaw_rad


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
        self._result: Optional[ActionResult] = None
        # Turn-specific bookkeeping.
        self._turn_phase: _TurnPhase = _TurnPhase.LEAVE
        self._turn_direction: int = 0  # -1 left (CCW), +1 right (CW)
        self._yaw_accum: float = 0.0   # |integrated commanded omega|
        self._lock_streak: int = 0
        self._leave_seen: bool = False  # safety: only LOCK after LEAVE

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
        self._yaw_accum = 0.0
        self._lock_streak = 0
        self._leave_seen = False
        if self._action == int(Action.FORWARD):
            self._state = _State.DRIVING
            return MotorCmd(self._cfg.forward_speed, 0.0)
        if self._action == int(Action.TURN_LEFT):
            self._state = _State.TURNING
            self._turn_direction = -1
            self._turn_phase = _TurnPhase.LEAVE
            return self._turn_cmd()
        if self._action == int(Action.TURN_RIGHT):
            self._state = _State.TURNING
            self._turn_direction = +1
            self._turn_phase = _TurnPhase.LEAVE
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
            self._t_since_line = 0.0
            if pose != pose:  # NaN check
                return MotorCmd(self._cfg.forward_speed, 0.0)
            ang = -self._cfg.line_p_gain * float(pose)
            return MotorCmd(self._cfg.forward_speed, ang)

        if self._state == _State.CROSSING:
            # Strip is over the cross during this creep -- pose is
            # ambiguous (all sensors on a line). Drive straight; only the
            # creep timer ends the phase.
            return MotorCmd(self._cfg.forward_speed, 0.0)

        if self._state == _State.TURNING:
            return self._turn_on_pose(pose)

        if self._state == _State.APPROACHING:
            # Use the line-follow controller (with reduced speed) to stay
            # straight while we wait for the goal marker. NaN -> coast.
            self._t_since_line = 0.0
            if pose != pose:
                return MotorCmd(self._cfg.approach_speed, 0.0)
            ang = -self._cfg.line_p_gain * float(pose)
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

    def on_line_lost(self) -> MotorCmd:
        """Signal that no IR sensor currently sees a line."""
        if self._state == _State.DRIVING:
            # Coast straight; line_lost_timeout will eventually fail us.
            return MotorCmd(self._cfg.forward_speed, 0.0)
        if self._state == _State.CROSSING:
            return MotorCmd(self._cfg.forward_speed, 0.0)
        if self._state == _State.APPROACHING:
            return MotorCmd(self._cfg.approach_speed, 0.0)
        if self._state == _State.TURNING:
            # During LEAVE, losing the line is exactly the signal we are
            # waiting for. Promote to ACQUIRE immediately so we do not wait
            # for an unattainable |pose| threshold.
            if self._turn_phase == _TurnPhase.LEAVE:
                self._leave_seen = True
                self._turn_phase = _TurnPhase.ACQUIRE
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
            if self._t_since_line >= self._cfg.line_lost_timeout_s:
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
    def _turn_on_pose(self, pose: float) -> MotorCmd:
        if self._turn_phase == _TurnPhase.LEAVE:
            if pose != pose:  # NaN -> treat as line lost
                self._leave_seen = True
                self._turn_phase = _TurnPhase.ACQUIRE
                return self._turn_cmd()
            if abs(pose) >= self._cfg.turn_leave_threshold:
                self._leave_seen = True
                self._turn_phase = _TurnPhase.ACQUIRE
            return self._turn_cmd()

        if self._turn_phase == _TurnPhase.ACQUIRE:
            if pose != pose:
                return self._turn_cmd()
            if not (self._leave_seen
                    and abs(pose) <= self._cfg.turn_acquire_threshold
                    and self._yaw_accum >= self._cfg.turn_min_yaw_rad):
                return self._turn_cmd()
            self._turn_phase = _TurnPhase.LOCK
            self._lock_streak = 0
            # Fall through to LOCK handling with this same sample.

        if self._turn_phase == _TurnPhase.LOCK:
            if pose != pose:
                self._lock_streak = 0
                return self._turn_cmd(lock=True)
            if abs(pose) < self._cfg.turn_lock_threshold:
                self._lock_streak += 1
                if self._lock_streak >= self._cfg.turn_lock_debounce:
                    self._finish(success=True,
                                 failure_mode=FailureMode.NONE)
                    return STOP
            else:
                self._lock_streak = 0
            lock_w = self._cfg.turn_speed * self._cfg.turn_lock_speed_factor
            ang = -self._cfg.line_p_gain * float(pose)
            if ang > lock_w:
                ang = lock_w
            elif ang < -lock_w:
                ang = -lock_w
            return MotorCmd(0.0, ang)

        return self._turn_cmd()

    def _turn_on_tick(self, dt: float) -> MotorCmd:
        # Integrate |commanded omega| only while actually spinning.
        if self._turn_phase in (_TurnPhase.LEAVE, _TurnPhase.ACQUIRE):
            self._yaw_accum += self._cfg.turn_speed * dt
        elif self._turn_phase == _TurnPhase.LOCK:
            self._yaw_accum += (
                self._cfg.turn_speed * self._cfg.turn_lock_speed_factor * dt)

        # Primary completion: enough commanded rotation has accrued.
        if self._yaw_accum >= self._cfg.turn_target_yaw_rad:
            self._finish(success=True, failure_mode=FailureMode.NONE)
            return STOP

        if self._yaw_accum >= self._cfg.turn_max_yaw_rad:
            # Spun past the safety bound without locking on -> fail.
            self._finish(success=False, failure_mode=FailureMode.LINE_LOST)
            return STOP

        if self._t_since_start >= self._cfg.action_timeout_s:
            self._finish(success=False, failure_mode=FailureMode.TIMEOUT)
            return STOP
        return self._turn_cmd()

    def _turn_cmd(self, lock: bool = False) -> MotorCmd:
        w = self._cfg.turn_speed
        if lock or self._turn_phase == _TurnPhase.LOCK:
            w *= self._cfg.turn_lock_speed_factor
        # turn_direction: -1 left -> +omega; +1 right -> -omega.
        return MotorCmd(0.0, -self._turn_direction * w)

    # ---------------------------------------------------------- internals
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

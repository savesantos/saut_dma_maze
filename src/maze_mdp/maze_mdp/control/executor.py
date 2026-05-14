"""
ROS-free state machine for executing discrete maze actions on the AlphaBot2.

Drives a differential-drive robot along a black-line grid:

- ``FORWARD``: line-follow with a proportional controller until the next
  intersection event (all five IR sensors on a line).
- ``TURN_LEFT`` / ``TURN_RIGHT``: rotate in place until a perpendicular line
  is acquired again under the centre sensor.

The executor is intentionally I/O-free: callers feed it events
(``on_line_pose``, ``on_intersection``, ``on_line_lost``, ``on_tick``) and read
back a :class:`MotorCmd` plus an optional :class:`ActionResult`.

This keeps the algorithm trivially unit-testable with plain pytest and lets the
same state machine drive both the hardware and the Gazebo wrapper.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from maze_mdp.mdp import Action


class _State(Enum):
    IDLE = 'idle'
    DRIVING = 'driving'
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
    turn_speed: float = 0.60          # rad/s for in-place rotation
    line_p_gain: float = 0.8          # rad/s per unit line_pose error
    action_timeout_s: float = 8.0     # global per-action timeout
    line_lost_timeout_s: float = 0.5  # forward fail after this with no line
    turn_exit_pose: float = 0.20      # |pose| < this locks the turn
    turn_exit_min_excursion: float = 0.5  # ... after first exceeding this
    approach_speed: float = 0.08      # m/s while creeping toward the marker


class ActionExecutor:
    """Closed-loop driver for one discrete maze action at a time."""

    def __init__(self, config: ExecutorConfig | None = None) -> None:
        self._cfg = config or ExecutorConfig()
        self._state = _State.IDLE
        self._action: int = -1
        self._goal_id: int = 0
        self._t_since_start: float = 0.0
        self._t_since_line: float = 0.0
        self._result: Optional[ActionResult] = None
        # Turn state: have we yet seen the strip leave the centre line?
        self._turn_excursion_seen: bool = False
        self._turn_direction: int = 0  # -1 left, +1 right

    # ----------------------------------------------------------- public API
    @property
    def state(self) -> _State:
        return self._state

    @property
    def is_active(self) -> bool:
        return self._state in (
            _State.DRIVING, _State.TURNING, _State.APPROACHING)

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
        self._turn_excursion_seen = False
        if self._action == int(Action.FORWARD):
            self._state = _State.DRIVING
            return MotorCmd(self._cfg.forward_speed, 0.0)
        if self._action == int(Action.TURN_LEFT):
            self._state = _State.TURNING
            self._turn_direction = -1
            return MotorCmd(0.0, +self._cfg.turn_speed)
        if self._action == int(Action.TURN_RIGHT):
            self._state = _State.TURNING
            self._turn_direction = +1
            return MotorCmd(0.0, -self._cfg.turn_speed)
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
        if self._state == _State.TURNING:
            if pose != pose:  # NaN -> still off the line, keep turning
                return self._turn_cmd()
            if abs(pose) >= self._cfg.turn_exit_min_excursion:
                self._turn_excursion_seen = True
            if (self._turn_excursion_seen
                    and abs(pose) < self._cfg.turn_exit_pose):
                self._finish(success=True, failure_mode=FailureMode.NONE)
                return STOP
            return self._turn_cmd()
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
            self._finish(success=True, failure_mode=FailureMode.NONE)
            return STOP
        # Intersections inside an APPROACHING phase are ignored: the
        # goal cell may sit beyond one more intersection, and only the
        # marker decides when to stop.
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
        if self._state == _State.APPROACHING:
            return MotorCmd(self._cfg.approach_speed, 0.0)
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
        # APPROACHING tolerates line loss (we may overshoot the last
        # intersection on the way to the marker); only the global
        # action_timeout aborts it.
        if self._t_since_start >= self._cfg.action_timeout_s:
            self._finish(success=False, failure_mode=FailureMode.TIMEOUT)
            return STOP
        return self._current_cmd()

    # ---------------------------------------------------------- internals
    def _turn_cmd(self) -> MotorCmd:
        return MotorCmd(0.0, -self._turn_direction * self._cfg.turn_speed)

    def _current_cmd(self) -> MotorCmd:
        if self._state == _State.DRIVING:
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

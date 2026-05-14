"""Unit tests for the ROS-free executor state machine."""

import math

import pytest

from maze_mdp.mdp import Action
from maze_mdp.control.executor import (
    ActionExecutor,
    DRIVE_UNTIL_MARKER,
    ExecutorConfig,
    FailureMode,
)


def _exec(**overrides):
    defaults = dict(
        forward_speed=0.1,
        turn_speed=0.6,
        line_p_gain=0.8,
        action_timeout_s=2.0,
        line_lost_timeout_s=0.5,
        turn_exit_pose=0.2,
        turn_exit_min_excursion=0.5,
    )
    defaults.update(overrides)
    return ActionExecutor(ExecutorConfig(**defaults))


# -------------------------------------------------------------- FORWARD path
def test_forward_starts_driving_straight():
    e = _exec()
    cmd = e.start(int(Action.FORWARD), goal_id=1)
    assert cmd.linear > 0 and cmd.angular == 0
    assert e.is_active


def test_forward_proportional_steer_left_when_line_is_right():
    e = _exec()
    e.start(int(Action.FORWARD), goal_id=1)
    cmd = e.on_line_pose(+0.5)  # line is to the right of centre
    # Positive pose -> negative angular (turn right toward the line).
    assert cmd.angular < 0
    assert cmd.linear > 0


def test_forward_succeeds_on_intersection():
    e = _exec()
    e.start(int(Action.FORWARD), goal_id=42)
    cmd = e.on_intersection()
    assert cmd.linear == 0 and cmd.angular == 0
    r = e.take_result()
    assert r is not None
    assert r.success and r.goal_id == 42
    assert r.failure_mode == FailureMode.NONE
    assert not e.is_active


def test_forward_fails_on_line_lost_timeout():
    e = _exec(line_lost_timeout_s=0.5)
    e.start(int(Action.FORWARD), goal_id=7)
    # No on_line_pose callbacks at all -> _t_since_line accumulates via ticks.
    e.on_tick(0.4)
    assert e.take_result() is None
    e.on_tick(0.2)
    r = e.take_result()
    assert r is not None and not r.success
    assert r.failure_mode == FailureMode.LINE_LOST


def test_forward_line_pose_resets_lost_timer():
    e = _exec(line_lost_timeout_s=0.5)
    e.start(int(Action.FORWARD), goal_id=1)
    e.on_tick(0.4)
    e.on_line_pose(0.0)  # line back -> timer reset
    e.on_tick(0.4)
    assert e.take_result() is None


def test_action_timeout_fires():
    e = _exec(action_timeout_s=1.0, line_lost_timeout_s=10.0)
    e.start(int(Action.FORWARD), goal_id=3)
    for _ in range(11):
        e.on_line_pose(0.0)
        e.on_tick(0.1)
    r = e.take_result()
    assert r is not None and not r.success
    assert r.failure_mode == FailureMode.TIMEOUT


# ----------------------------------------------------------- TURN_LEFT path
def test_turn_left_initial_command_is_ccw():
    e = _exec()
    cmd = e.start(int(Action.TURN_LEFT), goal_id=1)
    # CCW rotation = positive angular in ROS REP-103.
    assert cmd.angular > 0 and cmd.linear == 0


def test_turn_right_initial_command_is_cw():
    e = _exec()
    cmd = e.start(int(Action.TURN_RIGHT), goal_id=1)
    assert cmd.angular < 0 and cmd.linear == 0


def test_turn_completes_after_line_excursion_and_return():
    e = _exec()
    e.start(int(Action.TURN_LEFT), goal_id=9)
    # The strip first leaves the centre (line goes off to one side)...
    e.on_line_pose(0.8)
    assert e.is_active
    # ...then the perpendicular line slides under the centre sensor.
    cmd = e.on_line_pose(0.1)
    assert cmd.linear == 0 and cmd.angular == 0
    r = e.take_result()
    assert r is not None and r.success and r.goal_id == 9
    assert not e.is_active


def test_turn_does_not_complete_without_initial_excursion():
    e = _exec()
    e.start(int(Action.TURN_LEFT), goal_id=1)
    # Immediately seeing pose near zero must not trigger completion:
    # the strip might just be still on the original line.
    e.on_line_pose(0.05)
    assert e.is_active
    assert e.take_result() is None


def test_turn_handles_nan_line_pose():
    e = _exec()
    e.start(int(Action.TURN_LEFT), goal_id=1)
    cmd = e.on_line_pose(math.nan)
    # Still rotating; not yet complete.
    assert cmd.angular > 0
    assert e.is_active


# ------------------------------------------------------------------- misc
def test_starting_new_action_pre_empts_active_one():
    e = _exec()
    e.start(int(Action.FORWARD), goal_id=1)
    e.start(int(Action.TURN_LEFT), goal_id=2)
    r = e.take_result()
    # First action aborted, then the second one started fresh.
    assert r is not None and not r.success
    assert r.failure_mode == FailureMode.ABORTED
    assert r.goal_id == 1
    assert e.is_active


def test_abort_emits_aborted_result():
    e = _exec()
    e.start(int(Action.FORWARD), goal_id=5)
    cmd = e.abort()
    assert cmd.linear == 0 and cmd.angular == 0
    r = e.take_result()
    assert r is not None and not r.success
    assert r.failure_mode == FailureMode.ABORTED


def test_unknown_action_raises():
    e = _exec()
    with pytest.raises(ValueError):
        e.start(99, goal_id=1)


def test_idle_events_are_noops():
    e = _exec()
    assert e.on_tick(0.1).linear == 0 and e.on_tick(0.1).angular == 0
    assert e.on_line_pose(0.5).linear == 0
    assert e.on_intersection().linear == 0
    assert e.take_result() is None


# ----------------------------------------------------- DRIVE_UNTIL_MARKER
def test_approach_starts_creeping_forward():
    e = _exec()
    cmd = e.start(DRIVE_UNTIL_MARKER, goal_id=11)
    assert cmd.linear > 0 and cmd.angular == 0
    assert e.is_active


def test_approach_completes_on_marker_seen():
    e = _exec()
    e.start(DRIVE_UNTIL_MARKER, goal_id=11)
    cmd = e.on_marker_seen()
    assert cmd.linear == 0 and cmd.angular == 0
    r = e.take_result()
    assert r is not None and r.success and r.goal_id == 11
    assert r.failure_mode == FailureMode.NONE
    assert not e.is_active


def test_approach_ignores_intersection():
    e = _exec()
    e.start(DRIVE_UNTIL_MARKER, goal_id=12)
    e.on_intersection()
    assert e.is_active
    assert e.take_result() is None


def test_approach_line_pose_steers_proportionally():
    e = _exec()
    e.start(DRIVE_UNTIL_MARKER, goal_id=13)
    cmd = e.on_line_pose(+0.5)
    assert cmd.linear > 0 and cmd.angular < 0


def test_approach_tolerates_line_lost():
    e = _exec(line_lost_timeout_s=0.2, action_timeout_s=2.0)
    e.start(DRIVE_UNTIL_MARKER, goal_id=14)
    # No line for longer than line_lost_timeout_s: must NOT abort.
    e.on_tick(0.5)
    assert e.is_active
    assert e.take_result() is None


def test_approach_action_timeout_still_aborts():
    e = _exec(action_timeout_s=0.5, line_lost_timeout_s=10.0)
    e.start(DRIVE_UNTIL_MARKER, goal_id=15)
    for _ in range(6):
        e.on_tick(0.1)
    r = e.take_result()
    assert r is not None and not r.success
    assert r.failure_mode == FailureMode.TIMEOUT


def test_marker_seen_outside_approach_is_noop():
    e = _exec()
    e.start(int(Action.FORWARD), goal_id=1)
    e.on_marker_seen()
    assert e.is_active
    assert e.take_result() is None

"""Unit tests for the ROS-free executor state machine."""

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
        action_timeout_s=5.0,
        line_lost_timeout_s=0.5,
        # Tiny creep so a handful of ticks finishes FORWARD.
        pivot_creep_s=0.10,
        align_lost_threshold=0.5,
        align_aligned_threshold=0.15,
        align_debounce=3,
        turn_max_yaw_rad=12.0,
    )
    defaults.update(overrides)
    return ActionExecutor(ExecutorConfig(**defaults))


def _accumulate_yaw(e, dt=0.05, n=12):
    """Tick ``n`` times of ``dt`` to satisfy the turn yaw safety bound."""
    for _ in range(n):
        e.on_tick(dt)


# -------------------------------------------------------------- FORWARD path
def test_forward_starts_driving_straight():
    e = _exec()
    cmd = e.start(int(Action.FORWARD), goal_id=1)
    assert cmd.linear > 0 and cmd.angular == 0
    assert e.is_active


def test_forward_proportional_steer_left_when_line_is_right():
    e = _exec()
    e.start(int(Action.FORWARD), goal_id=1)
    cmd = e.on_line_pose(+0.5)
    assert cmd.angular < 0
    assert cmd.linear > 0


def test_forward_intersection_starts_crossing_creep_not_stop():
    """Strip on the cross -> keep driving forward, do NOT stop yet."""
    e = _exec()
    e.start(int(Action.FORWARD), goal_id=42)
    cmd = e.on_intersection()
    # Robot must keep moving forward through the cross to centre the axle.
    assert cmd.linear > 0 and cmd.angular == 0
    assert e.is_active
    assert e.take_result() is None


def test_forward_crossing_ignores_pose_and_drives_straight():
    """During CROSSING the pose signal is ambiguous; steer straight."""
    e = _exec()
    e.start(int(Action.FORWARD), goal_id=1)
    e.on_intersection()
    cmd = e.on_line_pose(0.8)  # would normally steer hard
    assert cmd.linear > 0 and cmd.angular == 0


def test_forward_completes_after_crossing_creep_elapses():
    e = _exec(pivot_creep_s=0.10)
    e.start(int(Action.FORWARD), goal_id=42)
    e.on_intersection()
    # Halfway through the creep -> not done.
    e.on_tick(0.05)
    assert e.is_active
    assert e.take_result() is None
    # Past the creep -> FORWARD finishes successfully (now centred).
    e.on_tick(0.10)
    r = e.take_result()
    assert r is not None and r.success and r.goal_id == 42
    assert r.failure_mode == FailureMode.NONE
    assert not e.is_active


def test_forward_fails_on_line_lost_timeout():
    e = _exec(line_lost_timeout_s=0.5)
    e.start(int(Action.FORWARD), goal_id=7)
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
    e.on_line_pose(0.0)
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


# ---------------------------------------------------------- TURN: completion
def test_turn_finishes_when_new_line_returns_to_vertical():
    """Originating line departs -> new aligned line arrives -> turn ends."""
    e = _exec(align_debounce=3)
    e.start(int(Action.TURN_LEFT), goal_id=9)
    # Originating line tilts away (|angle| > align_lost_threshold).
    e.on_line_alignment(0.9)
    # New line arrives near vertical: needs ``align_debounce`` in a row.
    e.on_line_alignment(0.05)
    assert e.take_result() is None
    e.on_line_alignment(0.05)
    assert e.take_result() is None
    e.on_line_alignment(0.05)
    r = e.take_result()
    assert r is not None and r.success and r.goal_id == 9
    assert r.failure_mode == FailureMode.NONE
    assert not e.is_active


def test_turn_does_not_complete_without_departure():
    """Aligned frames before the line has ever departed must not finish."""
    e = _exec()
    e.start(int(Action.TURN_LEFT), goal_id=1)
    for _ in range(10):
        e.on_line_alignment(0.0)
        e.on_tick(0.05)
    assert e.is_active
    assert e.take_result() is None


def test_turn_nan_alignment_counts_as_departure():
    e = _exec(align_debounce=2)
    e.start(int(Action.TURN_RIGHT), goal_id=2)
    e.on_line_alignment(float('nan'))
    e.on_line_alignment(0.05)
    e.on_line_alignment(0.05)
    r = e.take_result()
    assert r is not None and r.success


def test_turn_misaligned_resets_debounce_streak():
    e = _exec(align_debounce=3)
    e.start(int(Action.TURN_LEFT), goal_id=3)
    e.on_line_alignment(0.9)        # depart
    e.on_line_alignment(0.05)       # streak = 1
    e.on_line_alignment(0.05)       # streak = 2
    e.on_line_alignment(0.9)        # reset
    e.on_line_alignment(0.05)       # streak = 1 again
    e.on_line_alignment(0.05)       # streak = 2 again
    assert e.is_active
    assert e.take_result() is None


def test_turn_ignores_line_pose():
    """The IR strip must not be consulted during the spin."""
    e = _exec()
    e.start(int(Action.TURN_LEFT), goal_id=1)
    # Feed IR pose right on the line; should keep spinning.
    for _ in range(10):
        e.on_line_pose(0.0)
        e.on_tick(0.05)
    assert e.is_active


def test_turn_left_starts_spinning_immediately_ccw():
    e = _exec()
    cmd = e.start(int(Action.TURN_LEFT), goal_id=1)
    assert cmd.linear == 0
    assert cmd.angular > 0  # CCW
    assert e.is_active


def test_turn_right_starts_spinning_immediately_cw():
    e = _exec()
    cmd = e.start(int(Action.TURN_RIGHT), goal_id=1)
    assert cmd.linear == 0
    assert cmd.angular < 0


def test_turn_hard_fails_after_max_yaw():
    e = _exec(turn_max_yaw_rad=0.30, action_timeout_s=10.0)
    e.start(int(Action.TURN_LEFT), goal_id=1)
    for _ in range(20):
        e.on_tick(0.05)
    r = e.take_result()
    assert r is not None and not r.success
    assert r.failure_mode == FailureMode.LINE_LOST


# ------------------------------------------------------------------- misc
def test_starting_new_action_pre_empts_active_one():
    e = _exec()
    e.start(int(Action.FORWARD), goal_id=1)
    e.start(int(Action.TURN_LEFT), goal_id=2)
    r = e.take_result()
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

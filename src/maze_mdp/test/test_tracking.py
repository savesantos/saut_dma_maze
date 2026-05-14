"""Unit tests for the ROS-free :mod:`maze_mdp.tracking` dead-reckoner."""

import pytest

from maze_mdp.mdp import Action, Heading
from maze_mdp.tracking import CellPose, CellTracker


def _tracker(row=1, col=1, heading=Heading.N, rows=3, cols=3):
    return CellTracker(rows, cols, CellPose(row, col, int(heading)))


def test_initial_pose_is_returned():
    t = _tracker(2, 0, Heading.E)
    assert t.pose == CellPose(2, 0, int(Heading.E))


def test_forward_success_moves_one_cell_north():
    t = _tracker(2, 1, Heading.N)
    pose = t.apply(int(Action.FORWARD), success=True)
    assert pose == CellPose(1, 1, int(Heading.N))


def test_forward_success_moves_one_cell_east():
    t = _tracker(1, 0, Heading.E)
    pose = t.apply(int(Action.FORWARD), success=True)
    assert pose == CellPose(1, 1, int(Heading.E))


def test_forward_failure_keeps_pose():
    t = _tracker(1, 1, Heading.S)
    before = t.pose
    pose = t.apply(int(Action.FORWARD), success=False)
    assert pose == before


def test_turn_left_rotates_ccw():
    t = _tracker(0, 0, Heading.N)
    assert t.apply(int(Action.TURN_LEFT), True) \
        == CellPose(0, 0, int(Heading.W))
    assert t.apply(int(Action.TURN_LEFT), True) \
        == CellPose(0, 0, int(Heading.S))


def test_turn_right_rotates_cw():
    t = _tracker(0, 0, Heading.N)
    assert t.apply(int(Action.TURN_RIGHT), True) \
        == CellPose(0, 0, int(Heading.E))
    assert t.apply(int(Action.TURN_RIGHT), True) \
        == CellPose(0, 0, int(Heading.S))


def test_turn_failure_keeps_pose():
    t = _tracker(0, 0, Heading.N)
    pose = t.apply(int(Action.TURN_LEFT), success=False)
    assert pose == CellPose(0, 0, int(Heading.N))


def test_forward_off_grid_is_clamped():
    t = _tracker(0, 1, Heading.N, rows=3, cols=3)
    # Even if the executor erroneously reports success,
    # we must not leave the grid.
    pose = t.apply(int(Action.FORWARD), success=True)
    assert pose == CellPose(0, 1, int(Heading.N))


def test_reset_changes_pose():
    t = _tracker(0, 0, Heading.N)
    t.reset(CellPose(2, 2, int(Heading.W)))
    assert t.pose == CellPose(2, 2, int(Heading.W))


def test_reset_validates_bounds():
    t = _tracker(0, 0, Heading.N, rows=3, cols=3)
    with pytest.raises(ValueError):
        t.reset(CellPose(3, 0, int(Heading.N)))


def test_invalid_heading_in_pose():
    with pytest.raises(ValueError):
        CellPose(0, 0, 4)


def test_invalid_action_raises():
    t = _tracker()
    with pytest.raises(ValueError):
        t.apply(99, success=True)


def test_drive_until_marker_advances_like_forward():
    t = _tracker(2, 1, Heading.N)
    # DiscreteActionGoal.DRIVE_UNTIL_MARKER == 3
    pose = t.apply(3, success=True)
    assert pose == CellPose(1, 1, int(Heading.N))


def test_drive_until_marker_failure_keeps_pose():
    t = _tracker(2, 1, Heading.N)
    pose = t.apply(3, success=False)
    assert pose == CellPose(2, 1, int(Heading.N))


def test_constructor_rejects_out_of_bounds_start():
    with pytest.raises(ValueError):
        CellTracker(3, 3, CellPose(3, 0, int(Heading.N)))


def test_constructor_rejects_nonpositive_dims():
    with pytest.raises(ValueError):
        CellTracker(0, 3, CellPose(0, 0, int(Heading.N)))


def test_full_loop_returns_to_start():
    t = _tracker(1, 1, Heading.N, rows=3, cols=3)
    # square loop: N forward, turn right, E forward, turn right, S forward,
    # turn right, W forward, turn right -> back at start same heading
    seq = [
        Action.FORWARD, Action.TURN_RIGHT,
        Action.FORWARD, Action.TURN_RIGHT,
        Action.FORWARD, Action.TURN_RIGHT,
        Action.FORWARD, Action.TURN_RIGHT,
    ]
    for a in seq:
        t.apply(int(a), success=True)
    assert t.pose == CellPose(1, 1, int(Heading.N))

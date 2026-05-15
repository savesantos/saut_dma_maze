"""Unit tests for the analytic IR/marker estimator (ROS-free)."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from maze_mdp.analysis.maze_to_sdf import (
    MazeSpec,
    _line_segments,
    cell_center,
)
from maze_mdp.control.ir_geom import (
    IRGeomConfig,
    estimate_line_pose,
    goal_marker_visible,
    perpendicular_crossings,
)


CS = 0.20


def _open_3x3_segments():
    spec = MazeSpec(
        rows=3, cols=3,
        walkable=[[True] * 3 for _ in range(3)],
        goal=(2, 2),
    )
    return spec, _line_segments(spec, CS)


# ---------------------------------------------------------------- line_pose


def test_line_pose_centered_on_horizontal_segment():
    """Robot exactly on the East-going line between (0,0) and (0,1)."""
    _, segs = _open_3x3_segments()
    cfg = IRGeomConfig(cell_size=CS)
    # Midpoint of segment between cells (0,0) and (0,1) is (0.10, 0.0).
    pose = estimate_line_pose(0.10, 0.0, 0.0, segs, cfg)
    assert pose is not None
    assert abs(pose) < 1e-6


def test_line_pose_off_to_left_returns_positive_pose_when_facing_east():
    """
    Robot at (0.10, +0.02) heading East (line is at y=0, i.e. *to the right*
    in robot frame). Sign convention: positive => line right.
    """
    _, segs = _open_3x3_segments()
    cfg = IRGeomConfig(cell_size=CS, line_capture_width=0.04)
    pose = estimate_line_pose(0.10, 0.02, 0.0, segs, cfg)
    assert pose is not None
    # Robot is north of the line (world +y); line is to the robot's right
    # (robot-frame -y). Sign convention: pose > 0 -> line right -> +0.5.
    assert pose == pytest.approx(0.5, abs=1e-6)


def test_line_pose_lost_when_lateral_too_large():
    _, segs = _open_3x3_segments()
    cfg = IRGeomConfig(cell_size=CS, line_lost_threshold=0.05)
    pose = estimate_line_pose(0.10, 0.10, 0.0, segs, cfg)
    assert pose is None


def test_line_pose_clipped_to_unit():
    _, segs = _open_3x3_segments()
    cfg = IRGeomConfig(cell_size=CS, line_capture_width=0.01,
                       line_lost_threshold=0.05)
    pose = estimate_line_pose(0.10, -0.04, 0.0, segs, cfg)
    assert pose is not None
    # Robot south of line -> line to the robot's left -> negative, clipped.
    assert pose == -1.0


def test_line_pose_sign_invariant_to_driving_direction():
    """
    Driving West along the same segment with the line slightly to the
    *right* (in robot frame) must still yield a positive pose.
    """
    _, segs = _open_3x3_segments()
    cfg = IRGeomConfig(cell_size=CS, line_capture_width=0.04)
    # East-facing, line at y=0, robot at y=+0.02 -> line is to the right
    # of robot (robot frame y<0) -> pose > 0.
    pose_e = estimate_line_pose(0.10, 0.02, 0.0, segs, cfg)
    # West-facing, robot at y=-0.02 -> line still to the right -> pose > 0.
    pose_w = estimate_line_pose(0.10, -0.02, math.pi, segs, cfg)
    assert pose_e is not None and pose_w is not None
    assert pose_e > 0 and pose_w > 0
    assert pose_e == pytest.approx(pose_w, abs=1e-6)


# ----------------------------------------------------------- intersection


def test_perpendicular_crossing_fires_at_cell_center():
    """
    Robot heading East, sitting exactly at cell (1,1) center. There are
    vertical (N-S) segments to its north and south whose centerlines pass
    *through* (and end at) this point. With a non-zero intersection radius
    the perpendicular detector should fire on at least one of them.
    """
    _, segs = _open_3x3_segments()
    cfg = IRGeomConfig(cell_size=CS, intersection_radius=0.04)
    cx, cy = cell_center(1, 1, CS)  # (0.20, -0.20)
    hits = perpendicular_crossings(cx, cy, 0.0, segs, cfg)
    assert len(hits) >= 1


def test_no_intersection_in_middle_of_corridor():
    _, segs = _open_3x3_segments()
    cfg = IRGeomConfig(cell_size=CS, intersection_radius=0.03)
    # 1/4 of the way between (0,0) and (0,1) on the horizontal segment.
    hits = perpendicular_crossings(0.05, 0.0, 0.0, segs, cfg)
    assert hits == []


# ------------------------------------------------------------- goal marker


def test_goal_marker_visible_when_near_and_facing():
    cfg = IRGeomConfig(marker_proximity_m=0.10, marker_facing_tol=0.6)
    # Robot at (0.35, 0) facing East, goal at (0.40, 0.0).
    assert goal_marker_visible(0.35, 0.0, 0.0, (0.40, 0.0), cfg) is True


def test_goal_marker_invisible_when_far():
    cfg = IRGeomConfig(marker_proximity_m=0.10, marker_facing_tol=0.6)
    assert goal_marker_visible(0.0, 0.0, 0.0, (0.40, 0.0), cfg) is False


def test_goal_marker_invisible_when_facing_wrong_way():
    cfg = IRGeomConfig(marker_proximity_m=0.10, marker_facing_tol=0.4)
    # Within proximity, but facing West (yaw=pi) while goal is East.
    assert goal_marker_visible(0.35, 0.0, math.pi, (0.40, 0.0), cfg) is False


# ------------------------------------------------------- real fixture sanity


def test_real_fixture_3x3_loads_and_estimates():
    here = Path(__file__).resolve().parents[2]
    fix = here / 'maze_bringup' / 'config' / 'mazes' / 'fixture_3x3.yaml'
    spec = MazeSpec.from_yaml(fix)
    segs = _line_segments(spec, CS)
    cfg = IRGeomConfig(cell_size=CS)
    # Start at cell (0,0) facing East; should be on a line.
    pose = estimate_line_pose(0.0, 0.0, 0.0, segs, cfg)
    assert pose is not None
    # Goal at (2,2); marker should be visible from one cell back facing East.
    gx, gy = cell_center(spec.goal[0], spec.goal[1], CS)
    seen = goal_marker_visible(gx - 0.05, gy, 0.0, (gx, gy), cfg)
    assert seen is True

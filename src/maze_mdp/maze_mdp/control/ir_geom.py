"""
ROS-free analytic IR/marker estimator for the Gazebo maze simulator.

Given the maze geometry (the same line segments emitted by
``maze_to_sdf._line_segments``) and the robot pose ``(x, y, yaw)``, computes
what the AlphaBot2's downward IR strip and goal-marker camera *would*
observe, so that the Gazebo wrapper can publish ``/line_pose``,
``/intersection``, ``/line_lost`` and ``/goal_marker_seen`` without sampling
ground textures or running ArUco vision.

World frame convention (same as :mod:`maze_mdp.analysis.maze_to_sdf`):

    x = col * cell_size
    y = -row * cell_size
    yaw = 0 -> heading East (+x).

Sign convention for ``line_pose`` matches the executor's expectation:

    angular_z = -line_p_gain * line_pose

so positive ``line_pose`` means *the line is to the right of the robot* (the
robot needs to turn clockwise to recentre), i.e. ``line_pose = -lat / W`` where
``lat`` is the line's y-coordinate in the robot frame and ``W`` is the half
capture width.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple


# A segment is (cx, cy, length, yaw) in world coords; same shape as the
# tuples returned by ``maze_to_sdf._line_segments``.
Segment = Tuple[float, float, float, float]


@dataclass(frozen=True)
class IRGeomConfig:
    """Tuning constants for the analytic IR/marker estimator."""

    cell_size: float = 0.20
    # Half-width of the physical IR strip on the floor; |lat|=W -> |pose|=1.
    line_capture_width: float = 0.04
    # |lat| beyond this -> we report the line as "lost" (NaN line_pose).
    line_lost_threshold: float = 0.06
    # Robot heading must be within this of a segment's tangent to count as
    # "tracking" that segment (and contribute to line_pose).
    parallel_angle_tol: float = 0.7      # rad (~40 deg)
    # Robot heading must be within this of perpendicular for an /intersection
    # to fire when the segment passes under the robot.
    perp_angle_tol: float = 0.7
    # Robot center must come within this of a segment's centerline to count
    # the perpendicular crossing as an intersection.
    intersection_radius: float = 0.04
    # Goal marker considered visible when within this of the goal cell centre
    # AND facing it (angle to goal within marker_facing_tol).
    marker_proximity_m: float = 0.10
    marker_facing_tol: float = 0.6       # rad


# ---------------------------------------------------------------- primitives


def _wrap(angle: float) -> float:
    """Wrap to [-pi, pi]."""
    return math.atan2(math.sin(angle), math.cos(angle))


def _signed_min_angle(a: float, b: float) -> float:
    """Smallest signed angle that takes ``a`` to ``b``."""
    return _wrap(b - a)


def _project_into_segment_frame(rx: float, ry: float, seg: Segment
                                ) -> Tuple[float, float]:
    """Project a world point into the segment's (long, lat) local frame.

    ``long`` runs along the segment, ``lat`` is perpendicular to it.
    """
    cx, cy, _, yaw = seg
    dx = rx - cx
    dy = ry - cy
    cs = math.cos(yaw)
    sn = math.sin(yaw)
    long_ = dx * cs + dy * sn
    lat = -dx * sn + dy * cs
    return long_, lat


def _is_parallel(robot_yaw: float, seg_yaw: float, tol: float
                 ) -> Tuple[bool, bool]:
    """Return ``(parallel, reversed_)`` for the (heading, segment) pair.

    ``parallel`` means the robot heading is within ``tol`` of the segment's
    tangent (either direction). ``reversed_`` is True when the heading is
    closer to seg_yaw + pi than to seg_yaw -- needed to flip the lateral
    sign so that "line to the right" stays consistent regardless of which way
    the robot drives along the line.
    """
    a0 = abs(_signed_min_angle(robot_yaw, seg_yaw))
    a1 = abs(_signed_min_angle(robot_yaw, seg_yaw + math.pi))
    if a0 <= tol and a0 <= a1:
        return True, False
    if a1 <= tol and a1 < a0:
        return True, True
    return False, False


def _is_perpendicular(robot_yaw: float, seg_yaw: float, tol: float) -> bool:
    a0 = abs(_signed_min_angle(robot_yaw, seg_yaw + math.pi / 2))
    a1 = abs(_signed_min_angle(robot_yaw, seg_yaw - math.pi / 2))
    return min(a0, a1) <= tol


# ---------------------------------------------------------------- public API


def estimate_line_pose(rx: float, ry: float, yaw: float,
                       segments: List[Segment],
                       cfg: IRGeomConfig
                       ) -> Optional[float]:
    """Return the IR-strip line_pose in [-1, +1], or ``None`` if line is lost.

    Picks the *parallel* segment whose center-line passes closest to the
    robot's lateral axis, projects the robot center onto it, and normalises
    the lateral offset by ``cfg.line_capture_width``.
    """
    best_pose: Optional[float] = None
    best_abs_lat: float = float('inf')
    pad = cfg.cell_size * 0.05  # small overshoot tolerance at endpoints
    for seg in segments:
        parallel, reversed_ = _is_parallel(yaw, seg[3], cfg.parallel_angle_tol)
        if not parallel:
            continue
        long_, lat = _project_into_segment_frame(rx, ry, seg)
        half_len = seg[2] / 2.0
        if abs(long_) > half_len + pad:
            continue
        if abs(lat) > cfg.line_lost_threshold:
            continue
        if abs(lat) < best_abs_lat:
            best_abs_lat = abs(lat)
            signed_lat = -lat if reversed_ else lat
            # ``signed_lat`` is the robot's lateral coordinate in the line
            # frame (left-positive when robot heading == segment yaw). The
            # *line*'s lateral coordinate in the robot frame is therefore
            # ``-signed_lat`` (left-positive), so pose (right-positive) is
            # ``+signed_lat / W``.
            pose = signed_lat / cfg.line_capture_width
            if pose > 1.0:
                pose = 1.0
            elif pose < -1.0:
                pose = -1.0
            best_pose = pose
    return best_pose


def perpendicular_crossings(rx: float, ry: float, yaw: float,
                            segments: List[Segment],
                            cfg: IRGeomConfig
                            ) -> List[int]:
    """Return indices of segments that constitute an active intersection.

    A segment counts when it is perpendicular to the robot heading and its
    centerline passes within ``cfg.intersection_radius`` of the robot center,
    *under the strip* (the segment-frame ``long_`` coordinate of the robot
    must lie inside the segment's length).
    """
    hits: List[int] = []
    pad = cfg.cell_size * 0.10  # tolerance at segment endpoints
    for i, seg in enumerate(segments):
        if not _is_perpendicular(yaw, seg[3], cfg.perp_angle_tol):
            continue
        long_, lat = _project_into_segment_frame(rx, ry, seg)
        if abs(long_) > seg[2] / 2.0 + pad:
            continue
        if abs(lat) <= cfg.intersection_radius:
            hits.append(i)
    return hits


def goal_marker_visible(rx: float, ry: float, yaw: float,
                        goal_xy: Tuple[float, float],
                        cfg: IRGeomConfig
                        ) -> bool:
    """Return True when the goal fiducial would be visible to the camera.

    Stand-in for the ArUco/AprilTag detector: the marker is considered seen
    when the robot is within ``marker_proximity_m`` of the goal cell centre
    AND roughly facing it.
    """
    gx, gy = goal_xy
    dx = gx - rx
    dy = gy - ry
    dist = math.hypot(dx, dy)
    if dist > cfg.marker_proximity_m:
        return False
    if dist < 1e-4:
        return True
    bearing = math.atan2(dy, dx)
    return abs(_signed_min_angle(yaw, bearing)) <= cfg.marker_facing_tol

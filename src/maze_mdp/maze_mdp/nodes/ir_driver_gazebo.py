"""Synthesise IR strip + goal-marker observations from /virtual_odometry.

This is the Gazebo-Classic counterpart of ``ir_driver_sim``. It does **not**
sample ground textures or run ArUco vision; instead it uses the analytic
estimator in :mod:`maze_mdp.control.ir_geom` to compute what the AlphaBot2's
downward IR strip and forward camera would observe given the robot's true
pose in the simulated world. This keeps the simulator hermetic and identical
in behaviour to the unit tests of the geometry module.

Subscribes:
- ``/virtual_odometry`` (``nav_msgs/Odometry``)

Publishes:
- ``/line_pose``        (``std_msgs/Float32``) -- NaN if no line under strip.
- ``/intersection``     (``std_msgs/Empty``)   -- one shot per crossing.
- ``/line_lost``        (``std_msgs/Empty``)   -- after a grace period.
- ``/goal_marker_seen`` (``std_msgs/Bool``)    -- only on transition.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import List, Optional

import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_msgs.msg import Bool, Empty, Float32

from maze_mdp.analysis.maze_to_sdf import (
    MazeSpec,
    _line_segments,
    cell_center,
)
from maze_mdp.control.ir_geom import (
    IRGeomConfig,
    Segment,
    estimate_line_pose,
    goal_marker_visible,
)


def _yaw_from_quat(qx: float, qy: float, qz: float, qw: float) -> float:
    """ZYX yaw from a unit quaternion."""
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)


def _delta_to_cardinal(yaw: float) -> float:
    """Signed smallest angle from ``yaw`` to the nearest cardinal heading.

    Cardinals are 0, +/-pi/2 and +/-pi. The result is in (-pi/4, +pi/4].
    """
    # Normalise yaw to [-pi, pi], then take its modulo pi/2.
    y = math.atan2(math.sin(yaw), math.cos(yaw))
    # Closest multiple of pi/2.
    k = round(y / (math.pi / 2))
    return y - k * (math.pi / 2)


class IRDriverGazebo(Node):
    """Analytic IR strip + goal-marker driver for the Gazebo simulator."""

    def __init__(self) -> None:
        """Initialise publishers, subscriber and timer from parameters."""
        super().__init__('ir_driver_gazebo')

        # --- maze geometry ---
        self.declare_parameter('maze_path', '')
        self.declare_parameter('cell_size', 0.20)
        # --- IR / marker tuning (mirror ir_geom.IRGeomConfig defaults) ---
        self.declare_parameter('line_capture_width', 0.04)
        self.declare_parameter('line_lost_threshold', 0.06)
        self.declare_parameter('parallel_angle_tol', 0.7)
        self.declare_parameter('perp_angle_tol', 0.7)
        self.declare_parameter('intersection_radius', 0.04)
        self.declare_parameter('marker_proximity_m', 0.10)
        self.declare_parameter('marker_facing_tol', 0.6)
        # --- behaviour ---
        self.declare_parameter('publish_rate_hz', 30.0)
        self.declare_parameter('line_lost_grace_s', 0.3)
        # --- topics ---
        self.declare_parameter('odom_topic', '/virtual_odometry')
        self.declare_parameter('line_pose_topic', '/line_pose')
        self.declare_parameter('intersection_topic', '/intersection')
        self.declare_parameter('line_lost_topic', '/line_lost')
        self.declare_parameter('marker_topic', '/goal_marker_seen')

        maze_path = str(self.get_parameter('maze_path').value)
        if not maze_path:
            raise RuntimeError(
                'ir_driver_gazebo: required parameter "maze_path" is empty')
        self._cell_size = float(self.get_parameter('cell_size').value)

        self._cfg = IRGeomConfig(
            cell_size=self._cell_size,
            line_capture_width=float(
                self.get_parameter('line_capture_width').value),
            line_lost_threshold=float(
                self.get_parameter('line_lost_threshold').value),
            parallel_angle_tol=float(
                self.get_parameter('parallel_angle_tol').value),
            perp_angle_tol=float(
                self.get_parameter('perp_angle_tol').value),
            intersection_radius=float(
                self.get_parameter('intersection_radius').value),
            marker_proximity_m=float(
                self.get_parameter('marker_proximity_m').value),
            marker_facing_tol=float(
                self.get_parameter('marker_facing_tol').value),
        )

        maze = MazeSpec.from_yaml(Path(maze_path))
        self._segments: List[Segment] = _line_segments(maze, self._cell_size)
        self._maze = maze
        self._goal_xy = cell_center(maze.goal[0], maze.goal[1],
                                    self._cell_size)
        self.get_logger().info(
            f'Loaded {len(self._segments)} segments from {maze_path}; '
            f'goal cell {maze.goal} at world {self._goal_xy}')

        # --- ROS plumbing ---
        self._pose_pub = self.create_publisher(
            Float32, str(self.get_parameter('line_pose_topic').value), 10)
        self._cross_pub = self.create_publisher(
            Empty, str(self.get_parameter('intersection_topic').value), 10)
        self._lost_pub = self.create_publisher(
            Empty, str(self.get_parameter('line_lost_topic').value), 10)
        self._marker_pub = self.create_publisher(
            Bool, str(self.get_parameter('marker_topic').value), 10)
        self.create_subscription(
            Odometry, str(self.get_parameter('odom_topic').value),
            self._on_odom, 50)

        self._lost_grace = float(
            self.get_parameter('line_lost_grace_s').value)
        rate = float(self.get_parameter('publish_rate_hz').value)
        self._timer = self.create_timer(1.0 / max(rate, 1e-3), self._tick)

        self._pose: Optional[tuple] = None  # (x, y, yaw)
        self._yaw_rate: float = 0.0
        self._lost_since: Optional[float] = None
        self._lost_published = False
        self._last_marker = False
        # Last walkable cell the robot was *inside* (within half-cell radius
        # of its centre); /intersection fires on the rising edge of every
        # change to a new walkable cell.
        self._last_cell: Optional[tuple] = None

    # ------------------------------------------------------------ callbacks

    def _on_odom(self, msg: Odometry) -> None:
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        self._pose = (float(p.x), float(p.y),
                      _yaw_from_quat(q.x, q.y, q.z, q.w))
        self._yaw_rate = float(msg.twist.twist.angular.z)

    def _now_s(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _tick(self) -> None:
        if self._pose is None:
            return
        rx, ry, yaw = self._pose
        now = self._now_s()

        # 1) line_pose / line_lost
        pose_val = estimate_line_pose(rx, ry, yaw, self._segments, self._cfg)
        # When the robot is rotating in place at an intersection (cell
        # centre), the analytic geometry yields pose ~= 0 throughout the
        # turn, so the executor never sees the excursion that signals
        # turn completion. Inject a synthetic excursion based on the
        # angular deviation from the nearest cardinal heading: as the
        # robot rotates 90 deg, sin(2*delta) sweeps 0 -> ~+/-1 -> 0,
        # giving the executor a clean spike-then-zero waveform.
        if abs(self._yaw_rate) > 0.2:
            delta = _delta_to_cardinal(yaw)
            spike = math.sin(2.0 * delta)
            # Sign aligned with the actual turn direction so the executor's
            # TURNING state sees a consistent waveform.
            spike = math.copysign(abs(spike), -self._yaw_rate)
            if pose_val is None or abs(spike) > abs(pose_val):
                pose_val = max(-1.0, min(1.0, spike))
        if pose_val is None:
            self._pose_pub.publish(Float32(data=float('nan')))
            if self._lost_since is None:
                self._lost_since = now
            elif (not self._lost_published
                    and now - self._lost_since >= self._lost_grace):
                self._lost_pub.publish(Empty())
                self._lost_published = True
        else:
            self._pose_pub.publish(Float32(data=float(pose_val)))
            self._lost_since = None
            self._lost_published = False

        # 2) intersection: rising-edge per walkable cell entered. Use
        # cell-centre crossings rather than perpendicular-segment
        # detection so that mazes whose neighbouring cells are walls
        # (no perpendicular tape) still produce one event per cell.
        cs = self._cell_size
        # Snap robot to nearest cell index from world (x, y) =
        # (col*cs, -row*cs).
        col = int(round(rx / cs))
        row = int(round(-ry / cs))
        cur_cell: Optional[tuple] = None
        if (0 <= row < self._maze.rows and 0 <= col < self._maze.cols
                and self._maze.walkable[row][col]):
            cx, cy = cell_center(row, col, cs)
            if (abs(rx - cx) <= self._cfg.intersection_radius
                    and abs(ry - cy) <= self._cfg.intersection_radius):
                cur_cell = (row, col)
        if cur_cell is not None and cur_cell != self._last_cell:
            self._cross_pub.publish(Empty())
            self._last_cell = cur_cell
        elif cur_cell is None:
            # Robot left the cell-centre window; arm the next rising edge.
            self._last_cell = None

        # 3) goal marker. Republish every tick while visible (the executor
        # only acts on True transitions, but it may enter APPROACHING state
        # *after* the marker first became visible, so an edge-only stream
        # can leave it deaf to a marker that's already in view).
        seen = goal_marker_visible(rx, ry, yaw, self._goal_xy, self._cfg)
        if seen and not self._last_marker:
            self.get_logger().info(
                f'goal marker visible at robot=({rx:.3f}, {ry:.3f}) '
                f'yaw={yaw:.2f}')
        if seen:
            self._marker_pub.publish(Bool(data=True))
        elif self._last_marker:
            # Falling edge.
            self._marker_pub.publish(Bool(data=False))
        self._last_marker = seen


def main(args: list[str] | None = None) -> None:
    """Entry point: ``ros2 run maze_mdp ir_driver_gazebo``."""
    rclpy.init(args=args)
    node = IRDriverGazebo()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()

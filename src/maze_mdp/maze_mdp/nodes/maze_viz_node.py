"""
RViz2 visualization bridge for the maze stack.

Subscribes to the project's custom topics and republishes them as standard
ROS 2 visualization types so RViz2 can render them out of the box. Stays
purely on the ROS side: no algorithm code is touched and the maze MDP
modules remain ROS-free.

Subscriptions:
    * ``/maze``              (``maze_msgs/MazeGrid``,  latched)
    * ``/robot_cell``        (``maze_msgs/CellPose``)
    * ``/virtual_odometry``  (``nav_msgs/Odometry``)

Publications:
    * ``/maze_grid``     (``nav_msgs/OccupancyGrid``,        latched)
    * ``/maze_goal``     (``visualization_msgs/Marker``,     latched)
    * ``/robot_marker``  (``visualization_msgs/Marker``)
    * ``/robot_trail``   (``nav_msgs/Path``)
    * ``/policy_arrows`` (``visualization_msgs/MarkerArray``, latched, optional)
    * ``/value_heatmap`` (``visualization_msgs/MarkerArray``, latched, optional)
    * ``tf`` ``map -> base_link`` (broadcast from ``/virtual_odometry``)

World-frame convention (matches ``maze_sim_node``): cell ``(r, c)`` centres at
``x = c * cell_size_m``, ``y = -r * cell_size_m`` in the ``map`` frame.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, TransformStamped
from nav_msgs.msg import OccupancyGrid, Odometry, Path as PathMsg
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import ColorRGBA
from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster
from visualization_msgs.msg import Marker, MarkerArray

from maze_msgs.msg import CellPose, MazeGrid

from maze_mdp.mdp import Action, Heading
from maze_mdp.policy import greedy_policy, load_policy

_HEADING_YAW = {
    int(Heading.N): math.pi / 2.0,
    int(Heading.E): 0.0,
    int(Heading.S): -math.pi / 2.0,
    int(Heading.W): math.pi,
}

_FREE = 0
_WALL = 1
_GOAL = 2


def _latched_qos() -> QoSProfile:
    return QoSProfile(
        depth=1,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
        history=HistoryPolicy.KEEP_LAST,
    )


def _yaw_to_quat(yaw: float) -> tuple[float, float, float, float]:
    """Return (x, y, z, w) of the unit quaternion encoding a yaw-only rotation."""
    return 0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)


class MazeVizNode(Node):
    """Bridge custom maze topics to RViz-friendly visualization messages."""

    def __init__(self) -> None:
        super().__init__('maze_viz_node')
        # Topics (inputs).
        self.declare_parameter('maze_topic', '/maze')
        self.declare_parameter('cell_topic', '/robot_cell')
        self.declare_parameter('odom_topic', '/virtual_odometry')
        # Topics (outputs).
        self.declare_parameter('grid_topic', '/maze_grid')
        self.declare_parameter('walls_topic', '/maze_walls')
        self.declare_parameter('goal_topic', '/maze_goal')
        self.declare_parameter('robot_marker_topic', '/robot_marker')
        self.declare_parameter('trail_topic', '/robot_trail')
        self.declare_parameter('policy_arrows_topic', '/policy_arrows')
        self.declare_parameter('value_heatmap_topic', '/value_heatmap')
        # Frames + geometry.
        self.declare_parameter('frame_id', 'map')
        self.declare_parameter('odom_frame_id', 'odom')
        self.declare_parameter('base_frame_id', 'base_link')
        self.declare_parameter('cell_size_m', 0.20)
        # Optional overlay: load a policy / Q / V bundle for arrows + heatmap.
        self.declare_parameter('policy_path', '')
        # Behaviour toggles.
        self.declare_parameter('publish_tf', True)
        self.declare_parameter('trail_max_poses', 1000)
        # Re-publish the latched MarkerArrays at this rate (Hz). RViz Humble
        # frequently misses the initial TRANSIENT_LOCAL delivery for
        # MarkerArray displays; periodic republishing makes the maze appear
        # reliably regardless of subscriber-startup ordering. Set to <= 0 to
        # disable.
        self.declare_parameter('latched_republish_hz', 1.0)

        gp = self.get_parameter
        self._frame = gp('frame_id').get_parameter_value().string_value
        self._odom_frame = gp('odom_frame_id').get_parameter_value().string_value
        self._base_frame = gp('base_frame_id').get_parameter_value().string_value
        self._cell = float(gp('cell_size_m').value)
        self._publish_tf = bool(gp('publish_tf').value)
        self._trail_max = int(gp('trail_max_poses').value)
        self._policy_path = gp('policy_path').get_parameter_value().string_value

        # Publishers.
        self._grid_pub = self.create_publisher(
            OccupancyGrid, gp('grid_topic').get_parameter_value().string_value, _latched_qos()
        )
        self._walls_pub = self.create_publisher(
            MarkerArray, gp('walls_topic').get_parameter_value().string_value, _latched_qos()
        )
        self._goal_pub = self.create_publisher(
            Marker, gp('goal_topic').get_parameter_value().string_value, _latched_qos()
        )
        self._robot_pub = self.create_publisher(
            Marker, gp('robot_marker_topic').get_parameter_value().string_value, 10
        )
        self._trail_pub = self.create_publisher(
            PathMsg, gp('trail_topic').get_parameter_value().string_value, 10
        )
        self._arrows_pub = self.create_publisher(
            MarkerArray,
            gp('policy_arrows_topic').get_parameter_value().string_value,
            _latched_qos(),
        )
        self._heatmap_pub = self.create_publisher(
            MarkerArray,
            gp('value_heatmap_topic').get_parameter_value().string_value,
            _latched_qos(),
        )
        self._tf = TransformBroadcaster(self) if self._publish_tf else None
        self._static_tf = StaticTransformBroadcaster(self) if self._publish_tf else None
        if self._static_tf is not None:
            self._publish_static_tf()

        # Subscriptions.
        self.create_subscription(
            MazeGrid,
            gp('maze_topic').get_parameter_value().string_value,
            self._on_maze,
            _latched_qos(),
        )
        self.create_subscription(
            CellPose,
            gp('cell_topic').get_parameter_value().string_value,
            self._on_cell,
            10,
        )
        self.create_subscription(
            Odometry,
            gp('odom_topic').get_parameter_value().string_value,
            self._on_odom,
            10,
        )

        # State.
        self._rows = 0
        self._cols = 0
        self._trail = PathMsg()
        self._trail.header.frame_id = self._frame
        self._last_cell: tuple[int, int] | None = None
        self._policy_loaded = False
        # Cached latched payloads (refreshed by ``_on_maze``).
        self._cached_walls: MarkerArray | None = None
        self._cached_goal: Marker | None = None
        self._cached_arrows: MarkerArray | None = None
        self._cached_heatmap: MarkerArray | None = None

        rate = float(gp('latched_republish_hz').value)
        if rate > 0.0:
            self.create_timer(1.0 / rate, self._republish_latched)

    # ----------------------------------------------------------- maze layout
    def _publish_static_tf(self) -> None:
        """Broadcast an identity ``map -> odom`` so the Odometry display resolves.

        ``maze_sim_node`` stamps ``/virtual_odometry`` with ``frame_id=odom``,
        but in pure-sim mode there is no localizer between odom and map; the
        two frames coincide. RViz then needs the static link to render the
        odometry trail in the ``map`` fixed frame.
        """
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = self._frame
        t.child_frame_id = self._odom_frame
        t.transform.rotation.w = 1.0
        self._static_tf.sendTransform(t)

    def _on_maze(self, msg: MazeGrid) -> None:
        rows, cols = int(msg.rows), int(msg.cols)
        cells = np.asarray(msg.cells, dtype=np.int8).reshape(rows, cols)
        self._rows, self._cols = rows, cols
        self._publish_grid(cells)
        self._publish_walls(cells)
        self._publish_goal(cells)
        if not self._policy_loaded and self._policy_path:
            self._load_and_publish_policy(rows, cols)
            self._policy_loaded = True

    def _publish_grid(self, cells: np.ndarray) -> None:
        rows, cols = cells.shape
        grid = OccupancyGrid()
        grid.header.frame_id = self._frame
        grid.header.stamp = self.get_clock().now().to_msg()
        grid.info.resolution = self._cell
        grid.info.width = cols
        grid.info.height = rows
        # OccupancyGrid is row-major from (gx=0, gy=0) at origin going +x then +y.
        # We want grid row gy=0 to be the maze's bottom row (r = rows-1) so that
        # the maze appears with row 0 at the top (+y) under REP-103.
        grid.info.origin.position.x = -self._cell / 2.0
        grid.info.origin.position.y = -(rows - 1) * self._cell - self._cell / 2.0
        grid.info.origin.orientation.w = 1.0
        # Map maze cell value -> occupancy: walls=100, free/goal=0.
        occ = np.zeros((rows, cols), dtype=np.int8)
        occ[cells == _WALL] = 100
        # Vertical flip so maze row 0 ends up at the top (+y).
        flipped = np.flipud(occ)
        grid.data = flipped.flatten().astype(np.int8).tolist()
        self._grid_pub.publish(grid)

    def _republish_latched(self) -> None:
        """Re-emit cached latched MarkerArrays for late RViz subscribers.

        Works around an RViz Humble bug where the MarkerArray display can
        miss the initial TRANSIENT_LOCAL delivery, leaving the scene black.
        """
        if self._cached_walls is not None:
            self._walls_pub.publish(self._cached_walls)
        if self._cached_goal is not None:
            self._goal_pub.publish(self._cached_goal)
        if self._cached_arrows is not None:
            self._arrows_pub.publish(self._cached_arrows)
        if self._cached_heatmap is not None:
            self._heatmap_pub.publish(self._cached_heatmap)

    def _publish_walls(self, cells: np.ndarray) -> None:
        """Walls + floor as a MarkerArray (avoids the OccupancyGrid shader bug).

        The ``rviz/glsl120/indexed_8bit_image`` shader fails to link on some
        Mesa drivers, which leaves the OccupancyGrid display rendering as
        solid black. Using cube markers sidesteps the bug entirely.
        """
        rows, cols = cells.shape
        arr = MarkerArray()
        stamp = self.get_clock().now().to_msg()
        marker_id = 0
        # Floor plane (light grey) so free cells are visible against the
        # dark RViz background even with the heatmap disabled.
        floor = Marker()
        floor.header.frame_id = self._frame
        floor.header.stamp = stamp
        floor.ns = 'walls'
        floor.id = marker_id
        marker_id += 1
        floor.type = Marker.CUBE
        floor.action = Marker.ADD
        floor.pose.position.x = (cols - 1) * self._cell / 2.0
        floor.pose.position.y = -(rows - 1) * self._cell / 2.0
        floor.pose.position.z = -0.02
        floor.pose.orientation.w = 1.0
        floor.scale.x = cols * self._cell
        floor.scale.y = rows * self._cell
        floor.scale.z = 0.01
        floor.color = ColorRGBA(r=0.92, g=0.92, b=0.92, a=1.0)
        arr.markers.append(floor)
        # Wall cubes.
        for r in range(rows):
            for c in range(cols):
                if int(cells[r, c]) != _WALL:
                    continue
                m = Marker()
                m.header.frame_id = self._frame
                m.header.stamp = stamp
                m.ns = 'walls'
                m.id = marker_id
                marker_id += 1
                m.type = Marker.CUBE
                m.action = Marker.ADD
                x, y = self._cell_xy(r, c)
                m.pose.position.x = x
                m.pose.position.y = y
                m.pose.position.z = self._cell * 0.5
                m.pose.orientation.w = 1.0
                m.scale.x = m.scale.y = self._cell
                m.scale.z = self._cell
                m.color = ColorRGBA(r=0.15, g=0.15, b=0.18, a=1.0)
                arr.markers.append(m)
        self._cached_walls = arr
        self._walls_pub.publish(arr)

    def _publish_goal(self, cells: np.ndarray) -> None:
        goals = np.argwhere(cells == _GOAL)
        if goals.size == 0:
            return
        r, c = int(goals[0, 0]), int(goals[0, 1])
        m = Marker()
        m.header.frame_id = self._frame
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = 'goal'
        m.id = 0
        m.type = Marker.SPHERE
        m.action = Marker.ADD
        x, y = self._cell_xy(r, c)
        m.pose.position.x = x
        m.pose.position.y = y
        m.pose.position.z = self._cell * 0.25
        m.pose.orientation.w = 1.0
        m.scale.x = m.scale.y = m.scale.z = self._cell * 0.6
        m.color = ColorRGBA(r=0.0, g=0.85, b=0.1, a=0.85)
        self._cached_goal = m
        self._goal_pub.publish(m)

    # --------------------------------------------------------------- runtime
    def _on_cell(self, msg: CellPose) -> None:
        r, c, h = int(msg.row), int(msg.col), int(msg.heading)
        x, y = self._cell_xy(r, c)
        yaw = _HEADING_YAW.get(h, 0.0)
        qx, qy, qz, qw = _yaw_to_quat(yaw)
        stamp = msg.header.stamp if msg.header.stamp.sec or msg.header.stamp.nanosec \
            else self.get_clock().now().to_msg()

        # Robot marker (arrow at the discrete cell pose).
        m = Marker()
        m.header.frame_id = self._frame
        m.header.stamp = stamp
        m.ns = 'robot'
        m.id = 0
        m.type = Marker.ARROW
        m.action = Marker.ADD
        m.pose.position.x = x
        m.pose.position.y = y
        m.pose.position.z = self._cell * 0.15
        m.pose.orientation.x = qx
        m.pose.orientation.y = qy
        m.pose.orientation.z = qz
        m.pose.orientation.w = qw
        m.scale.x = self._cell * 0.7
        m.scale.y = self._cell * 0.18
        m.scale.z = self._cell * 0.18
        m.color = ColorRGBA(r=0.1, g=0.4, b=1.0, a=1.0)
        self._robot_pub.publish(m)

        # Append to the trail on every fresh cell transition.
        if self._last_cell != (r, c):
            self._last_cell = (r, c)
            ps = PoseStamped()
            ps.header.frame_id = self._frame
            ps.header.stamp = stamp
            ps.pose.position.x = x
            ps.pose.position.y = y
            ps.pose.orientation.w = 1.0
            self._trail.poses.append(ps)
            if len(self._trail.poses) > self._trail_max:
                self._trail.poses = self._trail.poses[-self._trail_max:]
            self._trail.header.stamp = stamp
            self._trail_pub.publish(self._trail)

    def _on_odom(self, msg: Odometry) -> None:
        if not self._publish_tf or self._tf is None:
            return
        # Re-broadcast the continuous pose as ``odom -> base_link``. The
        # complementary static ``map -> odom`` (identity) is published once
        # at startup, so RViz always has a complete map -> base_link chain.
        t = TransformStamped()
        t.header.stamp = msg.header.stamp
        t.header.frame_id = self._odom_frame
        t.child_frame_id = self._base_frame
        t.transform.translation.x = msg.pose.pose.position.x
        t.transform.translation.y = msg.pose.pose.position.y
        t.transform.translation.z = msg.pose.pose.position.z
        t.transform.rotation = msg.pose.pose.orientation
        if t.transform.rotation.x == 0.0 and t.transform.rotation.y == 0.0 \
                and t.transform.rotation.z == 0.0 and t.transform.rotation.w == 0.0:
            t.transform.rotation.w = 1.0
        self._tf.sendTransform(t)

    # ------------------------------------------------------------ overlays
    def _load_and_publish_policy(self, rows: int, cols: int) -> None:
        path = Path(self._policy_path)
        if not path.is_file():
            self.get_logger().warn(f'policy_path {path} not found; skipping overlays.')
            return
        try:
            bundle = load_policy(path)
        except Exception as exc:  # noqa: BLE001 - log and continue
            self.get_logger().warn(f'Failed to load {path}: {exc}')
            return

        if 'pi' in bundle:
            pi = np.asarray(bundle['pi'], dtype=np.int64)
        elif 'Q' in bundle:
            pi = greedy_policy(np.asarray(bundle['Q']))
        else:
            pi = None

        n_states = rows * cols * 4
        if pi is not None and pi.size == n_states:
            arrows = self._build_policy_arrows(pi, rows, cols)
            self._cached_arrows = arrows
            self._arrows_pub.publish(arrows)
        elif pi is not None:
            self.get_logger().warn(
                f'Policy size {pi.size} != expected {n_states}; arrows skipped.'
            )

        V = bundle.get('V')
        if V is None and 'Q' in bundle:
            V = np.asarray(bundle['Q']).max(axis=1)
        if V is not None and V.size == n_states:
            heatmap = self._build_value_heatmap(V, rows, cols)
            self._cached_heatmap = heatmap
            self._heatmap_pub.publish(heatmap)

    def _build_policy_arrows(self, pi: np.ndarray, rows: int, cols: int) -> MarkerArray:
        """One small arrow per (cell, heading); colour encodes the action."""
        arr = MarkerArray()
        marker_id = 0
        # Slight in-cell offsets so the four headings don't overlap.
        offsets = {
            int(Heading.N): (0.0, +0.18),
            int(Heading.E): (+0.18, 0.0),
            int(Heading.S): (0.0, -0.18),
            int(Heading.W): (-0.18, 0.0),
        }
        action_color = {
            int(Action.FORWARD): ColorRGBA(r=0.1, g=0.85, b=0.1, a=0.9),
            int(Action.TURN_LEFT): ColorRGBA(r=0.2, g=0.4, b=1.0, a=0.9),
            int(Action.TURN_RIGHT): ColorRGBA(r=1.0, g=0.4, b=0.2, a=0.9),
        }
        stamp = self.get_clock().now().to_msg()
        for r in range(rows):
            for c in range(cols):
                base_x, base_y = self._cell_xy(r, c)
                for h in range(4):
                    s = (r * cols + c) * 4 + h
                    a = int(pi[s])
                    yaw = _HEADING_YAW[h]
                    qx, qy, qz, qw = _yaw_to_quat(yaw)
                    ox, oy = offsets[h]
                    m = Marker()
                    m.header.frame_id = self._frame
                    m.header.stamp = stamp
                    m.ns = 'policy'
                    m.id = marker_id
                    marker_id += 1
                    m.type = Marker.ARROW
                    m.action = Marker.ADD
                    m.pose.position.x = base_x + ox * self._cell
                    m.pose.position.y = base_y + oy * self._cell
                    m.pose.position.z = self._cell * 0.05
                    m.pose.orientation.x = qx
                    m.pose.orientation.y = qy
                    m.pose.orientation.z = qz
                    m.pose.orientation.w = qw
                    m.scale.x = self._cell * 0.30
                    m.scale.y = self._cell * 0.06
                    m.scale.z = self._cell * 0.06
                    m.color = action_color.get(
                        a, ColorRGBA(r=0.5, g=0.5, b=0.5, a=0.9)
                    )
                    arr.markers.append(m)
        return arr

    def _build_value_heatmap(self, V: np.ndarray, rows: int, cols: int) -> MarkerArray:
        """One translucent CUBE per cell; colour scaled by max-over-heading V."""
        v_per_cell = V.reshape(rows, cols, 4).max(axis=2)
        finite = v_per_cell[np.isfinite(v_per_cell)]
        if finite.size == 0:
            return MarkerArray()
        vmin, vmax = float(finite.min()), float(finite.max())
        span = max(vmax - vmin, 1e-9)
        arr = MarkerArray()
        stamp = self.get_clock().now().to_msg()
        marker_id = 0
        for r in range(rows):
            for c in range(cols):
                v = float(v_per_cell[r, c])
                if not np.isfinite(v):
                    continue
                t = (v - vmin) / span  # 0..1
                m = Marker()
                m.header.frame_id = self._frame
                m.header.stamp = stamp
                m.ns = 'value'
                m.id = marker_id
                marker_id += 1
                m.type = Marker.CUBE
                m.action = Marker.ADD
                x, y = self._cell_xy(r, c)
                m.pose.position.x = x
                m.pose.position.y = y
                m.pose.position.z = -0.01  # just below the floor plane
                m.pose.orientation.w = 1.0
                m.scale.x = m.scale.y = self._cell * 0.95
                m.scale.z = 0.01
                # Viridis-ish: low=purple, high=yellow.
                m.color = ColorRGBA(
                    r=float(0.27 + 0.6 * t),
                    g=float(0.05 + 0.85 * t),
                    b=float(0.55 - 0.45 * t),
                    a=0.55,
                )
                arr.markers.append(m)
        return arr

    # ----------------------------------------------------------------- utils
    def _cell_xy(self, row: int, col: int) -> tuple[float, float]:
        return float(col) * self._cell, -float(row) * self._cell


def main(args: list[str] | None = None) -> None:
    """Entry point used by ``ros2 run maze_mdp maze_viz_node``."""
    rclpy.init(args=args)
    node = MazeVizNode()
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

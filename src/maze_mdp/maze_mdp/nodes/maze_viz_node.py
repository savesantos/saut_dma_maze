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


def build_car_marker_array(
    *,
    frame_id: str,
    stamp,
    namespace: str,
    base_id: int,
    x: float,
    y: float,
    yaw: float,
    cell_size: float,
    body_color: ColorRGBA,
    cabin_color: ColorRGBA,
    front_color: ColorRGBA,
) -> MarkerArray:
    """Compose a tiny "car" out of three CUBE markers, oriented by ``yaw``.

    The three markers share ``namespace`` but use successive IDs starting at
    ``base_id`` so multiple cars can coexist by allocating disjoint ID ranges
    (or by using different namespaces).
    """
    qx, qy, qz, qw = _yaw_to_quat(yaw)
    arr = MarkerArray()
    s = float(cell_size)
    # Local-frame offsets (x = forward, y = left).
    parts = [
        # (dx, dy, dz, sx, sy, sz, color)
        (0.0, 0.0, s * 0.10, s * 0.55, s * 0.32, s * 0.16, body_color),  # body
        (-s * 0.08, 0.0, s * 0.22, s * 0.30, s * 0.26, s * 0.12, cabin_color),  # cabin
        (s * 0.30, 0.0, s * 0.10, s * 0.10, s * 0.32, s * 0.05, front_color),  # bumper/light
    ]
    cy = math.cos(yaw)
    sy = math.sin(yaw)
    for i, (dx, dy, dz, sx, sxy, sz, color) in enumerate(parts):
        m = Marker()
        m.header.frame_id = frame_id
        m.header.stamp = stamp
        m.ns = namespace
        m.id = base_id + i
        m.type = Marker.CUBE
        m.action = Marker.ADD
        # Rotate (dx, dy) by yaw before adding to the cell centre so the
        # cabin / front sit on the correct side regardless of heading.
        m.pose.position.x = float(x + cy * dx - sy * dy)
        m.pose.position.y = float(y + sy * dx + cy * dy)
        m.pose.position.z = float(dz)
        m.pose.orientation.x = qx
        m.pose.orientation.y = qy
        m.pose.orientation.z = qz
        m.pose.orientation.w = qw
        m.scale.x = sx
        m.scale.y = sxy
        m.scale.z = sz
        m.color = color
        arr.markers.append(m)
    return arr


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
        # Source for the robot marker + trail:
        #   'cell' -> snap to /robot_cell (discrete, good for tabular sim)
        #   'odom' -> follow /virtual_odometry (continuous, good for the
        #            realistic sim / hardware where the executor steers the
        #            robot inside a cell)
        self.declare_parameter('robot_source', 'cell')
        # Floor rendering style:
        #   'cells' -> walls as cubes + light-grey floor (default; matches
        #              the discrete-MDP view of the maze).
        #   'lines' -> white floor + black line segments between adjacent
        #              passable cells (matches the physical maze paper that
        #              the IR strip actually follows, and what
        #              ``physics_sim_node`` senses internally).
        self.declare_parameter('floor_style', 'cells')
        self.declare_parameter('line_half_width_m', 0.012)
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
        self._robot_source = gp('robot_source').get_parameter_value().string_value
        if self._robot_source not in ('cell', 'odom'):
            self._robot_source = 'cell'
        self._floor_style = gp('floor_style').get_parameter_value().string_value
        if self._floor_style not in ('cells', 'lines'):
            self._floor_style = 'cells'
        self._line_half_width = float(gp('line_half_width_m').value)

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
            MarkerArray,
            gp('robot_marker_topic').get_parameter_value().string_value,
            10,
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
        # Last decoded maze layout (rows, cols int8 with values _FREE/_WALL/_GOAL).
        # Needed by the overlay builders to mask walls and the goal cell.
        self._cells: np.ndarray | None = None
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
        self._cells = cells
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
        """Dispatch wall rendering on ``floor_style`` ('cells' or 'lines')."""
        if self._floor_style == 'lines':
            self._publish_walls_lines(cells)
            return
        self._publish_walls_cubes(cells)

    def _publish_walls_lines(self, cells: np.ndarray) -> None:
        """White floor + black line segments between adjacent passable cells.

        Matches the physical maze (black tape on white paper) and the
        geometry that ``physics_sim_node`` senses internally.
        """
        rows, cols = cells.shape
        arr = MarkerArray()
        stamp = self.get_clock().now().to_msg()
        marker_id = 0
        # White floor plane covering the full maze.
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
        floor.pose.position.z = -0.01
        floor.pose.orientation.w = 1.0
        floor.scale.x = cols * self._cell
        floor.scale.y = rows * self._cell
        floor.scale.z = 0.005
        floor.color = ColorRGBA(r=0.98, g=0.98, b=0.98, a=1.0)
        arr.markers.append(floor)
        # Black line segments between every pair of adjacent passable cells,
        # rendered as thin flat cubes at z=0 in the viz frame.
        black = ColorRGBA(r=0.05, g=0.05, b=0.05, a=1.0)
        thickness = max(2.0 * self._line_half_width, 0.005)
        height = 0.002

        def passable(r: int, c: int) -> bool:
            return 0 <= r < rows and 0 <= c < cols \
                and int(cells[r, c]) != _WALL

        for r in range(rows):
            for c in range(cols):
                if not passable(r, c):
                    continue
                cx, cy = self._cell_xy(r, c)
                if passable(r, c + 1):
                    m = Marker()
                    m.header.frame_id = self._frame
                    m.header.stamp = stamp
                    m.ns = 'walls'
                    m.id = marker_id
                    marker_id += 1
                    m.type = Marker.CUBE
                    m.action = Marker.ADD
                    m.pose.position.x = cx + self._cell / 2.0
                    m.pose.position.y = cy
                    m.pose.position.z = 0.0
                    m.pose.orientation.w = 1.0
                    m.scale.x = self._cell
                    m.scale.y = thickness
                    m.scale.z = height
                    m.color = black
                    arr.markers.append(m)
                if passable(r + 1, c):
                    m = Marker()
                    m.header.frame_id = self._frame
                    m.header.stamp = stamp
                    m.ns = 'walls'
                    m.id = marker_id
                    marker_id += 1
                    m.type = Marker.CUBE
                    m.action = Marker.ADD
                    m.pose.position.x = cx
                    m.pose.position.y = cy - self._cell / 2.0
                    m.pose.position.z = 0.0
                    m.pose.orientation.w = 1.0
                    m.scale.x = thickness
                    m.scale.y = self._cell
                    m.scale.z = height
                    m.color = black
                    arr.markers.append(m)
        self._cached_walls = arr
        self._walls_pub.publish(arr)

    def _publish_walls_cubes(self, cells: np.ndarray) -> None:
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
        if self._robot_source != 'cell':
            return
        r, c, h = int(msg.row), int(msg.col), int(msg.heading)
        x, y = self._cell_xy(r, c)
        yaw = _HEADING_YAW.get(h, 0.0)
        stamp = msg.header.stamp if msg.header.stamp.sec or msg.header.stamp.nanosec \
            else self.get_clock().now().to_msg()

        # Robot "car": body cube + cabin cube + small front wedge so the
        # heading is unambiguous even when the camera is top-down.
        self._publish_car(x, y, yaw, stamp)

        # Append to the trail on every fresh cell transition.
        if self._last_cell != (r, c):
            self._last_cell = (r, c)
            self._append_trail(x, y, stamp)

    def _publish_car(self, x: float, y: float, yaw: float, stamp) -> None:
        body_color = ColorRGBA(r=0.10, g=0.40, b=1.00, a=1.0)
        cabin_color = ColorRGBA(r=0.05, g=0.20, b=0.55, a=1.0)
        front_color = ColorRGBA(r=1.00, g=0.95, b=0.30, a=1.0)
        car = build_car_marker_array(
            frame_id=self._frame,
            stamp=stamp,
            namespace='robot',
            base_id=0,
            x=x,
            y=y,
            yaw=yaw,
            cell_size=self._cell,
            body_color=body_color,
            cabin_color=cabin_color,
            front_color=front_color,
        )
        self._robot_pub.publish(car)

    def _append_trail(self, x: float, y: float, stamp) -> None:
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
        # Drive the robot marker + trail directly from continuous odometry
        # when configured (realistic sim / hardware path).
        if self._robot_source == 'odom':
            x = msg.pose.pose.position.x
            y = msg.pose.pose.position.y
            q = msg.pose.pose.orientation
            # Yaw from quaternion (z-axis rotation).
            yaw = math.atan2(
                2.0 * (q.w * q.z + q.x * q.y),
                1.0 - 2.0 * (q.y * q.y + q.z * q.z),
            )
            stamp = msg.header.stamp if msg.header.stamp.sec \
                or msg.header.stamp.nanosec \
                else self.get_clock().now().to_msg()
            self._publish_car(x, y, yaw, stamp)
            # Sub-sample the trail to ~5 cm steps so it stays light.
            last = self._trail.poses[-1].pose.position if self._trail.poses else None
            if last is None or math.hypot(x - last.x, y - last.y) > 0.02:
                self._append_trail(x, y, stamp)

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
        elif V is not None:
            self.get_logger().warn(
                f'V size {V.size} != expected {n_states}; heatmap skipped.'
            )

    def _build_policy_arrows(self, pi: np.ndarray, rows: int, cols: int) -> MarkerArray:
        """
        Render one arrow per ``(cell, heading)`` for every reachable cell.

        Walls and the goal cell are skipped (they carry no decision).
        Each arrow is anchored on the side of the cell that matches the
        state heading (so the four headings don't overlap), and points
        toward the heading the agent will be facing **after applying the
        chosen action** — i.e. it visually shows where the policy sends
        the agent. Colour also encodes the action as a redundant cue:

        * green  = FORWARD       (arrow keeps the state heading)
        * blue   = TURN_LEFT     (arrow rotated 90 deg counter-clockwise)
        * orange = TURN_RIGHT    (arrow rotated 90 deg clockwise)
        """
        arr = MarkerArray()
        marker_id = 0
        # Slight in-cell offsets so the four headings don't overlap. The
        # offset direction matches the state heading so the arrow tail sits
        # on the side of the cell that the agent is facing.
        offsets = {
            int(Heading.N): (0.0, +0.22),
            int(Heading.E): (+0.22, 0.0),
            int(Heading.S): (0.0, -0.22),
            int(Heading.W): (-0.22, 0.0),
        }
        action_color = {
            int(Action.FORWARD): ColorRGBA(r=0.1, g=0.85, b=0.1, a=0.95),
            int(Action.TURN_LEFT): ColorRGBA(r=0.2, g=0.4, b=1.0, a=0.95),
            int(Action.TURN_RIGHT): ColorRGBA(r=1.0, g=0.4, b=0.2, a=0.95),
        }
        # Resulting heading after each action, given the state heading ``h``.
        # FORWARD keeps the heading (the agent moves into the next cell but
        # still faces the same way). TURNs rotate in place.
        action_to_resulting_h = {
            int(Action.FORWARD): lambda h: h,
            int(Action.TURN_LEFT): lambda h: (h - 1) % 4,
            int(Action.TURN_RIGHT): lambda h: (h + 1) % 4,
        }
        cells = self._cells
        stamp = self.get_clock().now().to_msg()
        goal_cell: tuple[int, int] | None = None
        if cells is not None:
            goals = np.argwhere(cells == _GOAL)
            if goals.size:
                goal_cell = (int(goals[0, 0]), int(goals[0, 1]))
        for r in range(rows):
            for c in range(cols):
                # Skip walls and the absorbing goal cell — neither carries
                # a meaningful policy decision for the user to inspect.
                if cells is not None and int(cells[r, c]) == _WALL:
                    continue
                if goal_cell is not None and (r, c) == goal_cell:
                    continue
                base_x, base_y = self._cell_xy(r, c)
                for h in range(4):
                    s = (r * cols + c) * 4 + h
                    a = int(pi[s])
                    res_h = action_to_resulting_h.get(a, lambda hh: hh)(h)
                    yaw = _HEADING_YAW[res_h]
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
                    m.scale.x = self._cell * 0.28
                    m.scale.y = self._cell * 0.06
                    m.scale.z = self._cell * 0.06
                    m.color = action_color.get(
                        a, ColorRGBA(r=0.5, g=0.5, b=0.5, a=0.9)
                    )
                    arr.markers.append(m)
        return arr

    def _build_value_heatmap(self, V: np.ndarray, rows: int, cols: int) -> MarkerArray:
        """
        Render one translucent CUBE per non-wall cell using ``max_h V(s)``.

        Walls are skipped entirely (their state values are either an
        absorbing 0 from VI or the pessimistic ``q_init`` from SARSA /
        Q-Learning, neither of which carries information). The goal cell
        is forced to the top of the colour scale so it stands out as the
        attractor regardless of the algorithm's terminal-state value
        convention.
        """
        v_per_cell = V.reshape(rows, cols, 4).max(axis=2)
        cells = self._cells
        # Build a mask of cells whose value should drive the colour scale
        # (free + goal, but not walls). When ``cells`` is unavailable for
        # any reason we fall back to all finite cells to preserve the old
        # behaviour.
        if cells is not None:
            non_wall = cells != _WALL
        else:
            non_wall = np.ones_like(v_per_cell, dtype=bool)
        valid = non_wall & np.isfinite(v_per_cell)
        if not np.any(valid):
            return MarkerArray()
        # Locate the goal cell so we can pin it to vmax even when the
        # underlying terminal-state Q row is left at its pessimistic init.
        goal_cell: tuple[int, int] | None = None
        if cells is not None:
            goals = np.argwhere(cells == _GOAL)
            if goals.size:
                goal_cell = (int(goals[0, 0]), int(goals[0, 1]))
        # Compute the normalisation range from non-wall cells *excluding*
        # the goal: the goal's stored V is not comparable across algorithms
        # (VI: 0; TD: q_init) and would otherwise dominate vmin/vmax.
        scale_mask = valid.copy()
        if goal_cell is not None:
            scale_mask[goal_cell] = False
        if np.any(scale_mask):
            scale_vals = v_per_cell[scale_mask]
            vmin = float(scale_vals.min())
            vmax = float(scale_vals.max())
        else:
            vmin = float(v_per_cell[valid].min())
            vmax = float(v_per_cell[valid].max())
        span = max(vmax - vmin, 1e-9)
        arr = MarkerArray()
        stamp = self.get_clock().now().to_msg()
        marker_id = 0
        for r in range(rows):
            for c in range(cols):
                if not valid[r, c]:
                    continue
                if goal_cell is not None and (r, c) == goal_cell:
                    t = 1.0  # always render the goal at the top of the scale
                else:
                    v = float(v_per_cell[r, c])
                    t = (v - vmin) / span
                    t = max(0.0, min(1.0, t))
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

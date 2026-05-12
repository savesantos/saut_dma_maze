"""
ROS 2 micro-simulator node — Option A bridge to the pure-Python sim.

Drop-in replacement for the AlphaBot2 hardware + ``fiducial_localizer`` pair:
the same ``maze_publisher`` -> ``policy_runner`` topic graph runs unchanged,
but cell-pose updates and odometry come from the stochastic grid-MDP defined
in :mod:`maze_mdp.simulator` instead of the camera + ArUco pipeline.

Topic interface (matches the physical robot exactly):

* Subscribes
    - ``/maze``              (latched ``maze_msgs/MazeGrid``)
    - ``/alphabot2/cmd_vel`` (``geometry_msgs/Twist``)
* Publishes
    - ``/robot_cell``        (``maze_msgs/CellPose``) on every cell transition
    - ``/virtual_odometry``  (``nav_msgs/Odometry``)  at the tick rate

Continuous-time mapping ``Twist -> Action``:

The latest ``cmd_vel`` is integrated at ``tick_rate_hz``. When the linear
displacement integral exceeds ``cell_size_m`` an MDP ``FORWARD`` step is
sampled (preserving the configured slip noise); when the angular integral
exceeds ``step_angle_rad`` a ``TURN_LEFT`` / ``TURN_RIGHT`` step fires. The
algorithm module ``maze_mdp.simulator`` stays ROS-free; this node only
calls into it.
"""

from __future__ import annotations

import math

import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Empty

from maze_msgs.msg import CellPose, MazeGrid

from maze_mdp.mdp import Action, Heading, MDPConfig
from maze_mdp.simulator import GOAL, GridMaze, Maze

# Heading -> yaw (radians) under REP-103 (x=East, y=North).
_HEADING_YAW = {
    int(Heading.N): math.pi / 2.0,
    int(Heading.E): 0.0,
    int(Heading.S): -math.pi / 2.0,
    int(Heading.W): math.pi,
}


def _latched_sub_qos() -> QoSProfile:
    return QoSProfile(
        depth=1,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
        history=HistoryPolicy.KEEP_LAST,
    )


def _yaw_to_quat_z(yaw: float) -> tuple[float, float]:
    """Return (z, w) of the unit quaternion encoding a yaw-only rotation."""
    return math.sin(yaw / 2.0), math.cos(yaw / 2.0)


class MazeSimNode(Node):
    """ROS-side wrapper around :class:`GridMaze` for sim-only deployment runs."""

    def __init__(self) -> None:
        super().__init__('maze_sim_node')
        # Topics
        self.declare_parameter('maze_topic', '/maze')
        self.declare_parameter('cmd_topic', '/alphabot2/cmd_vel')
        self.declare_parameter('cell_topic', '/robot_cell')
        self.declare_parameter('odom_topic', '/virtual_odometry')
        self.declare_parameter('reset_topic', '/sim_reset')
        self.declare_parameter('reset_to_topic', '/sim_reset_to')
        # Frames (used in Odometry header).
        self.declare_parameter('odom_frame_id', 'odom')
        self.declare_parameter('base_frame_id', 'base_link')
        # Continuous integration.
        self.declare_parameter('tick_rate_hz', 20.0)
        self.declare_parameter('cell_size_m', 0.20)
        self.declare_parameter('step_angle_rad', math.pi / 2.0)
        # MDP dynamics.
        self.declare_parameter('slip_prob', 0.1)
        self.declare_parameter('seed', -1)
        # Initial pose (overrides random reset if start_row >= 0).
        self.declare_parameter('start_row', -1)
        self.declare_parameter('start_col', -1)
        self.declare_parameter('start_heading', int(Heading.E))

        gp = self.get_parameter
        self._cell_topic = gp('cell_topic').get_parameter_value().string_value
        self._odom_topic = gp('odom_topic').get_parameter_value().string_value
        self._odom_frame = gp('odom_frame_id').get_parameter_value().string_value
        self._base_frame = gp('base_frame_id').get_parameter_value().string_value
        self._dt = 1.0 / max(float(gp('tick_rate_hz').value), 1e-3)
        self._cell_size = float(gp('cell_size_m').value)
        self._step_angle = float(gp('step_angle_rad').value)
        self._slip = float(gp('slip_prob').value)
        seed = int(gp('seed').value)
        self._rng = np.random.default_rng(seed if seed >= 0 else None)

        sr = int(gp('start_row').value)
        sc = int(gp('start_col').value)
        sh = int(gp('start_heading').value)
        self._start: tuple[int, int, int] | None = (sr, sc, sh) if sr >= 0 and sc >= 0 else None

        # I/O
        self._cell_pub = self.create_publisher(CellPose, self._cell_topic, 10)
        self._odom_pub = self.create_publisher(Odometry, self._odom_topic, 10)
        self._maze_sub = self.create_subscription(
            MazeGrid,
            gp('maze_topic').get_parameter_value().string_value,
            self._on_maze,
            _latched_sub_qos(),
        )
        self._cmd_sub = self.create_subscription(
            Twist,
            gp('cmd_topic').get_parameter_value().string_value,
            self._on_cmd,
            10,
        )
        self._reset_sub = self.create_subscription(
            Empty,
            gp('reset_topic').get_parameter_value().string_value,
            self._on_reset,
            10,
        )
        self._reset_to_sub = self.create_subscription(
            CellPose,
            gp('reset_to_topic').get_parameter_value().string_value,
            self._on_reset_to,
            10,
        )
        self._timer = self.create_timer(self._dt, self._tick)

        # Lazy state — populated on first MazeGrid.
        self._env: GridMaze | None = None
        self._twist = Twist()
        self._lin_progress = 0.0
        self._ang_progress = 0.0
        self._done = False

    # --------------------------------------------------------------- callbacks
    def _on_maze(self, msg: MazeGrid) -> None:
        if self._env is not None:
            return  # latched re-delivery; ignore.
        cells = np.asarray(msg.cells, dtype=np.int8).reshape(int(msg.rows), int(msg.cols))
        goals = np.argwhere(cells == GOAL)
        if goals.size == 0:
            self.get_logger().warn('MazeGrid has no goal cell; waiting.')
            return
        goal = (int(goals[0, 0]), int(goals[0, 1]))
        # Strip GOAL marker from cell array — Maze stores it via the goal field.
        layout_cells = cells.copy()
        layout_cells[layout_cells == GOAL] = 0
        layout_cells[goal] = GOAL
        maze = Maze(cells=layout_cells, goal=goal)
        self._env = GridMaze(
            maze=maze,
            config=MDPConfig(slip_prob=self._slip),
            rng=self._rng,
            start=self._start,
        )
        self._env.reset()
        self.get_logger().info(
            f'Sim ready: {maze.rows}x{maze.cols}, goal={goal}, slip={self._slip}'
        )
        self._publish_cell()
        self._publish_odom()

    def _on_cmd(self, msg: Twist) -> None:
        self._twist = msg

    def _on_reset(self, _msg: Empty) -> None:
        """
        Re-sample the start state and resume simulating from scratch.

        Honours any updated ``start_row`` / ``start_col`` / ``start_heading``
        parameters (settable via ``ros2 param set /maze_sim_node ...``) so a
        user can pick a new initial cell without restarting the launch.
        """
        if self._env is None:
            self.get_logger().warn('Reset requested before maze loaded; ignoring.')
            return
        self._refresh_start_from_params()
        self._env._start = self._start  # type: ignore[attr-defined]
        self._env.reset()
        self._after_reset('Sim reset; episode restarted.')

    def _on_reset_to(self, msg: CellPose) -> None:
        """Reset to the explicit (row, col, heading) carried in ``msg``."""
        if self._env is None:
            self.get_logger().warn('Reset-to requested before maze loaded; ignoring.')
            return
        target = (int(msg.row), int(msg.col), int(msg.heading))
        if not self._is_valid_start(target):
            self.get_logger().warn(
                f'Reset-to target {target} is outside the maze, on a wall, '
                'or on the goal; ignoring.'
            )
            return
        self._env._start = target  # type: ignore[attr-defined]
        self._env.reset()
        self._after_reset(f'Sim reset to cell {target}.')

    # ----------------------------------------------------------------- start
    def _refresh_start_from_params(self) -> None:
        gp = self.get_parameter
        sr = int(gp('start_row').value)
        sc = int(gp('start_col').value)
        sh = int(gp('start_heading').value)
        self._start = (sr, sc, sh) if sr >= 0 and sc >= 0 else None

    def _is_valid_start(self, cell: tuple[int, int, int]) -> bool:
        assert self._env is not None
        r, c, h = cell
        maze = self._env.maze
        if not (0 <= r < maze.rows and 0 <= c < maze.cols):
            return False
        if maze.cells[r, c] == 1:  # WALL
            return False
        if (r, c) == maze.goal:
            return False
        if not 0 <= h < 4:
            return False
        return True

    def _after_reset(self, log_msg: str) -> None:
        self._twist = Twist()
        self._lin_progress = 0.0
        self._ang_progress = 0.0
        self._done = False
        self.get_logger().info(log_msg)
        self._publish_cell()
        self._publish_odom()

    # ------------------------------------------------------------------- tick
    def _tick(self) -> None:
        if self._env is None or self._done:
            return
        # Integrate the latest command. A live policy_runner re-publishes the
        # same Twist every control tick, so this approximates continuous-time
        # motion at the AlphaBot2's nominal speeds.
        self._lin_progress += float(self._twist.linear.x) * self._dt
        self._ang_progress += float(self._twist.angular.z) * self._dt

        action = self._dominant_action()
        if action is not None:
            self._step(action)

        # Republish the current cell every tick: cheap, and guarantees that
        # subscribers that connect after the first /maze callback still see
        # an initial pose. Otherwise the policy_runner stays in WAITING_CELL.
        self._publish_cell()
        self._publish_odom()

    # ----------------------------------------------------------------- helpers
    def _dominant_action(self) -> int | None:
        """Pick the discrete action whose threshold has been crossed, if any."""
        # Translation wins ties — matches policy_runner: FORWARD has linear.x
        # set and angular.z = 0, so the angular accumulator stays at 0.
        if self._lin_progress >= self._cell_size:
            return int(Action.FORWARD)
        if self._ang_progress >= self._step_angle:
            return int(Action.TURN_LEFT)
        if self._ang_progress <= -self._step_angle:
            return int(Action.TURN_RIGHT)
        return None

    def _step(self, action: int) -> None:
        assert self._env is not None
        _, _, done, info = self._env.step(action)
        self._lin_progress = 0.0
        self._ang_progress = 0.0
        self._publish_cell()
        if done:
            self._done = True
            kind = 'goal' if info.get('terminated') else 'truncated'
            self.get_logger().info(f'Sim episode ended ({kind}) after {info["steps"]} steps.')

    def _decode(self) -> tuple[int, int, int]:
        assert self._env is not None
        return self._env.mdp.decode(self._env.state)

    def _publish_cell(self) -> None:
        r, c, h = self._decode()
        msg = CellPose()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self._odom_frame
        msg.row = int(r)
        msg.col = int(c)
        msg.heading = int(h)
        msg.confidence = 1.0
        self._cell_pub.publish(msg)

    def _publish_odom(self) -> None:
        if self._env is None:
            return
        r, c, h = self._decode()
        # Cell-center pose: world x = col * cell_size, y = -row * cell_size
        # so increasing rows go "south" (negative y) under REP-103.
        x = c * self._cell_size
        y = -r * self._cell_size
        yaw = _HEADING_YAW[int(h)]
        qz, qw = _yaw_to_quat_z(yaw)

        msg = Odometry()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self._odom_frame
        msg.child_frame_id = self._base_frame
        msg.pose.pose.position.x = float(x)
        msg.pose.pose.position.y = float(y)
        msg.pose.pose.orientation.z = float(qz)
        msg.pose.pose.orientation.w = float(qw)
        msg.twist.twist.linear.x = float(self._twist.linear.x)
        msg.twist.twist.angular.z = float(self._twist.angular.z)
        self._odom_pub.publish(msg)


def main(args: list[str] | None = None) -> None:
    """Entry point used by ``ros2 run maze_mdp maze_sim_node``."""
    rclpy.init(args=args)
    node = MazeSimNode()
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

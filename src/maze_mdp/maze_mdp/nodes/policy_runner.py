"""
Closed-loop policy executor for the AlphaBot2 maze stack.

State machine:

    WAITING_MAZE  -- got /maze --> WAITING_CELL
    WAITING_CELL  -- got /robot_cell --> RUNNING
    RUNNING       -- terminal cell reached --> DONE

The node loads a precomputed policy or Q-table from disk, looks up the action
for the current cell estimate at every control tick, and converts the discrete
action into a ``geometry_msgs/Twist`` published on ``/alphabot2/cmd_vel``.
"""

from __future__ import annotations

from enum import Enum

import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy

from maze_msgs.msg import CellPose, MazeGrid

from maze_mdp.mdp import Action
from maze_mdp.policy import load_policy, greedy_policy


class _State(Enum):
    WAITING_MAZE = 'waiting_maze'
    WAITING_CELL = 'waiting_cell'
    RUNNING = 'running'
    DONE = 'done'


def _latched_sub_qos() -> QoSProfile:
    return QoSProfile(
        depth=1,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
        history=HistoryPolicy.KEEP_LAST,
    )


class PolicyRunner(Node):
    """Drive the AlphaBot2 by following a precomputed discrete policy."""

    def __init__(self) -> None:
        super().__init__('policy_runner')
        self.declare_parameter('policy_path', '')
        self.declare_parameter('cmd_topic', '/alphabot2/cmd_vel')
        self.declare_parameter('cell_topic', '/robot_cell')
        self.declare_parameter('maze_topic', '/maze')
        self.declare_parameter('control_rate_hz', 5.0)
        self.declare_parameter('forward_speed', 0.1)
        self.declare_parameter('turn_speed', 0.6)

        policy_path = self.get_parameter('policy_path').get_parameter_value().string_value
        if not policy_path:
            raise RuntimeError('policy_runner requires a non-empty policy_path parameter.')
        bundle = load_policy(policy_path)
        if 'pi' in bundle:
            self._pi = np.asarray(bundle['pi'], dtype=np.int64)
        elif 'Q' in bundle:
            self._pi = greedy_policy(np.asarray(bundle['Q']))
        else:
            raise RuntimeError(f'No pi or Q array found in {policy_path}')
        self.get_logger().info(f'Loaded policy with {self._pi.size} states from {policy_path}')

        cmd_topic = self.get_parameter('cmd_topic').get_parameter_value().string_value
        cell_topic = self.get_parameter('cell_topic').get_parameter_value().string_value
        maze_topic = self.get_parameter('maze_topic').get_parameter_value().string_value
        rate_hz = float(self.get_parameter('control_rate_hz').value)
        self._fwd = float(self.get_parameter('forward_speed').value)
        self._turn = float(self.get_parameter('turn_speed').value)

        self._cmd_pub = self.create_publisher(Twist, cmd_topic, 10)
        self._maze_sub = self.create_subscription(
            MazeGrid, maze_topic, self._on_maze, _latched_sub_qos()
        )
        self._cell_sub = self.create_subscription(
            CellPose, cell_topic, self._on_cell, 10
        )
        self._timer = self.create_timer(1.0 / max(rate_hz, 1e-3), self._tick)

        self._state = _State.WAITING_MAZE
        self._n_cols: int | None = None
        self._goal: tuple[int, int] | None = None
        self._cell: tuple[int, int, int] | None = None  # (row, col, heading)

    # --------------------------------------------------------------- callbacks
    def _on_maze(self, msg: MazeGrid) -> None:
        self._n_cols = int(msg.cols)
        cells = np.asarray(msg.cells, dtype=np.int8).reshape(int(msg.rows), int(msg.cols))
        goals = np.argwhere(cells == 2)
        if goals.size == 0:
            self.get_logger().warn('Maze has no goal cell yet; waiting.')
            return
        self._goal = (int(goals[0, 0]), int(goals[0, 1]))
        if self._state == _State.WAITING_MAZE:
            self._transition(_State.WAITING_CELL)

    def _on_cell(self, msg: CellPose) -> None:
        self._cell = (int(msg.row), int(msg.col), int(msg.heading))
        if self._state == _State.WAITING_CELL:
            self._transition(_State.RUNNING)

    # ------------------------------------------------------------------- tick
    def _tick(self) -> None:
        if self._state != _State.RUNNING:
            return
        assert self._cell is not None and self._n_cols is not None and self._goal is not None
        r, c, h = self._cell
        if (r, c) == self._goal:
            self._cmd_pub.publish(Twist())  # stop
            self._transition(_State.DONE)
            self.get_logger().info('Goal reached.')
            return

        s = (r * self._n_cols + c) * 4 + h
        if not 0 <= s < self._pi.size:
            self.get_logger().warn(f'State index {s} out of range; stopping.')
            self._cmd_pub.publish(Twist())
            return
        action = int(self._pi[s])
        self._cmd_pub.publish(self._action_to_twist(action))

    # ----------------------------------------------------------------- helpers
    def _action_to_twist(self, action: int) -> Twist:
        msg = Twist()
        if action == int(Action.FORWARD):
            msg.linear.x = self._fwd
        elif action == int(Action.TURN_LEFT):
            msg.angular.z = self._turn
        elif action == int(Action.TURN_RIGHT):
            msg.angular.z = -self._turn
        return msg

    def _transition(self, new_state: _State) -> None:
        self.get_logger().info(f'{self._state.value} -> {new_state.value}')
        self._state = new_state


def main(args: list[str] | None = None) -> None:
    """Entry point used by ``ros2 run maze_mdp policy_runner``."""
    rclpy.init(args=args)
    node = PolicyRunner()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

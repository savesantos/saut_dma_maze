"""
Closed-loop policy executor for the AlphaBot2 maze stack.

State machine:

    WAITING_MAZE  -- got /maze --> WAITING_CELL
    WAITING_CELL  -- got /robot_cell --> RUNNING
    RUNNING       -- terminal cell reached --> DONE

The node loads a precomputed policy or Q-table from disk and looks up the
action for the current cell estimate. Two output modes are supported:

- ``mode='twist'`` (default, legacy): publish a ``geometry_msgs/Twist`` on
  ``/alphabot2/cmd_vel`` every control tick. Compatible with
  ``maze_sim_node`` which integrates Twist into discrete cell transitions.
- ``mode='action'``: publish one ``maze_msgs/DiscreteActionGoal`` per cell
  and wait for the matching ``DiscreteActionResult`` before issuing the
  next. Used with ``action_executor`` + ``cell_tracker`` on hardware and
  in the Gazebo stack.
"""

from __future__ import annotations

from enum import Enum

import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Empty

from maze_msgs.msg import CellPose, DiscreteActionGoal, DiscreteActionResult, MazeGrid

from maze_mdp.mdp import Action, _HEADING_DELTA
from maze_mdp.policy import load_policy, greedy_policy

# Matches DiscreteActionGoal.DRIVE_UNTIL_MARKER constant.
_DRIVE_UNTIL_MARKER: int = 3


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
        self.declare_parameter('mode', 'twist')
        self.declare_parameter('cmd_topic', '/alphabot2/cmd_vel')
        self.declare_parameter('goal_topic', '/action_goal')
        self.declare_parameter('result_topic', '/action_result')
        self.declare_parameter('cell_topic', '/robot_cell')
        self.declare_parameter('maze_topic', '/maze')
        self.declare_parameter('reset_topic', '/sim_reset')
        self.declare_parameter('exit_on_goal', False)
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

        self._mode = str(self.get_parameter('mode').value)
        if self._mode not in ('twist', 'action'):
            raise RuntimeError(f"mode must be 'twist' or 'action', got {self._mode!r}")

        cmd_topic = self.get_parameter('cmd_topic').get_parameter_value().string_value
        goal_topic = self.get_parameter('goal_topic').get_parameter_value().string_value
        result_topic = self.get_parameter('result_topic').get_parameter_value().string_value
        cell_topic = self.get_parameter('cell_topic').get_parameter_value().string_value
        maze_topic = self.get_parameter('maze_topic').get_parameter_value().string_value
        reset_topic = self.get_parameter('reset_topic').get_parameter_value().string_value
        self._exit_on_goal = bool(self.get_parameter('exit_on_goal').value)
        rate_hz = float(self.get_parameter('control_rate_hz').value)
        self._fwd = float(self.get_parameter('forward_speed').value)
        self._turn = float(self.get_parameter('turn_speed').value)

        # Output channel(s) depending on mode.
        if self._mode == 'twist':
            self._cmd_pub = self.create_publisher(Twist, cmd_topic, 10)
            self._goal_pub = None
        else:
            self._cmd_pub = None
            self._goal_pub = self.create_publisher(
                DiscreteActionGoal, goal_topic, 10)
            self.create_subscription(
                DiscreteActionResult, result_topic, self._on_result, 10)

        self._maze_sub = self.create_subscription(
            MazeGrid, maze_topic, self._on_maze, _latched_sub_qos()
        )
        # In action mode the cell publisher (cell_tracker) is latched
        # TRANSIENT_LOCAL; in twist mode it is VOLATILE (maze_sim_node).
        cell_qos = _latched_sub_qos() if self._mode == 'action' else 10
        self._cell_sub = self.create_subscription(
            CellPose, cell_topic, self._on_cell, cell_qos
        )
        self._reset_sub = self.create_subscription(
            Empty, reset_topic, self._on_reset, 10
        )
        self._tick_period = 1.0 / max(rate_hz, 1e-3)
        self._timer = self.create_timer(self._tick_period, self._tick)

        self._state = _State.WAITING_MAZE
        self._n_cols: int | None = None
        self._goal: tuple[int, int] | None = None
        self._cell: tuple[int, int, int] | None = None  # (row, col, heading)
        # Action-mode bookkeeping.
        self._goal_id: int = 0
        self._waiting_for_result: bool = False

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
        # In action mode, every cell update is a chance to dispatch the
        # next action (the previous one has finished by definition).
        if self._mode == 'action' and self._state == _State.RUNNING:
            self._maybe_dispatch_action()

    def _on_result(self, msg: DiscreteActionResult) -> None:
        if msg.goal_id != self._goal_id:
            return  # stale or other runner
        self._waiting_for_result = False
        if not msg.success:
            self.get_logger().warn(
                f'action {self._goal_id} failed '
                f'(failure_mode={msg.failure_mode})')
        # Do NOT dispatch here: cell_tracker publishes the updated
        # /robot_cell shortly after the result, and _on_cell handles
        # dispatch. Dispatching now would race against the cell update
        # and re-issue the just-finished action with stale pose.

    def _on_reset(self, _msg: Empty) -> None:
        """Restart the policy from whatever cell the sim resets us to."""
        if self._cmd_pub is not None:
            self._cmd_pub.publish(Twist())
        self.done = False
        self._waiting_for_result = False
        if self._timer.is_canceled():
            self._timer.reset()
        # Drop back into WAITING_CELL; the sim's post-reset CellPose will
        # promote us to RUNNING via _on_cell.
        if self._goal is None:
            self._transition(_State.WAITING_MAZE)
        else:
            self._transition(_State.WAITING_CELL)
        self.get_logger().info('Reset received; awaiting fresh cell pose.')

    # ------------------------------------------------------------------- tick
    def _tick(self) -> None:
        if self._state != _State.RUNNING:
            return
        if self._mode == 'twist':
            self._tick_twist()
        else:
            self._maybe_dispatch_action()

    def _tick_twist(self) -> None:
        assert (self._cell is not None and self._n_cols is not None
                and self._goal is not None and self._cmd_pub is not None)
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

    def _maybe_dispatch_action(self) -> None:
        if self._state != _State.RUNNING or self._waiting_for_result:
            return
        assert (self._cell is not None and self._n_cols is not None
                and self._goal is not None and self._goal_pub is not None)
        # Avoid the DDS discovery race: if no executor has matched our
        # goal publisher yet, defer; _tick() will retry next period.
        if self._goal_pub.get_subscription_count() == 0:
            return
        r, c, h = self._cell
        if (r, c) == self._goal:
            self._transition(_State.DONE)
            self.get_logger().info('Goal reached.')
            return
        s = (r * self._n_cols + c) * 4 + h
        if not 0 <= s < self._pi.size:
            self.get_logger().warn(f'State index {s} out of range; stopping.')
            return
        action = int(self._pi[s])
        # Final-approach substitution: if the policy says FORWARD and the
        # resulting cell is the goal, swap in DRIVE_UNTIL_MARKER so the
        # executor stops on the goal fiducial instead of at the next
        # intersection.
        if action == int(Action.FORWARD):
            dr, dc = _HEADING_DELTA[h]
            nr, nc = r + int(dr), c + int(dc)
            if (nr, nc) == self._goal:
                action = _DRIVE_UNTIL_MARKER
        self._goal_id += 1
        msg = DiscreteActionGoal()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.goal_id = self._goal_id
        msg.action = action
        self._goal_pub.publish(msg)
        self._waiting_for_result = True
        self.get_logger().info(
            f'dispatch goal_id={self._goal_id} action={action} '
            f'from cell=({r},{c},{h})')

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
        if new_state == _State.DONE:
            if self._cmd_pub is not None:
                self._cmd_pub.publish(Twist())
            if self._exit_on_goal:
                # Headless / batch mode: stop ticking and let main() exit.
                self._timer.cancel()
                self.done = True
            # Otherwise the timer keeps running but _tick() returns early
            # because state is DONE, so we sit idle until /sim_reset.

    @property
    def done(self) -> bool:
        return getattr(self, '_done_flag', False)

    @done.setter
    def done(self, value: bool) -> None:
        self._done_flag = bool(value)


def main(args: list[str] | None = None) -> None:
    """Entry point used by ``ros2 run maze_mdp policy_runner``."""
    rclpy.init(args=args)
    node = PolicyRunner()
    try:
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()

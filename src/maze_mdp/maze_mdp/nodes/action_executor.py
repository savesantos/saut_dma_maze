"""
ROS 2 wrapper around :class:`maze_mdp.control.executor.ActionExecutor`.

Subscribes:
- ``/action_goal``  (``maze_msgs/DiscreteActionGoal``): next discrete action.
- ``/line_pose``    (``std_msgs/Float32``): -1..+1, NaN if no line visible.
- ``/intersection`` (``std_msgs/Empty``):   crossing detected (all sensors on).
- ``/line_lost``    (``std_msgs/Empty``):   no line visible (optional hint).
- ``/goal_marker_seen`` (``std_msgs/Bool``): True when the goal fiducial is
  detected at final-approach proximity (only meaningful while executing
  ``DRIVE_UNTIL_MARKER``).

Publishes:
- ``/alphabot2/cmd_vel`` (``geometry_msgs/Twist``): wheel command.
- ``/action_result``     (``maze_msgs/DiscreteActionResult``): one per action.
"""

from __future__ import annotations

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from std_msgs.msg import Bool, Empty, Float32

from maze_msgs.msg import DiscreteActionGoal, DiscreteActionResult

from maze_mdp.control.executor import (
    ActionExecutor,
    ExecutorConfig,
    FailureMode,
    MotorCmd,
)


def _cmd_to_twist(cmd: MotorCmd) -> Twist:
    msg = Twist()
    msg.linear.x = float(cmd.linear)
    msg.angular.z = float(cmd.angular)
    return msg


class ActionExecutorNode(Node):
    """Drives one discrete maze action at a time using the IR strip events."""

    def __init__(self) -> None:
        super().__init__('action_executor')
        self.declare_parameter('cmd_topic', '/alphabot2/cmd_vel')
        self.declare_parameter('goal_topic', '/action_goal')
        self.declare_parameter('result_topic', '/action_result')
        self.declare_parameter('line_pose_topic', '/line_pose')
        self.declare_parameter('intersection_topic', '/intersection')
        self.declare_parameter('line_lost_topic', '/line_lost')
        self.declare_parameter('marker_topic', '/goal_marker_seen')
        self.declare_parameter('control_rate_hz', 20.0)
        self.declare_parameter('forward_speed', 0.10)
        self.declare_parameter('turn_speed', 0.60)
        self.declare_parameter('line_p_gain', 0.8)
        self.declare_parameter('action_timeout_s', 8.0)
        self.declare_parameter('line_lost_timeout_s', 0.5)
        self.declare_parameter('turn_exit_pose', 0.20)
        self.declare_parameter('turn_exit_min_excursion', 0.5)

        cfg = ExecutorConfig(
            forward_speed=float(self.get_parameter('forward_speed').value),
            turn_speed=float(self.get_parameter('turn_speed').value),
            line_p_gain=float(self.get_parameter('line_p_gain').value),
            action_timeout_s=float(
                self.get_parameter('action_timeout_s').value),
            line_lost_timeout_s=float(
                self.get_parameter('line_lost_timeout_s').value),
            turn_exit_pose=float(self.get_parameter('turn_exit_pose').value),
            turn_exit_min_excursion=float(
                self.get_parameter('turn_exit_min_excursion').value),
        )
        self._exec = ActionExecutor(cfg)

        cmd_topic = self.get_parameter('cmd_topic').value
        goal_topic = self.get_parameter('goal_topic').value
        result_topic = self.get_parameter('result_topic').value
        pose_topic = self.get_parameter('line_pose_topic').value
        cross_topic = self.get_parameter('intersection_topic').value
        lost_topic = self.get_parameter('line_lost_topic').value
        marker_topic = self.get_parameter('marker_topic').value

        self._cmd_pub = self.create_publisher(Twist, cmd_topic, 10)
        self._result_pub = self.create_publisher(
            DiscreteActionResult, result_topic, 10)
        self.create_subscription(
            DiscreteActionGoal, goal_topic, self._on_goal, 10)
        self.create_subscription(
            Float32, pose_topic, self._on_line_pose, 10)
        self.create_subscription(Empty, cross_topic, self._on_intersection, 10)
        self.create_subscription(Empty, lost_topic, self._on_line_lost, 10)
        self.create_subscription(
            Bool, marker_topic, self._on_marker_seen, 10)

        rate = float(self.get_parameter('control_rate_hz').value)
        self._tick_period = 1.0 / max(rate, 1e-3)
        self._timer = self.create_timer(self._tick_period, self._tick)

    # --------------------------------------------------------- subscribers
    def _on_goal(self, msg: DiscreteActionGoal) -> None:
        self.get_logger().info(
            f'goal {msg.goal_id}: action={msg.action}')
        cmd = self._exec.start(int(msg.action), int(msg.goal_id))
        self._publish(cmd)
        self._drain_result()

    def _on_line_pose(self, msg: Float32) -> None:
        cmd = self._exec.on_line_pose(float(msg.data))
        if self._exec.is_active or self._exec.state.value == 'done':
            self._publish(cmd)
        self._drain_result()

    def _on_intersection(self, _msg: Empty) -> None:
        cmd = self._exec.on_intersection()
        self._publish(cmd)
        self._drain_result()

    def _on_line_lost(self, _msg: Empty) -> None:
        cmd = self._exec.on_line_lost()
        self._publish(cmd)
        self._drain_result()

    def _on_marker_seen(self, msg: Bool) -> None:
        if not bool(msg.data):
            return
        cmd = self._exec.on_marker_seen()
        self._publish(cmd)
        self._drain_result()

    # ---------------------------------------------------------------- tick
    def _tick(self) -> None:
        cmd = self._exec.on_tick(self._tick_period)
        if self._exec.is_active:
            self._publish(cmd)
        self._drain_result()

    # ------------------------------------------------------------- helpers
    def _publish(self, cmd: MotorCmd) -> None:
        self._cmd_pub.publish(_cmd_to_twist(cmd))

    def _drain_result(self) -> None:
        r = self._exec.take_result()
        if r is None:
            return
        out = DiscreteActionResult()
        out.header.stamp = self.get_clock().now().to_msg()
        out.goal_id = int(r.goal_id)
        out.success = bool(r.success)
        out.failure_mode = int(
            r.failure_mode.value if isinstance(r.failure_mode, FailureMode)
            else r.failure_mode
        )
        self._result_pub.publish(out)
        self.get_logger().info(
            f'result goal={out.goal_id} success={out.success} '
            f'failure_mode={out.failure_mode}')


def main(args: list[str] | None = None) -> None:
    """Entry point: ``ros2 run maze_mdp action_executor``."""
    rclpy.init(args=args)
    node = ActionExecutorNode()
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

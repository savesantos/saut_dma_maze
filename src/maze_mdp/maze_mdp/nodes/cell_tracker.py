"""
ROS 2 wrapper around :class:`maze_mdp.tracking.CellTracker`.

Maintains the robot's discrete ``(row, col, heading)`` pose by composing
``DiscreteActionResult`` events onto an initial pose declared by the user
(launch arg or ``SetStartPose`` service). Publishes the current pose on
``/robot_cell`` (latched).
"""

from __future__ import annotations

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)

from maze_msgs.msg import CellPose as CellPoseMsg
from maze_msgs.msg import DiscreteActionGoal, DiscreteActionResult
from maze_msgs.srv import SetStartPose

from maze_mdp.tracking import CellPose, CellTracker


def _latched_qos() -> QoSProfile:
    return QoSProfile(
        depth=1,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
        history=HistoryPolicy.KEEP_LAST,
    )


class CellTrackerNode(Node):
    """Dead-reckon the robot's discrete maze pose from action results."""

    def __init__(self) -> None:
        super().__init__('cell_tracker')
        self.declare_parameter('rows', 0)
        self.declare_parameter('cols', 0)
        self.declare_parameter('start_row', 0)
        self.declare_parameter('start_col', 0)
        self.declare_parameter('start_heading', 0)
        self.declare_parameter('cell_topic', '/robot_cell')
        self.declare_parameter('result_topic', '/action_result')
        self.declare_parameter('frame_id', 'maze')

        rows = int(self.get_parameter('rows').value)
        cols = int(self.get_parameter('cols').value)
        if rows <= 0 or cols <= 0:
            raise RuntimeError(
                'cell_tracker requires positive rows and cols parameters.')

        start = CellPose(
            row=int(self.get_parameter('start_row').value),
            col=int(self.get_parameter('start_col').value),
            heading=int(self.get_parameter('start_heading').value),
        )
        self._tracker = CellTracker(rows, cols, start)
        self._frame_id = str(self.get_parameter('frame_id').value)

        cell_topic = str(self.get_parameter('cell_topic').value)
        result_topic = str(self.get_parameter('result_topic').value)
        self._pub = self.create_publisher(
            CellPoseMsg, cell_topic, _latched_qos())
        self.create_subscription(
            DiscreteActionGoal, '/action_goal', self._on_goal, 10)
        self.create_subscription(
            DiscreteActionResult, result_topic, self._on_result, 10)
        self.create_service(
            SetStartPose, '~/set_start_pose', self._on_set_start)

        # goal_id -> action, populated on goal, consumed on result.
        self._pending: dict[int, int] = {}

        self.get_logger().info(
            f'cell_tracker started at {start} on {rows}x{cols} grid')
        self._publish()
        # TRANSIENT_LOCAL latching is sometimes missed by late-joining
        # subscribers under DDS discovery races. Republish at a low rate
        # so policy_runner reliably picks up the initial pose.
        self._heartbeat = self.create_timer(1.0, self._publish)

    def _on_goal(self, msg: DiscreteActionGoal) -> None:
        self._pending[int(msg.goal_id)] = int(msg.action)

    def _on_result(self, msg: DiscreteActionResult) -> None:
        action = self._pending.pop(int(msg.goal_id), None)
        if action is None:
            # Either a stale duplicate, or a result without a known goal
            # (e.g. tracker started after the goal was sent). Idempotent.
            self.get_logger().debug(
                f'result for unknown goal_id {msg.goal_id}; ignoring')
            return
        self._tracker.apply(action, success=bool(msg.success))
        self._publish()

    def _on_set_start(self, request, response):
        try:
            self._tracker.reset(
                CellPose(int(request.row),
                         int(request.col),
                         int(request.heading))
            )
        except ValueError as exc:
            response.success = False
            response.message = str(exc)
            return response
        response.success = True
        response.message = ''
        self.get_logger().info(
            f'set_start_pose -> {self._tracker.pose}')
        self._publish()
        return response

    def _publish(self) -> None:
        pose = self._tracker.pose
        msg = CellPoseMsg()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self._frame_id
        msg.row = int(pose.row)
        msg.col = int(pose.col)
        msg.heading = int(pose.heading)
        msg.confidence = 1.0
        self._pub.publish(msg)


def main(args: list[str] | None = None) -> None:
    """Entry point: ``ros2 run maze_mdp cell_tracker``."""
    rclpy.init(args=args)
    node = CellTrackerNode()
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

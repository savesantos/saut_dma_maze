"""
ROS 2 node: measure camera-yaw scale by commanding a slow spin.

Use this to calibrate the ``gain`` parameter of :mod:`yaw_estimator`
for a given camera mount geometry. Procedure:

1. Place the robot in a feature-rich part of the maze (not staring at
   a uniform wall).
2. Run this node. It commands the robot to spin in place at a low rate
   for a fixed duration, then prints the ratio
   ``commanded_yaw / measured_yaw``. Set the ``yaw_estimator`` ``gain``
   parameter to that ratio.

Publishes ``/alphabot2/cmd_vel``; subscribes ``/yaw_delta``.

This node is intentionally separate from the runtime stack: do NOT run
it concurrently with ``action_executor`` (both would fight for
``/cmd_vel``).
"""

from __future__ import annotations

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from std_msgs.msg import Float32


class YawCalibratorNode(Node):
    """Spin in place and report the camera-vs-commanded yaw ratio."""

    def __init__(self) -> None:
        super().__init__('yaw_calibrator')
        self.declare_parameter('cmd_topic', '/alphabot2/cmd_vel')
        self.declare_parameter('yaw_topic', '/yaw_delta')
        self.declare_parameter('turn_speed', 0.4)
        self.declare_parameter('duration_s', 5.0)
        self.declare_parameter('warmup_s', 1.0)
        self.declare_parameter('rate_hz', 20.0)

        self._w = float(self.get_parameter('turn_speed').value)
        self._duration = float(self.get_parameter('duration_s').value)
        self._warmup = float(self.get_parameter('warmup_s').value)
        rate = float(self.get_parameter('rate_hz').value)
        self._dt = 1.0 / max(rate, 1e-3)

        self._cmd_pub = self.create_publisher(
            Twist, self.get_parameter('cmd_topic').value, 10)
        self.create_subscription(
            Float32,
            self.get_parameter('yaw_topic').value,
            self._on_yaw, 10)

        self._t = 0.0
        self._measured = 0.0
        self._done = False
        self.create_timer(self._dt, self._tick)
        self.get_logger().info(
            f'yaw_calibrator: spinning at {self._w:.3f} rad/s for '
            f'{self._duration:.1f} s after {self._warmup:.1f} s warmup')

    def _on_yaw(self, msg: Float32) -> None:
        if self._t < self._warmup or self._done:
            return
        self._measured += abs(float(msg.data))

    def _tick(self) -> None:
        if self._done:
            return
        self._t += self._dt
        cmd = Twist()
        if self._t < self._warmup + self._duration:
            cmd.angular.z = self._w
            self._cmd_pub.publish(cmd)
            return
        # Stop and report.
        cmd.angular.z = 0.0
        self._cmd_pub.publish(cmd)
        commanded = self._w * self._duration
        ratio = commanded / self._measured if self._measured > 1e-6 else float('inf')
        self.get_logger().info(
            f'commanded={commanded:.3f} rad, measured={self._measured:.3f} rad, '
            f'recommended gain = commanded/measured = {ratio:.3f}')
        self._done = True


def main(args: list[str] | None = None) -> None:
    """Entry point: ``ros2 run maze_mdp yaw_calibrator``."""
    rclpy.init(args=args)
    node = YawCalibratorNode()
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

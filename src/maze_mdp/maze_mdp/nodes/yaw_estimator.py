"""
ROS 2 node: publish per-frame yaw deltas from the AlphaBot2 camera.

Subscribes:
- ``/image/compressed`` (``sensor_msgs/CompressedImage``).

Publishes:
- ``/yaw_delta`` (``std_msgs/Float32``): yaw change since the previous
  frame, in radians. Positive = CCW (left). Consumed by
  ``action_executor`` to close the turn loop without depending on
  motor calibration.

Implementation: horizontal phase correlation between consecutive
grayscale frames (see :mod:`maze_mdp.perception.yaw_from_flow`). The
pixel-to-radian conversion uses the camera's horizontal field of view
only; it does not depend on motor characteristics, gear wear or battery
voltage, so the same node works across every robot in the lab.

OpenCV and cv_bridge are imported lazily so the unit tests on the
ROS-free helper need no ROS environment.
"""

from __future__ import annotations

import math

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Float32

from maze_mdp.perception.yaw_from_flow import YawFromFlow, YawFromFlowConfig


class YawEstimatorNode(Node):
    """Republish camera frames as per-frame yaw deltas."""

    def __init__(self) -> None:
        super().__init__('yaw_estimator')
        self.declare_parameter('image_topic', '/image/compressed')
        self.declare_parameter('yaw_topic', '/yaw_delta')
        # Raspberry Pi Camera v2 wide module: 62.2 deg HFOV.
        self.declare_parameter('camera_hfov_deg', 62.2)
        # Sign convention: +1 if features shifting right means CCW.
        # Default -1 matches a forward-facing camera under REP-103.
        self.declare_parameter('sign', -1.0)
        # Pitch / mount compensation. For a camera pitched down by p
        # radians, set gain ~ 1 / cos(p) to recover true yaw from the
        # horizontal pixel shift (the URDF's 45 deg pitch needs ~1.414).
        self.declare_parameter('gain', 1.0)
        self.declare_parameter('min_pixels', 0.5)
        self.declare_parameter('max_pixels', 200.0)
        # Optional downscale for speed; 1.0 keeps the original size.
        self.declare_parameter('downscale', 1.0)

        cfg = YawFromFlowConfig(
            camera_hfov_rad=math.radians(
                float(self.get_parameter('camera_hfov_deg').value)),
            sign=float(self.get_parameter('sign').value),
            gain=float(self.get_parameter('gain').value),
            min_pixels=float(self.get_parameter('min_pixels').value),
            max_pixels=float(self.get_parameter('max_pixels').value),
        )
        self._estimator = YawFromFlow(cfg)
        self._downscale = float(self.get_parameter('downscale').value)
        if self._downscale <= 0.0:
            self._downscale = 1.0

        self._cv2 = None
        self._bridge = None

        self._pub = self.create_publisher(
            Float32, self.get_parameter('yaw_topic').value, 10)
        self.create_subscription(
            CompressedImage,
            self.get_parameter('image_topic').value,
            self._on_image, 10)

    def _ensure_cv(self) -> None:
        if self._cv2 is not None:
            return
        import cv2  # noqa: WPS433
        from cv_bridge import CvBridge  # noqa: WPS433
        self._cv2 = cv2
        self._bridge = CvBridge()

    def _on_image(self, msg: CompressedImage) -> None:
        self._ensure_cv()
        cv2 = self._cv2
        frame = self._bridge.compressed_imgmsg_to_cv2(
            msg, desired_encoding='bgr8')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self._downscale != 1.0:
            gray = cv2.resize(
                gray, None,
                fx=self._downscale, fy=self._downscale,
                interpolation=cv2.INTER_AREA)
        delta = self._estimator.update(gray, cv2.phaseCorrelate)
        if delta is None:
            return
        self._pub.publish(Float32(data=float(delta)))


def main(args: list[str] | None = None) -> None:
    """Entry point: ``ros2 run maze_mdp yaw_estimator``."""
    rclpy.init(args=args)
    node = YawEstimatorNode()
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

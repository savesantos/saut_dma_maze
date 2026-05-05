"""
ROS 2 node that turns ArUco detections into discrete ``CellPose`` estimates.

Subscribes to the AlphaBot2 compressed image stream, runs OpenCV's ArUco
detector, looks each marker id up in a YAML marker map and publishes the
robot's discrete cell + heading on ``/robot_cell``.

OpenCV / cv_bridge / numpy are imported lazily so that the unit tests on the
ROS-free helpers in :mod:`maze_mdp.perception.aruco_to_cell` do not need
those heavy runtime deps.
"""

from __future__ import annotations

import math
from pathlib import Path

import rclpy
import yaml
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage

from maze_msgs.msg import CellPose

from maze_mdp.perception.aruco_to_cell import detection_to_cell, load_marker_map


class FiducialLocalizer(Node):
    """Detect ArUco markers and publish the robot's discrete cell estimate."""

    def __init__(self) -> None:
        super().__init__('fiducial_localizer')
        self.declare_parameter('image_topic', '/image/compressed')
        self.declare_parameter('cell_topic', '/robot_cell')
        self.declare_parameter('marker_map_path', '')
        self.declare_parameter('aruco_dict', 'DICT_4X4_50')

        image_topic = self.get_parameter('image_topic').get_parameter_value().string_value
        cell_topic = self.get_parameter('cell_topic').get_parameter_value().string_value
        map_path = self.get_parameter('marker_map_path').get_parameter_value().string_value

        self._marker_map = self._load_map(Path(map_path)) if map_path else {}
        if not self._marker_map:
            self.get_logger().warn('No marker_map_path configured; localizer will be a no-op.')

        # Lazy-imported OpenCV state.
        self._cv2 = None
        self._bridge = None
        self._detector = None

        self._publisher = self.create_publisher(CellPose, cell_topic, 10)
        self._subscription = self.create_subscription(
            CompressedImage, image_topic, self._on_image, 10
        )

    @staticmethod
    def _load_map(path: Path) -> dict:
        with path.open('r') as f:
            spec = yaml.safe_load(f) or {}
        return load_marker_map(spec.get('markers', []))

    def _ensure_cv(self) -> None:
        if self._cv2 is not None:
            return
        import cv2  # noqa: WPS433
        from cv_bridge import CvBridge  # noqa: WPS433

        self._cv2 = cv2
        self._bridge = CvBridge()
        dict_name = self.get_parameter('aruco_dict').get_parameter_value().string_value
        aruco_dict = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dict_name))
        params = cv2.aruco.DetectorParameters()
        self._detector = cv2.aruco.ArucoDetector(aruco_dict, params)

    def _on_image(self, msg: CompressedImage) -> None:
        if not self._marker_map:
            return
        self._ensure_cv()
        cv2 = self._cv2
        frame = self._bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='bgr8')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self._detector.detectMarkers(gray)
        if ids is None:
            return

        for marker_corners, marker_id in zip(corners, ids.flatten().tolist()):
            yaw = self._yaw_from_corners(marker_corners[0])
            est = detection_to_cell(int(marker_id), yaw, self._marker_map, confidence=1.0)
            if est is None:
                continue
            out = CellPose()
            out.header = msg.header
            out.row = int(est.row)
            out.col = int(est.col)
            out.heading = int(est.heading)
            out.confidence = float(est.confidence)
            self._publisher.publish(out)

    @staticmethod
    def _yaw_from_corners(corners) -> float:
        """Estimate marker yaw (radians) from its four image-plane corners."""
        # corners are in order: top-left, top-right, bottom-right, bottom-left.
        tl, tr, _br, _bl = corners
        dx = float(tr[0] - tl[0])
        dy = float(tr[1] - tl[1])
        # OpenCV image y grows downward; flip to keep math-standard yaw.
        return math.atan2(-dy, dx)


def main(args: list[str] | None = None) -> None:
    """Entry point used by ``ros2 run maze_mdp fiducial_localizer``."""
    rclpy.init(args=args)
    node = FiducialLocalizer()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

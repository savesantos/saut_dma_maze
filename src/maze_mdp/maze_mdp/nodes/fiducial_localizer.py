"""
ROS 2 node that turns ArUco detections into goal-marker proximity events.

Subscribes to the AlphaBot2 compressed image stream, runs OpenCV's ArUco
detector, and:

- Publishes ``std_msgs/Bool`` on ``/goal_marker_seen`` whenever the
  configured ``goal_marker_id`` is detected and its apparent size in the
  image (longer diagonal in pixels) is at least ``min_marker_pixel_size``.
  This is the final-approach trigger consumed by ``action_executor``.
- Optionally (``publish_cell:=true``) also publishes a discrete
  ``CellPose`` on ``cell_topic`` using a YAML marker map. This path is
  kept for legacy launch files; the MDP cell estimate is no longer the
  primary job of this node.

OpenCV / cv_bridge are imported lazily so the unit tests on the
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
from std_msgs.msg import Bool

from maze_msgs.msg import CellPose

from maze_mdp.perception.aruco_to_cell import detection_to_cell, load_marker_map


class FiducialLocalizer(Node):
    """Detect ArUco markers and emit goal-proximity / cell-pose events."""

    def __init__(self) -> None:
        super().__init__('fiducial_localizer')
        self.declare_parameter('image_topic', '/image/compressed')
        self.declare_parameter('cell_topic', '/robot_cell')
        self.declare_parameter('marker_topic', '/goal_marker_seen')
        self.declare_parameter('marker_map_path', '')
        self.declare_parameter('aruco_dict', 'DICT_4X4_50')
        # Final-approach trigger.
        self.declare_parameter('goal_marker_id', -1)
        self.declare_parameter('min_marker_pixel_size', 80.0)
        # Legacy CellPose publication is opt-in now.
        self.declare_parameter('publish_cell', False)

        image_topic = self.get_parameter(
            'image_topic').get_parameter_value().string_value
        cell_topic = self.get_parameter(
            'cell_topic').get_parameter_value().string_value
        marker_topic = self.get_parameter(
            'marker_topic').get_parameter_value().string_value
        map_path = self.get_parameter(
            'marker_map_path').get_parameter_value().string_value

        self._goal_marker_id = int(
            self.get_parameter('goal_marker_id').value)
        self._min_size_px = float(
            self.get_parameter('min_marker_pixel_size').value)
        self._publish_cell = bool(
            self.get_parameter('publish_cell').value)

        self._marker_map = (
            self._load_map(Path(map_path)) if map_path else {})
        if self._publish_cell and not self._marker_map:
            self.get_logger().warn(
                'publish_cell=True but no marker_map_path configured.')

        # Lazy-imported OpenCV state.
        self._cv2 = None
        self._bridge = None
        self._detector = None

        self._marker_pub = self.create_publisher(Bool, marker_topic, 10)
        self._cell_pub = (
            self.create_publisher(CellPose, cell_topic, 10)
            if self._publish_cell else None)
        self._subscription = self.create_subscription(
            CompressedImage, image_topic, self._on_image, 10
        )
        self._last_goal_seen: bool = False

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
        dict_name = self.get_parameter(
            'aruco_dict').get_parameter_value().string_value
        aruco_dict = cv2.aruco.getPredefinedDictionary(
            getattr(cv2.aruco, dict_name))
        params = cv2.aruco.DetectorParameters()
        self._detector = cv2.aruco.ArucoDetector(aruco_dict, params)

    def _on_image(self, msg: CompressedImage) -> None:
        self._ensure_cv()
        cv2 = self._cv2
        frame = self._bridge.compressed_imgmsg_to_cv2(
            msg, desired_encoding='bgr8')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self._detector.detectMarkers(gray)

        goal_seen = False
        if ids is not None:
            for marker_corners, marker_id in zip(
                    corners, ids.flatten().tolist()):
                mid = int(marker_id)
                if mid == self._goal_marker_id:
                    size_px = self._marker_size_px(marker_corners[0])
                    if size_px >= self._min_size_px:
                        goal_seen = True
                if self._cell_pub is not None and self._marker_map:
                    yaw = self._yaw_from_corners(marker_corners[0])
                    est = detection_to_cell(
                        mid, yaw, self._marker_map, confidence=1.0)
                    if est is not None:
                        out = CellPose()
                        out.header = msg.header
                        out.row = int(est.row)
                        out.col = int(est.col)
                        out.heading = int(est.heading)
                        out.confidence = float(est.confidence)
                        self._cell_pub.publish(out)

        # Debounce: only publish on edges (False -> True or True -> False).
        if goal_seen != self._last_goal_seen:
            self._marker_pub.publish(Bool(data=goal_seen))
            self._last_goal_seen = goal_seen

    @staticmethod
    def _marker_size_px(corners) -> float:
        """Return the longer diagonal of a marker quad, in pixels."""
        tl, tr, br, bl = corners
        d1 = math.hypot(float(br[0] - tl[0]), float(br[1] - tl[1]))
        d2 = math.hypot(float(bl[0] - tr[0]), float(bl[1] - tr[1]))
        return max(d1, d2)

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

"""
ROS 2 node: publish camera-based line-alignment for turn closure.

Subscribes:
- ``/image/compressed`` (``sensor_msgs/CompressedImage``).

Publishes:
- ``/line_alignment`` (``std_msgs/Float32``): line-tilt angle (rad)
    from image-vertical of the dominant ground line. ``0`` iff the line
    is vertical in the image (heading aligned). Positive means the line
    tilts to the right at the top. ``NaN`` when no usable line is in
    view (line lost, mid-turn, etc.).
- ``/line_alignment/debug/compressed``
  (``sensor_msgs/CompressedImage``): annotated camera frame showing
  the vertical image centre line (cyan), the fitted ground line
  (green), and the misalignment vector (yellow) from the bottom-
  centre anchor to where the ground line meets the bottom row.
  Subscribe in ``rqt_image_view`` while running Gazebo or the robot
  to see what the detector sees.

How the executor uses it: during a turn, the executor first waits for
the signal to go "misaligned" (large |angle| or NaN -- the originating
line has swept out of frame), then declares success when it returns to
near-zero for a few consecutive frames at slow rotation (the new line
is now ahead and the robot is not overshooting it). This is
motor-calibration agnostic: completion depends on geometric camera
alignment with the new line, not on how many radians the wheels
actually rotated.

OpenCV / cv_bridge are imported lazily so the unit tests on the
ROS-free helper need no ROS environment.
"""

from __future__ import annotations

import math
from typing import Optional

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Float32

from maze_mdp.perception.line_alignment import (
    LineAlignmentConfig,
    detect_dominant_line,
)


class LineAlignerNode(Node):
    """Republish camera frames as line-alignment angles."""

    def __init__(self) -> None:
        super().__init__('line_aligner')
        self.declare_parameter('image_topic', '/image/compressed')
        self.declare_parameter('alignment_topic', '/line_alignment')
        self.declare_parameter(
            'x_offset_topic', '/line_alignment/x_offset')
        self.declare_parameter(
            'debug_image_topic', '/line_alignment/debug/compressed')
        # Pixel-intensity threshold (0..255) below which a pixel is
        # treated as part of a black line. -1 selects Otsu's method.
        self.declare_parameter('threshold', 80)
        # Lower fraction of the image to analyse (0.5 = bottom half).
        self.declare_parameter('roi_top', 0.5)
        # Minimum number of dark pixels for a valid measurement.
        self.declare_parameter('min_pixels', 50)
        # Minimum eigenvalue-ratio (0..1) for the dominant axis to be
        # reported. Filters out ambiguous frames at the center of an
        # X intersection where two perpendicular lines are equally
        # visible (PCA principal direction flips between them).
        self.declare_parameter('min_linearity', 0.6)
        # Erosion kernel side (px) applied before contour finding.
        # Pinches off + intersections so the four arms become
        # independent contours. ~half the on-image line width.
        self.declare_parameter('erode_kernel', 7)
        # Optional downscale for speed; 1.0 keeps original resolution.
        self.declare_parameter('downscale', 1.0)
        # Rate limit: publish at most once per this many seconds.
        # Camera typically streams at 30 Hz; the executor only needs
        # a few measurements per second to drive turn closure.
        self.declare_parameter('publish_period_s', 0.05)
        # Publish the annotated debug image. Free in Gazebo; on the
        # robot it costs one extra JPEG encode per frame -- turn off
        # if bandwidth-constrained.
        self.declare_parameter('publish_debug_image', True)

        thr = int(self.get_parameter('threshold').value)
        cfg = LineAlignmentConfig(
            threshold=None if thr < 0 else thr,
            roi_top=float(self.get_parameter('roi_top').value),
            min_pixels=int(self.get_parameter('min_pixels').value),
            min_linearity=float(
                self.get_parameter('min_linearity').value),
            erode_kernel=int(
                self.get_parameter('erode_kernel').value),
        )
        self._cfg = cfg
        self._downscale = float(self.get_parameter('downscale').value)
        if self._downscale <= 0.0:
            self._downscale = 1.0
        self._publish_period = float(
            self.get_parameter('publish_period_s').value)
        self._last_pub_t: Optional[float] = None
        self._publish_debug = bool(
            self.get_parameter('publish_debug_image').value)

        self._cv2 = None
        self._bridge = None

        self._pub = self.create_publisher(
            Float32, self.get_parameter('alignment_topic').value, 10)
        self._x_offset_pub = self.create_publisher(
            Float32, self.get_parameter('x_offset_topic').value, 10)
        if self._publish_debug:
            self._debug_pub = self.create_publisher(
                CompressedImage,
                self.get_parameter('debug_image_topic').value, 1)
        else:
            self._debug_pub = None
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
        # Soft rate limit.
        now_s = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if self._last_pub_t is not None:
            if (now_s - self._last_pub_t) < self._publish_period:
                return

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

        det = detect_dominant_line(gray, self._cfg)
        # Turn closure uses heading alignment only (line tilt from
        # image-vertical). Using ``misalignment_rad`` (which folds in
        # lateral offset at the image bottom row) makes the FINAL_ALIGN
        # P-controller point the camera at the line rather than put
        # the robot on it, because in-place rotation cannot translate
        # the wheel base laterally. Pure heading alignment plus the
        # FORWARD line-follow PID is the correct decomposition: spin
        # to face the new corridor, then creep along it.
        angle = float('nan') if det is None else float(det.angle_rad)
        self._pub.publish(Float32(data=angle))
        x_off = float('nan') if det is None else float(det.x_offset_norm)
        self._x_offset_pub.publish(Float32(data=x_off))
        self._last_pub_t = now_s

        if self._debug_pub is not None:
            self._publish_debug_frame(gray, det, msg.header)

    def _publish_debug_frame(self, gray, det, header) -> None:
        cv2 = self._cv2
        h, w = gray.shape[:2]
        # Render onto a BGR copy of the analysed (possibly downscaled)
        # gray frame so the overlay matches what the detector sees.
        canvas = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        cx = w // 2
        # 1) Camera vertical centre line: where a perfectly aligned
        #    ground line should fall. Cyan, full image height.
        cv2.line(canvas, (cx, 0), (cx, h - 1), (255, 255, 0), 1)
        # 2) Bottom-centre anchor crosshair (yellow): the pivot for the
        #    misalignment angle.
        cv2.drawMarker(
            canvas, (cx, h - 1), (0, 255, 255),
            markerType=cv2.MARKER_TRIANGLE_UP,
            markerSize=14, thickness=2)

        if det is None:
            cv2.putText(
                canvas, 'NO LINE DETECTED', (8, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            ox, oy = det.centroid
            dx, dy = det.direction
            # 3) Fitted ground line (green), clipped to image bounds.
            t = max(w, h)
            p1 = (int(ox - dx * t), int(oy - dy * t))
            p2 = (int(ox + dx * t), int(oy + dy * t))
            cv2.line(canvas, p1, p2, (0, 255, 0), 2)
            # Line centroid (magenta dot).
            cv2.circle(canvas, (int(ox), int(oy)), 5, (255, 0, 255), -1)
            # 4) Misalignment vector (yellow): from the bottom-centre
            #    anchor to where the ground line meets the bottom row.
            #    Length and tilt of this arrow are the misalignment
            #    angle; aligned -> zero-length arrow at the anchor.
            x_bot = int(max(0, min(w - 1, int(round(det.x_bottom)))))
            cv2.arrowedLine(
                canvas, (cx, h - 1), (x_bot, h - 1),
                (0, 255, 255), 2, tipLength=0.25)
            mis_deg = math.degrees(det.misalignment_rad)
            ang_deg = math.degrees(det.angle_rad)
            cv2.putText(
                canvas,
                f'mis={mis_deg:+.1f}deg tilt={ang_deg:+.1f}deg '
                f'off={det.x_offset_norm:+.2f}',
                (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (0, 255, 0), 2)

        out = self._bridge.cv2_to_compressed_imgmsg(canvas, dst_format='jpeg')
        out.header = header
        self._debug_pub.publish(out)


def main(args: list[str] | None = None) -> None:
    """Entry point: ``ros2 run maze_mdp line_aligner``."""
    rclpy.init(args=args)
    node = LineAlignerNode()
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

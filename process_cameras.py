#!/usr/bin/env python3
from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Bool


class CrossDetector(Node):
    def __init__(self) -> None:
        super().__init__("cross_detector")

        default_ref = Path(__file__).resolve().parent / "image1.png"

        self.declare_parameter("image_topic", "alphabot2/image_raw")
        self.declare_parameter("output_topic", "cross_detected")
        self.declare_parameter("process_rate_hz", 5.0)

        self.declare_parameter("reference_image", str(default_ref))
        self.declare_parameter("black_threshold", 90)
        self.declare_parameter("bar_fill_ratio", 0.25)
        self.declare_parameter("corner_max_black", 0.25)
        self.declare_parameter("template_match_threshold", 0.45)
        self.declare_parameter("require_template_match", False)
        self.declare_parameter("debug", False)

        image_topic = self.get_parameter("image_topic").value
        output_topic = self.get_parameter("output_topic").value

        self._process_interval = 1.0 / max(
            0.1, float(self.get_parameter("process_rate_hz").value)
        )

        self._black_threshold = int(self.get_parameter("black_threshold").value)
        self._bar_fill_ratio = float(self.get_parameter("bar_fill_ratio").value)
        self._corner_max_black = float(self.get_parameter("corner_max_black").value)
        self._template_threshold = float(self.get_parameter("template_match_threshold").value)
        self._require_template = bool(self.get_parameter("require_template_match").value)
        self._debug = bool(self.get_parameter("debug").value)

        self._bridge = CvBridge()
        self._last_process_time = 0.0

        self._templates = self._load_templates(str(self.get_parameter("reference_image").value))

        self._pub = self.create_publisher(Bool, output_topic, 10)
        self._sub = self.create_subscription(Image, image_topic, self._image_callback, 10)

        self.get_logger().info(
            f"CrossDetector running: {image_topic} -> {output_topic} @ ~{1/self._process_interval:.1f} Hz"
        )

    # -----------------------------
    # Template loading
    # -----------------------------
    def _load_templates(self, path: str) -> list[np.ndarray]:
        p = Path(path)
        if not p.exists():
            self.get_logger().warn(f"Reference image not found: {p}")
            return []

        gray = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            self.get_logger().warn("Failed to load reference image")
            return []

        templates = []
        for scale in (0.4, 0.6, 0.8, 1.0):
            h, w = gray.shape
            resized = cv2.resize(
                gray,
                (max(32, int(w * scale)), max(32, int(h * scale))),
                interpolation=cv2.INTER_AREA,
            )
            templates.append(resized)

        self.get_logger().info(f"Loaded {len(templates)} templates")
        return templates

    # -----------------------------
    # ROS callback
    # -----------------------------
    def _image_callback(self, msg: Image) -> None:
        now = time.monotonic()

        if now - self._last_process_time < self._process_interval:
            return
        self._last_process_time = now

        # --- Safe conversion (Humble-friendly) ---
        try:
            frame = self._bridge.imgmsg_to_cv2(msg)
        except Exception as exc:
            self.get_logger().warn(f"cv_bridge conversion failed: {exc}")
            return

        # Normalize formats
        try:
            if frame.ndim == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            elif frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        except Exception as exc:
            self.get_logger().warn(f"format normalization failed: {exc}")
            return

        detected, score = self._detect_cross(frame)

        out = Bool()
        out.data = detected
        self._pub.publish(out)

        if self._debug or detected:
            self.get_logger().info(f"cross={detected} score={score:.2f}")

    # -----------------------------
    # Detection logic
    # -----------------------------
    def _detect_cross(self, bgr: np.ndarray) -> tuple[bool, float]:
        gray = cv2.GaussianBlur(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), (5, 5), 0)

        structural = self._structural_detect(gray)
        template_score = self._template_score(gray) if self._templates else 0.0

        template_ok = template_score >= self._template_threshold

        if self._require_template and self._templates:
            detected = structural and template_ok
        elif self._templates:
            detected = structural or template_ok
        else:
            detected = structural

        return detected, max(template_score, 1.0 if structural else 0.0)

    # -----------------------------
    # Structural detection
    # -----------------------------
    def _structural_detect(self, gray: np.ndarray) -> bool:
        h, w = gray.shape

        binary = gray < self._black_threshold

        cy, cx = h // 2, w // 2
        band_h = max(3, h // 14)
        band_w = max(3, w // 14)

        horizontal = binary[cy - band_h : cy + band_h + 1, :]
        vertical = binary[:, cx - band_w : cx + band_w + 1]

        h_fill = float(np.mean(horizontal))
        v_fill = float(np.mean(vertical))

        # relaxed corner constraint (important for real cameras)
        corners = [
            binary[0 : h // 5, 0 : w // 5],
            binary[0 : h // 5, 4 * w // 5 : w],
        ]
        corners_ok = all(float(np.mean(c)) <= self._corner_max_black for c in corners)

        center_ok = bool(binary[cy, cx])

        bars_ok = (
            h_fill >= self._bar_fill_ratio and v_fill >= self._bar_fill_ratio
        )

        return bars_ok and center_ok and corners_ok

    # -----------------------------
    # Template matching
    # -----------------------------
    def _template_score(self, gray: np.ndarray) -> float:
        best = 0.0

        for t in self._templates:
            th, tw = t.shape

            if th >= gray.shape[0] or tw >= gray.shape[1]:
                continue

            res = cv2.matchTemplate(gray, t, cv2.TM_CCOEFF_NORMED)
            best = max(best, float(res.max()))

        return best


# -----------------------------
# Main
# -----------------------------
def main(args=None):
    rclpy.init(args=args)
    node = CrossDetector()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
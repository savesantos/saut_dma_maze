"""Unit tests for the ROS-free dominant-line-angle detector."""

import math

import numpy as np

from maze_mdp.perception.line_alignment import (
    LineAlignmentConfig,
    detect_dominant_line,
    detect_dominant_line_angle,
)


def _vertical_line_image(h=80, w=80, x=40, width=3, value=0, bg=255):
    img = np.full((h, w), bg, dtype=np.uint8)
    img[:, max(0, x - width // 2): x + width // 2 + 1] = value
    return img


def _tilted_line_image(h=80, w=80, angle_rad=0.0, value=0, bg=255):
    """Draw a single thin line tilted by ``angle_rad`` from vertical."""
    img = np.full((h, w), bg, dtype=np.uint8)
    cx, cy = w / 2, h / 2
    # Parametric line through (cx, cy) with direction (sin(angle), cos(angle))
    # (so angle=0 -> vertical with dy>0).
    dx, dy = math.sin(angle_rad), math.cos(angle_rad)
    for t in range(-h, h):
        x = int(round(cx + t * dx))
        y = int(round(cy + t * dy))
        if 0 <= x < w and 0 <= y < h:
            for ox in (-1, 0, 1):
                xx = x + ox
                if 0 <= xx < w:
                    img[y, xx] = value
    return img


def test_no_line_returns_none():
    blank = np.full((60, 80), 255, dtype=np.uint8)
    assert detect_dominant_line_angle(blank) is None


def test_vertical_line_has_zero_angle():
    img = _vertical_line_image(x=40)
    out = detect_dominant_line_angle(img)
    assert out is not None
    angle, offset = out
    assert abs(angle) < 0.05  # tiny rasterisation residual
    assert abs(offset) < 0.05  # centred


def test_offset_line_reports_offset():
    img = _vertical_line_image(x=60, w=80)
    out = detect_dominant_line_angle(img)
    assert out is not None
    _, offset = out
    # x=60 with image width 80 -> normalised offset = (60 - 40)/40 = 0.5.
    assert offset > 0.3


def test_tilted_line_has_signed_angle():
    target = math.radians(20.0)
    img = _tilted_line_image(angle_rad=target)
    out = detect_dominant_line_angle(img)
    assert out is not None
    angle, _ = out
    # Drawn angle should be recovered within a few degrees.
    assert math.isclose(angle, target, abs_tol=math.radians(3.0))


def test_negative_tilt_has_negative_angle():
    target = math.radians(-20.0)
    img = _tilted_line_image(angle_rad=target)
    out = detect_dominant_line_angle(img)
    assert out is not None
    angle, _ = out
    assert math.isclose(angle, target, abs_tol=math.radians(3.0))


def test_min_pixels_threshold_rejects_tiny_lines():
    img = np.full((80, 80), 255, dtype=np.uint8)
    # A single 1-pixel dark cluster -- well below min_pixels.
    img[60, 40] = 0
    cfg = LineAlignmentConfig(threshold=80, min_pixels=10)
    assert detect_dominant_line_angle(img, cfg) is None


def test_roi_top_crops_upper_half():
    # Vertical line only in the *upper* half of the image. roi_top=0.5
    # should ignore it.
    img = np.full((80, 80), 255, dtype=np.uint8)
    img[0:30, 39:42] = 0
    cfg = LineAlignmentConfig(threshold=80, roi_top=0.5, min_pixels=10)
    assert detect_dominant_line_angle(img, cfg) is None


def test_x_intersection_is_rejected_by_linearity_gate():
    """At the dead centre of an X, PCA's principal direction is ambiguous.

    The eigenvalue-ratio gate must reject the frame so the executor
    stays in LEAVE and only completes the turn when the camera is
    pointed cleanly down a single line.
    """
    img = np.full((120, 120), 255, dtype=np.uint8)
    # Equal-strength vertical and horizontal lines crossing in the centre.
    img[:, 58:62] = 0
    img[58:62, :] = 0
    cfg = LineAlignmentConfig(threshold=80, roi_top=0.0,
                              min_pixels=50, min_linearity=0.6)
    assert detect_dominant_line_angle(img, cfg) is None


def test_full_image_prefers_long_line_over_short_stripe():
    """Long new-corridor line dominates over short stripe under chassis.

    Reproduces the 90 deg turn-through-intersection geometry that the
    previous bottom-half ROI got wrong (it locked onto the horizontal
    stripe and only completed at 180 deg).
    """
    img = np.full((120, 120), 255, dtype=np.uint8)
    # Long vertical line in the upper 2/3 (new corridor extending ahead).
    img[0:80, 58:62] = 0
    # Short horizontal stripe in the bottom (old line under the chassis).
    img[110:115, 30:90] = 0
    cfg = LineAlignmentConfig(threshold=80, roi_top=0.0,
                              min_pixels=50, min_linearity=0.6)
    out = detect_dominant_line_angle(img, cfg)
    assert out is not None
    angle, _ = out
    # The dominant axis is the long vertical line: angle near 0.
    assert abs(angle) < math.radians(10.0)


# ----------------------------- misalignment (combined angle + offset)

def test_misalignment_zero_when_centred_vertical():
    img = _vertical_line_image(x=40, w=80, h=80)
    det = detect_dominant_line(img)
    assert det is not None
    # Centred AND vertical -> misalignment near zero.
    assert abs(det.misalignment_rad) < math.radians(2.0)


def test_misalignment_nonzero_for_offset_vertical_line():
    """A vertical line laterally offset must register as misaligned."""
    img = _vertical_line_image(x=60, w=80, h=80)  # 20 px right of centre
    det = detect_dominant_line(img)
    assert det is not None
    # x_bottom ~ 60, anchor at (40, 79); atan2(20, 80) ~ 0.245 rad.
    assert det.misalignment_rad > math.radians(10.0)
    # Sign convention: line drifts right -> positive misalignment.
    assert det.misalignment_rad > 0


def test_misalignment_nonzero_for_tilted_centred_line():
    """A tilted line through image centre must register as misaligned."""
    target_tilt = math.radians(20.0)
    img = _tilted_line_image(h=120, w=120, angle_rad=target_tilt)
    det = detect_dominant_line(img)
    assert det is not None
    # Tilt right -> bottom of the line is left of centre -> negative
    # misalignment (anchor->x_bottom points left). The exact magnitude
    # depends on roi_top and rasterisation; just check it is not zero.
    assert abs(det.misalignment_rad) > math.radians(5.0)


def test_misalignment_sign_matches_offset():
    right = detect_dominant_line(_vertical_line_image(x=60, w=80, h=80))
    left = detect_dominant_line(_vertical_line_image(x=20, w=80, h=80))
    assert right is not None and left is not None
    assert right.misalignment_rad > 0
    assert left.misalignment_rad < 0


def test_x_bottom_reported_for_debug_overlay():
    img = _vertical_line_image(x=55, w=80, h=80)
    det = detect_dominant_line(img)
    assert det is not None
    # Detector should report x_bottom within a couple of pixels of the
    # actual drawn column (rasterisation tolerance).
    assert abs(det.x_bottom - 55) < 3.0

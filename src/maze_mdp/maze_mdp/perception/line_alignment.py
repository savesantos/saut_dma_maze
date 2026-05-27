"""
ROS-free line-alignment detector for camera-based turn closure.

Given a grayscale frame from the AlphaBot2's forward camera (pitched down
so it sees the floor ~6-20 cm ahead), this module fits the dominant
black-on-white line in the lower portion of the frame and reports how
the line is positioned relative to a vertical reference drawn down the
middle of the image.

Two related signals are produced:

- ``angle_rad``: tilt of the fitted line from image-vertical (positive
  = tilts right). Sensitive to robot heading only.
- ``misalignment_rad``: the geometric misalignment between the image's
  vertical centre line and the fitted ground line, defined as the
  angle from the bottom-centre pixel ``(w/2, h-1)`` to where the line
  crosses the bottom row of the image. This combines BOTH the line's
  tilt and its lateral offset into a single, physically meaningful
  number: ``0`` iff the line is centred AND vertical, and growing
  monotonically as the line drifts away from the camera's forward axis.

``misalignment_rad`` is what the action executor consumes to close a
turn: "robot is pointing down the new corridor" iff the line lies
along the camera's vertical centre line.

Designed to be unit-testable with synthetic numpy frames: the only
optional OpenCV dependency is for a robust threshold, which the caller
can override.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class LineAlignmentConfig:
    """Static tuning for :func:`detect_dominant_line_angle`.

    ``threshold`` is the 8-bit grayscale value below which a pixel is
    considered part of a (black) line. Bumps need to be calibrated for
    real-world lighting; 80 works well for the Gazebo world (pure
    black/white). Setting ``threshold`` to ``None`` enables Otsu's
    method on the ROI (requires OpenCV at runtime).

    ``roi_top`` selects the lower fraction of the image to analyse
    (default 0.5 = bottom half). Cropping the upper half removes the
    horizon and most perspective distortion, leaving the near-field
    line that the robot is about to traverse.

    ``min_pixels`` is the minimum number of dark pixels in the ROI for
    a measurement to be returned; below it, the helper returns
    ``None`` (no usable line in view).

    ``min_linearity`` is the minimum eigenvalue-ratio measure
    ``(lambda_max - lambda_min) / (lambda_max + lambda_min)`` required
    for the dominant axis to be trusted. A single straight line has
    ``min_linearity`` close to 1.0; a balanced ``X`` intersection where
    two perpendicular lines are equally visible drops to ~0.0 and the
    eigendecomposition's principal direction becomes ambiguous (it can
    flip between the two lines from frame to frame). Returning
    ``None`` in that case prevents the executor from declaring a turn
    complete on a spurious alignment at the center of an intersection.
    """

    threshold: Optional[int] = 80
    roi_top: float = 0.5
    min_pixels: int = 50
    min_linearity: float = 0.6
    # Morphological erosion kernel side (pixels) applied to the
    # binary mask before ``findContours``. The maze floor lines are
    # thick relative to their length, and at an intersection the +
    # shape comes back as one ambiguous connected component. Eroding
    # by ~half the line width pinches off the centre of the + so the
    # four arms become independent contours and the linearity gate
    # plus the vertical-extent selection (see ``_detect_with_opencv``)
    # can pick the corridor line going into the distance. 0 disables.
    erode_kernel: int = 0


@dataclass(frozen=True)
class LineDetection:
    """Full result of a successful line detection.

    Coordinates are in the **input image** frame (after ROI cropping
    has been undone by adding ``roi_y_offset``), so the caller can
    draw straight onto the source frame without re-translating.

    Fields:
    - ``angle_rad``: signed angle from image-vertical (positive =
      tilted right). Same convention as
      :func:`detect_dominant_line_angle`.
    - ``x_offset_norm``: centroid x offset from image centre, in
      ``[-1, +1]``.
    - ``misalignment_rad``: angle from the bottom-centre pixel of the
      image to the point where the fitted line meets the bottom row.
      0 iff the line is centred AND vertical; grows as the line tilts
      OR drifts laterally. This is the single scalar the turn-closure
      FSM consumes.
    - ``centroid``: ``(x, y)`` of the line centroid in input-image
      pixels.
    - ``direction``: unit vector ``(dx, dy)`` of the principal axis,
      with ``dy >= 0`` (points toward the bottom of the image).
    - ``x_bottom``: column (in input-image pixels) where the fitted
      line crosses the bottom row ``y = h - 1``. Useful for debug
      overlays; equals ``w / 2`` for a perfectly aligned line.
    """

    angle_rad: float
    x_offset_norm: float
    misalignment_rad: float
    centroid: Tuple[float, float]
    direction: Tuple[float, float]
    x_bottom: float


def detect_dominant_line_angle(
    gray,
    config: LineAlignmentConfig | None = None,
) -> Optional[Tuple[float, float]]:
    """Return ``(angle_rad, x_offset_norm)`` for the dominant dark line.

    Thin compatibility wrapper around :func:`detect_dominant_line`.
    """
    det = detect_dominant_line(gray, config)
    if det is None:
        return None
    return det.angle_rad, det.x_offset_norm


def detect_dominant_line(
    gray,
    config: LineAlignmentConfig | None = None,
) -> Optional[LineDetection]:
    """Detect the dominant dark line and return its full geometry.

    ``gray`` is a 2D numpy array (single-channel image, any numeric
    dtype). The returned ``angle_rad`` is measured from image-vertical
    (positive = tilted right). ``x_offset_norm`` is the line centroid's
    horizontal offset from the image centre, normalised to ``[-1, +1]``
    (negative = line is left of centre).

    Returns ``None`` when no usable line is in the ROI.

    Strategy: threshold the ROI, run ``cv2.findContours`` and pick the
    longest connected component, then fit a line through it with
    ``cv2.fitLine`` (L2 distance). This naturally rejects noise specks
    and -- critically for turn closure -- separates the short stripe
    of the line under the chassis from the long stretch of the new
    corridor line extending into the distance. The longer contour
    wins.

    Falls back to a pure-numpy PCA path when OpenCV is unavailable
    (e.g. unit tests in a minimal env), which keeps the algorithm
    module importable without a hard cv2 dependency.
    """
    import numpy as np  # noqa: WPS433

    cfg = config or LineAlignmentConfig()
    if gray.ndim != 2:
        raise ValueError('detect_dominant_line_angle expects a 2D array')

    h, w = gray.shape
    if h < 4 or w < 4:
        return None

    y0 = int(h * cfg.roi_top)
    if y0 >= h:
        return None
    roi = gray[y0:, :]

    try:
        import cv2  # noqa: WPS433
        return _detect_with_opencv(roi, h, w, y0, cfg, cv2, np)
    except ImportError:
        return _detect_with_numpy(roi, h, w, y0, cfg, np)


def _detect_with_opencv(roi, h, w, y0, cfg, cv2, np):
    """OpenCV-backed detector: threshold + contours + ``fitLine``."""
    thr = cfg.threshold
    if thr is None:
        _, mask = cv2.threshold(
            roi.astype('uint8'), 0, 255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    else:
        mask = (roi < int(thr)).astype('uint8') * 255

    # Light morphological clean-up: close 1-pixel gaps, then drop
    # specks that survived. Kernel size chosen so JPEG ringing on the
    # real Pi Camera does not split a single line in half.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Pinch off + intersections so the four arms become independent
    # contours. Without this, at a crossing the longest contour is
    # the whole + and PCA's principal direction is ambiguous --
    # the detector returns ``None`` for almost every frame mid-turn.
    if cfg.erode_kernel > 0:
        k_erode = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (int(cfg.erode_kernel), int(cfg.erode_kernel)))
        mask = cv2.erode(mask, k_erode)

    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None

    # Pick the contour with the largest vertical extent (height of
    # its axis-aligned bounding box). The corridor line ahead of the
    # robot extends from close (bottom of image, camera pitched 45 deg
    # down) to far (top of image) and therefore has the largest
    # y-span. The perpendicular cross-bar under the chassis has a
    # y-span equal to its width (small). This selection picks the
    # corridor direction even when the cross-bar momentarily has
    # more pixels overall.
    def _y_extent(c):
        ys = c[:, 0, 1]
        return int(ys.max() - ys.min())

    best = max(contours, key=_y_extent)
    if best.shape[0] < cfg.min_pixels:
        return None

    # Linearity gate from min-area-rect aspect ratio. A clean line has
    # one rect side >> the other; an X intersection has them similar.
    (_cx, _cy), (rw, rh), _ang = cv2.minAreaRect(best)
    long_side = max(rw, rh)
    short_side = max(min(rw, rh), 1e-6)
    if long_side < 1e-6:
        return None
    linearity = (long_side - short_side) / (long_side + short_side)
    if linearity < cfg.min_linearity:
        return None

    # cv2.fitLine returns (vx, vy, x0, y0): the unit direction vector
    # and a point on the line. DIST_L2 is plain least-squares; for
    # extra robustness against the residual stripe stub the caller can
    # post-process, but L2 is sufficient once contours have separated.
    vx, vy, x0, y0_roi = cv2.fitLine(
        best, cv2.DIST_L2, 0, 0.01, 0.01).ravel()
    if vy < 0.0:
        vx, vy = -vx, -vy  # principal direction points "down" in image

    angle = math.atan2(float(vx), float(vy))
    # Translate the centroid y back into input-image coordinates.
    return _build_detection(
        angle, float(x0), float(y0_roi) + float(y0),
        h, w, float(vx), float(vy))


def _build_detection(angle, x0, y0_centroid_img, h, w, vx, vy):
    """Construct a LineDetection from raw fit parameters.

    ``misalignment_rad`` is the angle, measured at the bottom-centre
    pixel ``(w/2, h-1)``, between the image's vertical centre line
    (pointing straight up) and the ray to where the fitted line meets
    the bottom row ``y = h - 1``. The line is parametrised as
    ``(x, y) = (x0, y0) + t (vx, vy)`` with ``vy >= 0``; setting
    ``y = h - 1`` gives ``x_bottom = x0 + vx (h - 1 - y0) / vy``. To
    avoid a division by zero for near-horizontal lines we feed atan2
    the un-normalised numerator and denominator directly.
    """
    cx = 0.5 * w
    x_offset_norm = (float(x0) - cx) / cx
    # numer = vy*(x0 - cx) + vx*(h - 1 - y0)   (since x_bottom - cx
    #          = (x0 - cx) + vx*(h - 1 - y0)/vy)
    # denom = vy * h   (vertical distance from bottom-centre to bottom
    #          row, scaled by vy so we can skip the division)
    numer = vy * (float(x0) - cx) + vx * (float(h) - 1.0 - float(y0_centroid_img))
    denom = vy * float(h)
    misalignment = math.atan2(numer, denom) if denom != 0.0 else math.atan2(numer, 1e-9)
    # x_bottom for debug overlays; clamp gracefully when vy ~ 0.
    if abs(vy) > 1e-6:
        x_bottom = float(x0) + vx * (float(h) - 1.0 - float(y0_centroid_img)) / vy
    else:
        x_bottom = float(x0)
    return LineDetection(
        angle_rad=float(angle),
        x_offset_norm=float(x_offset_norm),
        misalignment_rad=float(misalignment),
        centroid=(float(x0), float(y0_centroid_img)),
        direction=(float(vx), float(vy)),
        x_bottom=float(x_bottom),
    )


def _detect_with_numpy(roi, h, w, y0, cfg, np):
    """Fallback for environments without OpenCV (unit-test safety net).

    Uses PCA on the threshold mask -- less robust than the OpenCV
    contour path because all dark pixels (including noise) contribute
    equally, but adequate for synthetic test images.
    """
    thr = cfg.threshold
    if thr is None:
        return None  # no Otsu without cv2
    mask = roi < int(thr)

    n = int(mask.sum())
    if n < cfg.min_pixels:
        return None

    ys, xs = np.nonzero(mask)
    pts = np.column_stack(
        [xs.astype(np.float64), ys.astype(np.float64)])
    mean = pts.mean(axis=0)
    centered = pts - mean
    cov = centered.T @ centered / float(n)
    eigvals, eigvecs = np.linalg.eigh(cov)
    lam_min = float(eigvals[0])
    lam_max = float(eigvals[-1])
    denom = lam_max + lam_min
    linearity = 0.0 if denom <= 0.0 else (lam_max - lam_min) / denom
    if linearity < cfg.min_linearity:
        return None
    # Principal direction = eigenvector with largest eigenvalue
    # (column index -1 in numpy's ascending-sorted output).
    dx, dy = float(eigvecs[0, -1]), float(eigvecs[1, -1])
    # Eigenvector direction is sign-ambiguous; fix so dy >= 0 (the
    # principal direction points "down" in image coords, i.e. toward
    # the bottom of the ROI which is closest to the robot).
    if dy < 0.0:
        dx, dy = -dx, -dy

    angle = math.atan2(dx, dy)
    return _build_detection(
        angle, float(mean[0]), float(mean[1]) + float(y0),
        h, w, float(dx), float(dy))


__all__ = [
    'LineAlignmentConfig',
    'LineDetection',
    'detect_dominant_line',
    'detect_dominant_line_angle',
]

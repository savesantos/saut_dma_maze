"""
Monocular yaw-rate estimator from horizontal image phase correlation.

The AlphaBot2 fleet has no IMU and no usable wheel odometry, so the only
heading signal available on hardware is the camera. This module wraps
OpenCV's :func:`cv2.phaseCorrelate` into a tiny stateful helper that
returns the yaw delta (radians) between two consecutive grayscale frames.

The conversion from pixel shift to radians uses the camera's horizontal
field of view, which is a fixed property of the optics (62.2 deg for the
Raspberry Pi Camera v2). It is independent of motor calibration, gear
wear, or battery voltage -- exactly the properties that vary across the
robots in the lab and which make commanded-yaw integration unreliable.

The estimator is intentionally ROS-free so it can be unit-tested with
synthetic frames and reused by both the hardware and the Gazebo node
wrappers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional


@dataclass(frozen=True)
class YawFromFlowConfig:
    """Static tuning for :class:`YawFromFlow`.

    ``camera_hfov_rad`` is the camera horizontal field of view in
    radians (default: 62.2 deg, Raspberry Pi Camera v2 wide module).

    ``sign`` multiplies the pixel-to-radian conversion. With a
    forward-facing camera and OpenCV's image convention, features
    shift in the +x direction when the robot rotates clockwise
    (negative yaw under REP-103), so ``sign=-1.0`` yields positive
    deltas for CCW rotations.

    ``gain`` scales the converted yaw to compensate for camera mount
    geometry that violates the ideal forward-looking-pinhole assumption
    (the bare ``width / HFOV`` conversion is exact only when the camera
    optical axis is horizontal). For a camera pitched down by ``p``,
    pure-yaw horizontal pixel shift is reduced roughly by ``cos(p)``
    relative to a forward camera, so set ``gain = 1 / cos(p)`` to
    recover the true yaw (for the URDF's 45 deg pitch that is ~1.414).
    Defaults to 1.0 (no compensation). This is a per-mount property,
    not a per-motor property, so the same value applies to every
    robot built with the same hardware.

    ``min_pixels`` clamps sub-pixel shifts to zero (rejects
    floating-point noise from a static scene); ``max_pixels`` discards
    anything larger (rejects scene cuts, shutter glitches, frame drops).
    """

    camera_hfov_rad: float = math.radians(62.2)
    sign: float = -1.0
    gain: float = 1.0
    min_pixels: float = 0.5
    max_pixels: float = 200.0


class YawFromFlow:
    """Stateful frame-to-frame yaw delta estimator.

    Feed each new grayscale frame to :meth:`update`; the helper returns
    the yaw delta in radians since the previous frame, or ``None`` if no
    measurement is available (first frame, shape change, outlier).

    Example::

        est = YawFromFlow()
        for frame in camera_stream:
            delta = est.update(frame)
            if delta is not None:
                yaw_accumulated += delta
    """

    def __init__(self, config: YawFromFlowConfig | None = None) -> None:
        self._cfg = config or YawFromFlowConfig()
        self._prev = None  # numpy.ndarray, float32
        self._image_width: int = 0

    def reset(self) -> None:
        """Forget the previous frame (e.g. between executor actions)."""
        self._prev = None
        self._image_width = 0

    def update(
        self,
        gray,
        phase_correlate: Optional[Callable] = None,
    ) -> Optional[float]:
        """Process one frame; return the yaw delta in radians or ``None``.

        ``gray`` is a single-channel image (numpy array, any numeric
        dtype). ``phase_correlate`` is an optional override for the
        phase correlation function (used by unit tests to avoid the
        OpenCV import); when ``None``, ``cv2.phaseCorrelate`` is
        imported lazily.

        Returns the estimated yaw delta in radians, or ``None`` when no
        valid measurement could be produced from this frame.
        """
        import numpy as np  # noqa: WPS433

        if phase_correlate is None:
            import cv2  # noqa: WPS433
            phase_correlate = cv2.phaseCorrelate

        frame = np.ascontiguousarray(gray, dtype=np.float32)
        if self._prev is None or self._prev.shape != frame.shape:
            self._prev = frame
            self._image_width = int(frame.shape[1])
            return None

        (dx, _dy), _response = phase_correlate(self._prev, frame)
        self._prev = frame

        dx = float(dx)
        if abs(dx) > self._cfg.max_pixels:
            return None
        if abs(dx) < self._cfg.min_pixels:
            return 0.0

        pix_per_rad = self._image_width / self._cfg.camera_hfov_rad
        return self._cfg.sign * self._cfg.gain * dx / pix_per_rad


__all__ = ['YawFromFlow', 'YawFromFlowConfig']

"""Camera-based line alignment for the AlphaBot2 after a discrete turn.

Used both by the Pi-side ROS-free runner
(:mod:`maze_mdp.hardware.line_follow_policy`) and by the Gazebo
``gazebo_policy_runner`` node so that the same alignment logic runs in
sim and on hardware.

Pipeline (see ``docs/control.md`` section 9 for the rationale)::

    capture frame  ->  ROI crop  ->  Otsu threshold  ->  per-row centroid
                                                            |
                                                            v
                                            least-squares fit col = m*row + b
                                                            |
                       Phase A: rotate in place until |theta| <= theta_tol
                                                            |
                       Phase B: creep forward until the IR array sees the
                                line again (intersection condition false +
                                at least one inner channel on the line)

Camera geometry matches the Gazebo URDF
(`src/alphabot2_gazebo/urdf/alphabot2.urdf`): 320x240, horizontal FOV
1.0856 rad (~62.2 deg), pitched 0.7854 rad (45 deg) down.

The line extraction routine :func:`estimate_line_from_frame` is pure
numpy (no OpenCV / no picamera2) so it is safe to import from a ROS
node running on the laptop. The :class:`CameraAligner` wrapper uses
``picamera2`` when available, and otherwise falls back to OpenCV's
``VideoCapture`` backend.

Dependencies on the Pi (install once)::

    sudo apt install python3-picamera2 python3-libcamera python3-numpy

or, if picamera2/libcamera packages are unavailable on the image::

    sudo apt install python3-opencv python3-numpy
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum

import numpy as np

try:  # pragma: no cover - only available on the Pi
    from picamera2 import Picamera2  # type: ignore
    _PICAMERA2_AVAILABLE = True
    _PICAMERA2_IMPORT_ERROR: BaseException | None = None
except Exception as exc:  # pragma: no cover
    _PICAMERA2_AVAILABLE = False
    _PICAMERA2_IMPORT_ERROR = exc

try:  # pragma: no cover - optional backend on Ubuntu images
    import cv2  # type: ignore
    _OPENCV_AVAILABLE = True
    _OPENCV_IMPORT_ERROR: BaseException | None = None
except Exception as exc:  # pragma: no cover
    _OPENCV_AVAILABLE = False
    _OPENCV_IMPORT_ERROR = exc


# Camera geometry, mirroring the Gazebo URDF.
_IMG_W = 320
_IMG_H = 240
_HFOV_RAD = 1.0856

# ROI: bottom band, central strip. Tuned for a 45-deg-down camera so
# the cross-arms of the intersection have already left the frame
# downward by the time we reach the ROI top.
_ROI_TOP = 0.55      # fraction of image height
_ROI_BOTTOM = 0.95
_ROI_LEFT = 0.20
_ROI_RIGHT = 0.80

# Phase A (rotate-in-place) tuning.
_THETA_TOL_RAD = np.deg2rad(4.0)
_THETA_STABLE_FRAMES = 3
_ROTATE_PWM = 12
_ROTATE_PULSE_S = 0.06
_ROTATE_GAP_S = 0.04
_ALIGN_TIMEOUT_S = 1.5

# Phase B (creep forward) tuning.
_CREEP_PWM = 15
_CREEP_TIMEOUT_S = 1.0
_CREEP_POLL_S = 0.02
_IR_INTERSECTION_THR = 900   # same threshold as line_follow_policy
_IR_ON_LINE_THR = 600        # inner-channel threshold to declare "line found"

# Minimum number of valid centroids in a frame to trust the line fit.
_MIN_ROWS_FOR_FIT = 8


class AlignOutcome(Enum):
    """Result of one full ``align_and_creep`` call."""

    OK = 'ok'
    ALIGN_TIMEOUT = 'align_timeout'
    CREEP_TIMEOUT = 'creep_timeout'
    NO_CAMERA = 'no_camera'


@dataclass
class FrameEstimate:
    """Output of :func:`estimate_line_from_frame`."""

    valid: bool
    e_theta: float = 0.0    # rad, +ve = line tilts to the right at top
    e_x: float = 0.0        # px, +ve = line is to the right of image center
    n_rows: int = 0


# ---------------------------------------------------------------- line extraction
def _otsu_threshold(gray: np.ndarray) -> int:
    """Pure-numpy Otsu threshold on an 8-bit grayscale image."""
    hist, _ = np.histogram(gray, bins=256, range=(0, 256))
    hist = hist.astype(np.float64)
    total = hist.sum()
    if total <= 0:
        return 127
    omega = np.cumsum(hist)
    mu = np.cumsum(hist * np.arange(256))
    mu_t = mu[-1]
    denom = omega * (total - omega)
    denom[denom == 0] = 1.0
    sigma_b2 = (mu_t * omega - mu * total) ** 2 / denom
    return int(np.argmax(sigma_b2))


def estimate_line_from_frame(
    frame: np.ndarray,
    *,
    image_center_col: float | None = None,
    theta_offset: float = 0.0,
) -> FrameEstimate:
    """Fit a line to the black-on-white track in the bottom-center ROI."""
    if frame.ndim == 3:
        gray = (0.299 * frame[..., 0]
                + 0.587 * frame[..., 1]
                + 0.114 * frame[..., 2]).astype(np.uint8)
    else:
        gray = frame.astype(np.uint8)

    h, w = gray.shape
    r0 = int(_ROI_TOP * h)
    r1 = int(_ROI_BOTTOM * h)
    c0 = int(_ROI_LEFT * w)
    c1 = int(_ROI_RIGHT * w)
    roi = gray[r0:r1, c0:c1]

    thr = _otsu_threshold(roi)
    mask = roi < thr  # line is dark on light floor

    row_idx = np.arange(roi.shape[0])
    col_idx = np.arange(roi.shape[1])
    counts = mask.sum(axis=1).astype(np.float64)
    weighted = (mask * col_idx).sum(axis=1).astype(np.float64)
    valid_rows = counts >= max(3, 0.05 * roi.shape[1])
    n_valid = int(valid_rows.sum())
    if n_valid < _MIN_ROWS_FOR_FIT:
        return FrameEstimate(valid=False, n_rows=n_valid)

    centroids = weighted[valid_rows] / counts[valid_rows]
    rows = row_idx[valid_rows].astype(np.float64)

    A = np.stack([rows, np.ones_like(rows)], axis=1)
    coef, _residuals, _rank, _sv = np.linalg.lstsq(A, centroids, rcond=None)
    m, b = float(coef[0]), float(coef[1])

    col_bottom_roi = m * (roi.shape[0] - 1) + b
    col_bottom_full = col_bottom_roi + c0
    if image_center_col is None:
        image_center_col = w / 2.0
    e_x = col_bottom_full - image_center_col
    e_theta = float(np.arctan(m)) - theta_offset

    return FrameEstimate(valid=True, e_theta=e_theta, e_x=e_x, n_rows=n_valid)


# ---------------------------------------------------------------- aligner (Pi-side)
class CameraAligner:
    """Encapsulate camera lifecycle + the two-phase alignment routine.

    Pi-only: requires ``picamera2``. The Gazebo node implements the same
    behaviour directly against ``sensor_msgs/Image`` and does not use
    this class.
    """

    def __init__(
        self,
        motors,
        line_sensor,
        *,
        image_center_col: float | None = None,
        theta_offset: float = 0.0,
        debug_dir: str | None = None,
    ) -> None:
        self._motors = motors
        self._tr = line_sensor
        self._image_center_col = image_center_col
        self._theta_offset = float(theta_offset)
        self._debug_dir = debug_dir
        self._cam = None
        self._backend = None
        self._frame_idx = 0

    def start(self) -> bool:
        if _PICAMERA2_AVAILABLE:
            try:
                cam = Picamera2()
                config = cam.create_video_configuration(
                    main={'size': (_IMG_W, _IMG_H), 'format': 'RGB888'})
                cam.configure(config)
                cam.start()
                time.sleep(0.3)
                self._cam = cam
                self._backend = 'picamera2'
                print('[camera_align] camera backend: picamera2')
                return True
            except Exception as exc:  # pragma: no cover
                print('[camera_align] picamera2 start failed: {}'.format(exc))
                self._cam = None
                self._backend = None
        else:
            print('[camera_align] picamera2 not available: {}'
                  .format(_PICAMERA2_IMPORT_ERROR))

        if _OPENCV_AVAILABLE:
            try:
                cam = cv2.VideoCapture(0)
                cam.set(cv2.CAP_PROP_FRAME_WIDTH, _IMG_W)
                cam.set(cv2.CAP_PROP_FRAME_HEIGHT, _IMG_H)
                if not cam.isOpened():
                    raise RuntimeError('cv2.VideoCapture(0) could not open')
                time.sleep(0.2)
                self._cam = cam
                self._backend = 'opencv'
                print('[camera_align] camera backend: opencv')
                return True
            except Exception as exc:  # pragma: no cover
                print('[camera_align] opencv start failed: {}'.format(exc))
                self._cam = None
                self._backend = None
        else:
            print('[camera_align] opencv not available: {}'
                  .format(_OPENCV_IMPORT_ERROR))

        return False

    def stop(self) -> None:
        if self._cam is not None:
            try:
                if self._backend == 'picamera2':
                    self._cam.stop()
                elif self._backend == 'opencv':
                    self._cam.release()
            except Exception:
                pass
            self._cam = None
            self._backend = None

    def grab_estimate(self) -> FrameEstimate:
        if self._cam is None:
            return FrameEstimate(valid=False)
        if self._backend == 'picamera2':
            frame = self._cam.capture_array()
        elif self._backend == 'opencv':
            ok, frame_bgr = self._cam.read()
            if not ok:
                return FrameEstimate(valid=False)
            frame = frame_bgr[..., ::-1]
        else:
            return FrameEstimate(valid=False)
        est = estimate_line_from_frame(
            frame,
            image_center_col=self._image_center_col,
            theta_offset=self._theta_offset,
        )
        if self._debug_dir is not None:
            self._save_debug(frame, est)
        return est

    def _save_debug(self, frame: np.ndarray, est: FrameEstimate) -> None:
        try:
            import os
            os.makedirs(self._debug_dir, exist_ok=True)
            path = os.path.join(
                self._debug_dir,
                'align_{:04d}.npy'.format(self._frame_idx))
            np.save(path, frame)
            self._frame_idx += 1
            print('[camera_align] frame {} valid={} e_theta={:.3f} e_x={:.1f}'
                  .format(self._frame_idx, est.valid, est.e_theta, est.e_x))
        except Exception as exc:  # pragma: no cover
            print('[camera_align] debug save failed: {}'.format(exc))

    def align_and_creep(self) -> AlignOutcome:
        if self._cam is None:
            self._open_loop_forward()
            return AlignOutcome.NO_CAMERA

        align_ok = self._phase_rotate()
        creep_ok = self._phase_creep()

        if align_ok and creep_ok:
            return AlignOutcome.OK
        if not align_ok and creep_ok:
            return AlignOutcome.ALIGN_TIMEOUT
        return AlignOutcome.CREEP_TIMEOUT

    def _phase_rotate(self) -> bool:
        deadline = time.monotonic() + _ALIGN_TIMEOUT_S
        stable = 0
        while time.monotonic() < deadline:
            est = self.grab_estimate()
            if not est.valid:
                self._motors.setPWMA(_CREEP_PWM)
                self._motors.setPWMB(_CREEP_PWM)
                self._motors.forward()
                time.sleep(_CREEP_POLL_S * 3)
                self._motors.stop()
                continue
            if abs(est.e_theta) <= _THETA_TOL_RAD:
                stable += 1
                if stable >= _THETA_STABLE_FRAMES:
                    return True
                time.sleep(_ROTATE_GAP_S)
                continue
            stable = 0
            self._rotate_pulse(est.e_theta)
        return False

    def _rotate_pulse(self, e_theta: float) -> None:
        self._motors.setPWMA(_ROTATE_PWM)
        self._motors.setPWMB(_ROTATE_PWM)
        if e_theta > 0:
            self._motors.right()
        else:
            self._motors.left()
        time.sleep(_ROTATE_PULSE_S)
        self._motors.stop()
        time.sleep(_ROTATE_GAP_S)

    def _phase_creep(self) -> bool:
        self._motors.setPWMA(_CREEP_PWM)
        self._motors.setPWMB(_CREEP_PWM)
        self._motors.forward()
        deadline = time.monotonic() + _CREEP_TIMEOUT_S
        try:
            while time.monotonic() < deadline:
                _position, sensors = self._tr.readLine()
                all_on = all(s > _IR_INTERSECTION_THR for s in sensors)
                inner_on = (sensors[1] > _IR_ON_LINE_THR
                            or sensors[2] > _IR_ON_LINE_THR
                            or sensors[3] > _IR_ON_LINE_THR)
                if (not all_on) and inner_on:
                    return True
                time.sleep(_CREEP_POLL_S)
            return False
        finally:
            self._motors.stop()

    def _open_loop_forward(self) -> None:
        self._motors.setPWMA(_CREEP_PWM)
        self._motors.setPWMB(_CREEP_PWM)
        self._motors.forward()
        time.sleep(0.5)
        self._motors.stop()


__all__ = [
    'AlignOutcome',
    'CameraAligner',
    'FrameEstimate',
    'estimate_line_from_frame',
]

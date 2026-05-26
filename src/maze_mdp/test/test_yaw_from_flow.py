"""Unit tests for the ROS-free monocular yaw-from-flow estimator."""

import math

import numpy as np

from maze_mdp.perception.yaw_from_flow import YawFromFlow, YawFromFlowConfig


def _fake_phase_correlate(shift_x: float):
    """Build a stand-in for ``cv2.phaseCorrelate`` returning ``(dx, dy)``."""

    def _impl(prev, curr):
        # Signature matches cv2.phaseCorrelate(src1, src2) -> ((dx, dy), resp)
        return (float(shift_x), 0.0), 1.0

    return _impl


def test_first_frame_returns_none():
    est = YawFromFlow()
    frame = np.zeros((10, 20), dtype=np.float32)
    assert est.update(frame, _fake_phase_correlate(0.0)) is None


def test_small_shift_clamped_to_zero():
    cfg = YawFromFlowConfig(min_pixels=0.5, max_pixels=100.0)
    est = YawFromFlow(cfg)
    frame = np.zeros((10, 20), dtype=np.float32)
    est.update(frame, _fake_phase_correlate(0.0))
    out = est.update(frame, _fake_phase_correlate(0.1))
    assert out == 0.0


def test_outlier_shift_rejected():
    cfg = YawFromFlowConfig(min_pixels=0.5, max_pixels=10.0)
    est = YawFromFlow(cfg)
    frame = np.zeros((10, 20), dtype=np.float32)
    est.update(frame, _fake_phase_correlate(0.0))
    assert est.update(frame, _fake_phase_correlate(50.0)) is None


def test_pixel_shift_converts_to_radians():
    # 90 deg HFOV, 90 px wide -> 1 px = 1 deg = pi/180 rad.
    cfg = YawFromFlowConfig(
        camera_hfov_rad=math.radians(90.0), sign=-1.0,
        min_pixels=0.1, max_pixels=1e3)
    est = YawFromFlow(cfg)
    frame = np.zeros((10, 90), dtype=np.float32)
    est.update(frame, _fake_phase_correlate(0.0))
    out = est.update(frame, _fake_phase_correlate(+9.0))
    # 9 px shift -> 9 deg. Sign=-1 -> negative (CW under REP-103).
    assert out is not None
    assert math.isclose(out, -math.radians(9.0), rel_tol=1e-9)


def test_gain_scales_converted_yaw():
    """`gain` multiplies the output, e.g. to compensate camera pitch."""
    base = YawFromFlowConfig(
        camera_hfov_rad=math.radians(90.0), sign=1.0,
        gain=1.0, min_pixels=0.1, max_pixels=1e3)
    scaled = YawFromFlowConfig(
        camera_hfov_rad=math.radians(90.0), sign=1.0,
        gain=2.5, min_pixels=0.1, max_pixels=1e3)
    e_base = YawFromFlow(base)
    e_scaled = YawFromFlow(scaled)
    frame = np.zeros((10, 90), dtype=np.float32)
    e_base.update(frame, _fake_phase_correlate(0.0))
    e_scaled.update(frame, _fake_phase_correlate(0.0))
    out_base = e_base.update(frame, _fake_phase_correlate(+9.0))
    out_scaled = e_scaled.update(frame, _fake_phase_correlate(+9.0))
    assert out_base is not None and out_scaled is not None
    assert math.isclose(out_scaled, 2.5 * out_base, rel_tol=1e-9)


def test_shape_change_resets_estimator():
    est = YawFromFlow()
    a = np.zeros((10, 20), dtype=np.float32)
    b = np.zeros((12, 24), dtype=np.float32)
    assert est.update(a, _fake_phase_correlate(1.0)) is None
    # New shape -> no measurement on this frame either.
    assert est.update(b, _fake_phase_correlate(1.0)) is None


def test_reset_drops_previous_frame():
    cfg = YawFromFlowConfig(min_pixels=0.1, max_pixels=1e3)
    est = YawFromFlow(cfg)
    frame = np.zeros((10, 20), dtype=np.float32)
    est.update(frame, _fake_phase_correlate(0.0))
    est.reset()
    assert est.update(frame, _fake_phase_correlate(5.0)) is None

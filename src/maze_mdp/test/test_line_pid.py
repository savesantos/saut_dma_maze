"""Unit tests for the ROS-free line-follow PID controller."""

import math

import pytest

from maze_mdp.control.line_pid import LinePID, LinePIDConfig


def test_proportional_only_matches_legacy_behaviour():
    pid = LinePID(LinePIDConfig(kp=2.0, ki=0.0, kd=0.0,
                                d_filter_tau=0.0, output_clamp=0.0))
    # Sign convention: u = -Kp * pose, matching the previous P-only
    # ``ang = -line_p_gain * pose`` formula used by the executor.
    assert pid.step(+0.5, dt=0.05) == pytest.approx(-1.0)
    pid.reset()
    assert pid.step(-0.25, dt=0.05) == pytest.approx(+0.5)


def test_zero_pose_gives_zero_output():
    pid = LinePID(LinePIDConfig(kp=2.0, kd=0.3, ki=0.5))
    for _ in range(20):
        u = pid.step(0.0, dt=0.05)
    assert u == pytest.approx(0.0)


def test_first_sample_has_no_derivative_kick():
    """A pure-Kd controller must return 0 on its first sample (no prev)."""
    pid = LinePID(LinePIDConfig(kp=0.0, kd=10.0, d_filter_tau=0.0))
    assert pid.step(+0.8, dt=0.05) == pytest.approx(0.0)


def test_derivative_responds_to_change():
    """With Kd>0 and constant pose after a step, D term decays toward 0."""
    pid = LinePID(LinePIDConfig(kp=0.0, kd=1.0, d_filter_tau=0.0,
                                output_clamp=0.0))
    pid.step(0.0, dt=0.05)
    # error jumps from 0 to -0.5 over 0.05 s -> raw_d = -10, u = -10.
    u1 = pid.step(+0.5, dt=0.05)
    assert u1 == pytest.approx(-10.0)
    # Holding the same pose -> derivative collapses to 0.
    u2 = pid.step(+0.5, dt=0.05)
    assert u2 == pytest.approx(0.0)


def test_derivative_low_pass_filter_attenuates_step():
    """``d_filter_tau`` should reduce the instantaneous derivative spike."""
    pid_raw = LinePID(LinePIDConfig(kp=0.0, kd=1.0,
                                    d_filter_tau=0.0, output_clamp=0.0))
    pid_filt = LinePID(LinePIDConfig(kp=0.0, kd=1.0,
                                     d_filter_tau=0.2, output_clamp=0.0))
    for p in (pid_raw, pid_filt):
        p.step(0.0, dt=0.05)
    u_raw = pid_raw.step(+0.5, dt=0.05)
    u_filt = pid_filt.step(+0.5, dt=0.05)
    assert abs(u_filt) < abs(u_raw)


def test_integral_accumulates_constant_error():
    pid = LinePID(LinePIDConfig(kp=0.0, ki=1.0, kd=0.0,
                                i_clamp=10.0, output_clamp=0.0))
    pid.step(+0.5, dt=0.05)  # first step seeds prev, no integration yet
    # Subsequent steps accumulate.
    u = 0.0
    for _ in range(10):
        u = pid.step(+0.5, dt=0.05)
    # integral = 10 * 0.05 * (-0.5) = -0.25, u = ki * integral = -0.25.
    assert u == pytest.approx(-0.25, abs=1e-6)


def test_integral_clamp_bounds_state():
    pid = LinePID(LinePIDConfig(kp=0.0, ki=1.0, kd=0.0,
                                i_clamp=0.1, output_clamp=0.0))
    pid.step(+0.5, dt=0.05)
    for _ in range(1000):
        pid.step(+0.5, dt=0.05)
    # Final |u| = ki * i_clamp = 0.1, regardless of how long we ran.
    assert abs(pid.step(+0.5, dt=0.05)) == pytest.approx(0.1, abs=1e-6)


def test_output_saturation_clamps_command():
    pid = LinePID(LinePIDConfig(kp=10.0, ki=0.0, kd=0.0, output_clamp=1.5))
    assert pid.step(+0.8, dt=0.05) == pytest.approx(-1.5)
    assert pid.step(-0.8, dt=0.05) == pytest.approx(+1.5)


def test_anti_windup_stops_integrator_in_saturation():
    """When output is saturated, the integrator must not wind further."""
    pid = LinePID(LinePIDConfig(kp=10.0, ki=1.0, kd=0.0,
                                i_clamp=100.0, output_clamp=1.0))
    pid.step(+0.5, dt=0.05)
    for _ in range(50):
        pid.step(+0.5, dt=0.05)  # P alone already saturates the output
    # Now clear the error and see how fast the controller un-saturates.
    # If the integrator had wound up, we would see a large positive u when
    # error flips sign. With anti-windup it stays near 0.
    u = pid.step(0.0, dt=0.05)
    assert abs(u) < 0.05


def test_reset_clears_all_state():
    pid = LinePID(LinePIDConfig(kp=0.0, ki=1.0, kd=1.0,
                                d_filter_tau=0.0, output_clamp=0.0))
    for _ in range(10):
        pid.step(+0.5, dt=0.05)
    pid.reset()
    # First step after reset must look exactly like a fresh controller:
    # no D contribution, integrator zero -> u = 0.
    assert pid.step(0.0, dt=0.05) == pytest.approx(0.0)


def test_dt_independence_of_proportional_term():
    pid_a = LinePID(LinePIDConfig(kp=1.0, ki=0.0, kd=0.0,
                                  d_filter_tau=0.0, output_clamp=0.0))
    pid_b = LinePID(LinePIDConfig(kp=1.0, ki=0.0, kd=0.0,
                                  d_filter_tau=0.0, output_clamp=0.0))
    u_fast = pid_a.step(+0.5, dt=0.01)
    u_slow = pid_b.step(+0.5, dt=0.10)
    # P is purely algebraic on the current sample and must not depend on dt.
    assert u_fast == pytest.approx(u_slow)


def test_zero_dt_treated_as_p_only():
    pid = LinePID(LinePIDConfig(kp=2.0, ki=1.0, kd=1.0,
                                d_filter_tau=0.0, output_clamp=0.0))
    pid.step(+0.1, dt=0.05)
    # dt=0 must not divide-by-zero nor change the integrator.
    u = pid.step(+0.5, dt=0.0)
    assert math.isfinite(u)
    assert u == pytest.approx(-1.0)  # only P (Kp * -0.5)

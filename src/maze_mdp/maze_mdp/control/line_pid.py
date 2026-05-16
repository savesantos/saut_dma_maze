"""
ROS-free PID controller for IR-based line following.

The setpoint is ``pose = 0`` (line centred under the IR strip). The plant
output ``pose`` is in ``[-1, +1]`` with the same sign convention as
``estimate_line_pose`` (positive = line is to the right of the robot). The
controller emits an angular-velocity command ``omega = -PID(pose, dt)`` so
that the wrapper can do::

    cmd = MotorCmd(linear=forward_speed, angular=pid.step(pose, dt))

Best-practice features required for both the Gazebo simulator and the
AlphaBot2's noisy 5-channel IR strip:

* **dt-aware** so the same gains behave identically at 20 Hz (sim) and at
  the hardware loop rate (~50 Hz with the TRSensors driver).
* **Derivative on measurement, low-pass filtered** with a first-order
  filter (time constant ``d_filter_tau``). The IR strip is noisy and a raw
  finite difference would amplify it into a screaming motor command. The
  filter rolls off above ``f_c = 1 / (2*pi*tau)``.
* **Anti-windup by conditional integration**: the integral only accumulates
  when the (clamped) output is not saturated *or* the error would pull it
  out of saturation. Plus a hard ``i_clamp`` as a defensive bound.
* **Output saturation** (``output_clamp``) to keep the angular command
  inside what the diff-drive (sim) and motor driver (hardware) can track.
* **Stateless first sample**: D and I contribute nothing on the first
  ``step()`` after :py:meth:`reset`, so an action that starts already off
  the line doesn't fire a derivative spike.

The class is deliberately ROS-free so it can be unit-tested with plain
``pytest`` and reused by both the sim executor and a future hardware
node.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LinePIDConfig:
    """Static tuning for :class:`LinePID`.

    All fields are SI (rad/s for angular outputs, s for time constants).
    ``kp``/``ki``/``kd`` apply to the error signal ``e = setpoint - pose``
    with ``setpoint = 0``, so positive ``pose`` (line on the right) yields
    ``e < 0`` and thus a *negative* PID output before sign inversion at the
    call site, which the executor maps to a clockwise (right) yaw.

    Defaults reproduce the legacy P-only behaviour (``Kp=0.8``,
    ``Ki=Kd=0``) so existing tests pass unchanged.
    """

    kp: float = 0.8
    ki: float = 0.0
    kd: float = 0.0
    # Low-pass time constant on the derivative term. ``0.0`` disables the
    # filter (raw finite difference). For noisy IR strips a value of
    # roughly one to two control periods works well, e.g. 0.05 s at 20 Hz.
    d_filter_tau: float = 0.05
    # Hard symmetric clamp on the accumulated integral state, in the same
    # units as ``e * t`` (dimensionless * s). ``0.0`` disables the clamp.
    i_clamp: float = 0.5
    # Hard symmetric clamp on the output ``u`` (rad/s). ``0.0`` disables
    # the clamp. When clamped, the integrator stops winding into the
    # saturation side.
    output_clamp: float = 2.5
    # If the elapsed ``dt`` between successive ``step`` calls is below
    # this floor, the controller treats it as P-only for that sample
    # (avoids division-by-zero and derivative blow-up on duplicate
    # samples from a chatty sensor).
    min_dt: float = 1.0e-3


class LinePID:
    """Discrete-time PID with derivative low-pass and anti-windup.

    Usage::

        pid = LinePID(LinePIDConfig(kp=2.0, kd=0.3, d_filter_tau=0.05))
        # ... every time a new line_pose sample arrives ...
        omega = pid.step(pose, dt)

    Call :py:meth:`reset` whenever the line follower is (re)engaged so the
    derivative filter and integrator don't carry state from a previous
    action.
    """

    def __init__(self, cfg: LinePIDConfig | None = None) -> None:
        self._cfg = cfg or LinePIDConfig()
        self._integral: float = 0.0
        self._prev_error: float = 0.0
        self._d_state: float = 0.0
        self._has_prev: bool = False

    # ------------------------------------------------------------- API
    @property
    def config(self) -> LinePIDConfig:
        return self._cfg

    def reset(self) -> None:
        """Clear the integrator, last error, and derivative filter state."""
        self._integral = 0.0
        self._prev_error = 0.0
        self._d_state = 0.0
        self._has_prev = False

    def step(self, pose: float, dt: float) -> float:
        """Advance the controller by ``dt`` seconds with measurement ``pose``.

        Returns the angular-velocity command (rad/s), saturated to
        ``+/- output_clamp`` if configured. ``pose`` outside ``[-1, +1]`` is
        accepted (the executor clamps upstream).
        """
        cfg = self._cfg
        error = -float(pose)  # setpoint = 0

        # P
        p_term = cfg.kp * error

        # D on error, low-pass filtered.
        if self._has_prev and dt > cfg.min_dt:
            raw_d = (error - self._prev_error) / dt
        else:
            raw_d = 0.0
        if cfg.d_filter_tau > 0.0 and dt > cfg.min_dt:
            alpha = dt / (cfg.d_filter_tau + dt)
            self._d_state += alpha * (raw_d - self._d_state)
        else:
            self._d_state = raw_d
        d_term = cfg.kd * self._d_state

        # Tentative integral update (committed only if not winding into sat).
        if self._has_prev and dt > cfg.min_dt:
            new_integral = self._integral + error * dt
        else:
            new_integral = self._integral
        if cfg.i_clamp > 0.0:
            if new_integral > cfg.i_clamp:
                new_integral = cfg.i_clamp
            elif new_integral < -cfg.i_clamp:
                new_integral = -cfg.i_clamp
        i_term = cfg.ki * new_integral

        u_unsat = p_term + i_term + d_term

        # Saturate and anti-wind.
        if cfg.output_clamp > 0.0:
            if u_unsat > cfg.output_clamp:
                u = cfg.output_clamp
                # Only commit the integrator update when the error wants
                # to pull the output *out* of the positive saturation
                # (error <= 0 reduces u_unsat next step).
                if error <= 0.0:
                    self._integral = new_integral
            elif u_unsat < -cfg.output_clamp:
                u = -cfg.output_clamp
                if error >= 0.0:
                    self._integral = new_integral
            else:
                u = u_unsat
                self._integral = new_integral
        else:
            u = u_unsat
            self._integral = new_integral

        self._prev_error = error
        self._has_prev = True
        return u


__all__ = ['LinePID', 'LinePIDConfig']

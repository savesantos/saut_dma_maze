"""
Value Iteration for the tabular maze MDP (ROS-free).

Vectorized Bellman backup over the dense ``P[S, A, S']`` tensor. Returns the
optimal value function, a deterministic greedy policy, and convergence info
suitable for plotting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter

import numpy as np

from maze_mdp.mdp import MDP


@dataclass
class VIInfo:
    """Convergence diagnostics returned by :func:`value_iteration`."""

    iterations: int = 0
    converged: bool = False
    wall_clock_s: float = 0.0
    bellman_error_history: list[float] = field(default_factory=list)


def value_iteration(
    mdp: MDP,
    gamma: float | None = None,
    tol: float = 1e-6,
    max_iterations: int = 10_000,
) -> tuple[np.ndarray, np.ndarray, VIInfo]:
    """
    Run value iteration until the sup-norm Bellman error falls below ``tol``.

    Parameters
    ----------
    mdp:
        The :class:`MDP` to solve.
    gamma:
        Override the MDP's discount factor (defaults to ``mdp.gamma``).
    tol:
        Convergence threshold on ``max_s |V_{k+1}(s) - V_k(s)|``.
    max_iterations:
        Hard cap on Bellman sweeps; raises ``RuntimeError`` if hit.

    Returns
    -------
    V: ``np.ndarray`` of shape ``(n_states,)``.
    pi: ``np.ndarray[int]`` of shape ``(n_states,)``, the greedy policy.
    info: :class:`VIInfo` with iteration count, convergence flag and history.

    """
    if gamma is None:
        gamma = mdp.gamma
    if not 0.0 < gamma < 1.0:
        raise ValueError('gamma must be in (0, 1)')

    P = mdp.P  # (S, A, S')
    Rbar = mdp.expected_reward()  # (S, A)
    V = np.zeros(mdp.n_states, dtype=np.float64)
    info = VIInfo()
    start = perf_counter()

    for it in range(1, max_iterations + 1):
        # Q(s, a) = Rbar(s, a) + gamma * sum_s' P(s'|s,a) V(s')
        Q = Rbar + gamma * P.dot(V)
        V_new = Q.max(axis=1)
        delta = float(np.max(np.abs(V_new - V)))
        info.bellman_error_history.append(delta)
        V = V_new
        if delta < tol:
            info.iterations = it
            info.converged = True
            break
    else:
        info.iterations = max_iterations
        raise RuntimeError(
            f'Value iteration did not converge within {max_iterations} sweeps '
            f'(last delta={delta:.3e}).'
        )

    pi = (Rbar + gamma * P.dot(V)).argmax(axis=1).astype(np.int64)
    info.wall_clock_s = perf_counter() - start
    return V, pi, info


__all__ = ['value_iteration', 'VIInfo']

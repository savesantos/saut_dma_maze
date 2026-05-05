"""Policy utilities: extraction, evaluation, and on-disk persistence."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from maze_mdp.mdp import MDP


def greedy_policy(Q: np.ndarray) -> np.ndarray:
    """Return the deterministic greedy policy ``argmax_a Q(s, a)``."""
    return Q.argmax(axis=1).astype(np.int64)


def policy_value(
    mdp: MDP,
    pi: np.ndarray,
    gamma: float | None = None,
    tol: float = 1e-9,
    max_iterations: int = 10_000,
) -> np.ndarray:
    """Evaluate ``pi`` under ``mdp`` by iterative policy evaluation."""
    if gamma is None:
        gamma = mdp.gamma
    Rbar = mdp.expected_reward()
    P = mdp.P
    s_idx = np.arange(mdp.n_states)
    P_pi = P[s_idx, pi]            # (S, S')
    R_pi = Rbar[s_idx, pi]         # (S,)
    V = np.zeros(mdp.n_states, dtype=np.float64)
    for _ in range(max_iterations):
        V_new = R_pi + gamma * P_pi.dot(V)
        if np.max(np.abs(V_new - V)) < tol:
            return V_new
        V = V_new
    return V


def save_policy(
    path: str | Path,
    pi: np.ndarray,
    *,
    Q: np.ndarray | None = None,
    V: np.ndarray | None = None,
    metadata: dict | None = None,
) -> None:
    """Persist a policy (and optionally Q / V) as a single ``.npz`` file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {'pi': np.asarray(pi, dtype=np.int64)}
    if Q is not None:
        arrays['Q'] = np.asarray(Q, dtype=np.float64)
    if V is not None:
        arrays['V'] = np.asarray(V, dtype=np.float64)
    if metadata:
        # Stored as a 0-d object array for numpy<1.25 compatibility.
        arrays['metadata'] = np.array(metadata, dtype=object)
    np.savez(path, **arrays)


def load_policy(path: str | Path) -> dict:
    """Inverse of :func:`save_policy`. Returns a dict with present keys only."""
    data = np.load(Path(path), allow_pickle=True)
    out: dict = {'pi': data['pi']}
    for key in ('Q', 'V'):
        if key in data.files:
            out[key] = data[key]
    if 'metadata' in data.files:
        out['metadata'] = data['metadata'].item()
    return out


__all__ = ['greedy_policy', 'policy_value', 'save_policy', 'load_policy']

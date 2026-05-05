"""SARSA (on-policy TD control) for the tabular maze MDP."""

from __future__ import annotations

import numpy as np

from maze_mdp._td import TDConfig, TDInfo, td_control
from maze_mdp.simulator import GridMaze


def sarsa(
    env: GridMaze,
    cfg: TDConfig | None = None,
    seed: int | None = None,
) -> tuple[np.ndarray, TDInfo]:
    """
    Train an on-policy SARSA agent.

    The TD target is ``r + gamma * Q(s', a')`` where ``a'`` is the action
    actually selected by the ε-greedy behavior policy.
    """
    cfg = cfg or TDConfig()

    def target(Q: np.ndarray, s_next: int, a_next: int) -> float:
        return float(Q[s_next, a_next])

    return td_control(env, cfg, target, seed=seed)


__all__ = ['sarsa', 'TDConfig', 'TDInfo']

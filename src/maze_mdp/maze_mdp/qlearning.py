"""Q-Learning (off-policy TD control) for the tabular maze MDP."""

from __future__ import annotations

import numpy as np

from maze_mdp._td import TDConfig, TDInfo, td_control
from maze_mdp.simulator import GridMaze


def q_learning(
    env: GridMaze,
    cfg: TDConfig | None = None,
    seed: int | None = None,
) -> tuple[np.ndarray, TDInfo]:
    """
    Train an off-policy Q-Learning agent.

    The TD target is ``r + gamma * max_a' Q(s', a')``, independent of the
    ε-greedy behavior policy used for action selection.
    """
    cfg = cfg or TDConfig()

    def target(Q: np.ndarray, s_next: int, _a_next: int) -> float:
        return float(Q[s_next].max())

    return td_control(env, cfg, target, seed=seed)


__all__ = ['q_learning', 'TDConfig', 'TDInfo']

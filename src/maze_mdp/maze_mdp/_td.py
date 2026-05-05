"""
Shared TD(0) loop used by both SARSA and Q-Learning.

ROS-free utility module. Exposes :func:`td_control`, a generic ε-greedy /
α-decay training loop parameterised by the bootstrap target. Keeping the loop
in one place ensures SARSA vs Q-Learning differ *only* in their TD target,
so the comparative analysis in the report is apples-to-apples.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Callable

import numpy as np

from maze_mdp.simulator import GridMaze


@dataclass
class TDConfig:
    """Hyperparameters for the shared TD(0) loop (see ``docs/mdp_design.md`` §7)."""

    n_episodes: int = 2_000
    alpha0: float = 0.1
    alpha_min: float = 1e-3
    alpha_tau: float = 50.0
    epsilon0: float = 1.0
    epsilon_min: float = 0.05
    epsilon_anneal_frac: float = 0.5
    gamma: float | None = None  # falls back to env.mdp.gamma


@dataclass
class TDInfo:
    """Per-episode metrics + run-level summary returned by training."""

    episode_returns: list[float] = field(default_factory=list)
    episode_steps: list[int] = field(default_factory=list)
    episode_epsilon: list[float] = field(default_factory=list)
    episode_td_error: list[float] = field(default_factory=list)
    wall_clock_s: float = 0.0
    seed: int | None = None


# Type for the bootstrap target. Receives current Q, next state, next action
# (epsilon-greedy sample for SARSA / unused for QL). Returns the bootstrap value.
TargetFn = Callable[[np.ndarray, int, int], float]


def epsilon_schedule(cfg: TDConfig, episode: int) -> float:
    """Linear decay of ε from ``epsilon0`` to ``epsilon_min``."""
    anneal_n = max(1, int(cfg.n_episodes * cfg.epsilon_anneal_frac))
    frac = min(1.0, episode / anneal_n)
    return cfg.epsilon0 + frac * (cfg.epsilon_min - cfg.epsilon0)


def epsilon_greedy(
    Q: np.ndarray,
    s: int,
    epsilon: float,
    rng: np.random.Generator,
) -> int:
    """Sample an action ε-greedily; ties broken by ``np.argmax`` (stable)."""
    if rng.random() < epsilon:
        return int(rng.integers(Q.shape[1]))
    # Random tie-breaking among arg-max actions is fairer for early training
    # when many entries share Q=0; once Q diverges this is a no-op.
    q_row = Q[s]
    max_q = q_row.max()
    candidates = np.flatnonzero(q_row == max_q)
    if candidates.size == 1:
        return int(candidates[0])
    return int(rng.choice(candidates))


def td_control(
    env: GridMaze,
    cfg: TDConfig,
    target_fn: TargetFn,
    seed: int | None = None,
) -> tuple[np.ndarray, TDInfo]:
    """
    Run an ε-greedy TD(0) control loop, parameterised by ``target_fn``.

    ``target_fn(Q, s_next, a_next) -> bootstrap`` is the only thing that
    differs between SARSA (uses ``Q[s_next, a_next]``) and Q-Learning
    (uses ``Q[s_next].max()``).
    """
    if seed is not None:
        env.seed(seed)
    rng = env.rng  # reuse env RNG for action sampling -> single seed per run
    gamma = cfg.gamma if cfg.gamma is not None else env.mdp.gamma
    Q = np.zeros((env.n_states, env.n_actions), dtype=np.float64)
    visit_counts = np.zeros_like(Q, dtype=np.int64)
    info = TDInfo(seed=seed)
    start = perf_counter()

    for ep in range(cfg.n_episodes):
        eps = epsilon_schedule(cfg, ep)
        s = env.reset()
        a = epsilon_greedy(Q, s, eps, rng)
        ep_return = 0.0
        ep_steps = 0
        ep_td_sum = 0.0

        done = False
        while not done:
            s_next, r, done, _info = env.step(a)
            a_next = epsilon_greedy(Q, s_next, eps, rng) if not done else 0
            bootstrap = 0.0 if done else target_fn(Q, s_next, a_next)
            target = r + gamma * bootstrap
            td_error = target - Q[s, a]

            visit_counts[s, a] += 1
            alpha = max(cfg.alpha_min, cfg.alpha0 / (1.0 + visit_counts[s, a] / cfg.alpha_tau))
            Q[s, a] += alpha * td_error

            ep_return += r
            ep_steps += 1
            ep_td_sum += abs(td_error)
            s, a = s_next, a_next

        info.episode_returns.append(ep_return)
        info.episode_steps.append(ep_steps)
        info.episode_epsilon.append(eps)
        info.episode_td_error.append(ep_td_sum / max(1, ep_steps))

    info.wall_clock_s = perf_counter() - start
    return Q, info

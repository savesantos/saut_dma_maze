"""
Markov Decision Process for the maze, with vectorized P and R tensors.

Implements the discrete MDP specified in `docs/mdp_design.md`:

- State: (row, col, heading) flat-encoded as ``s = (r * C + c) * 4 + theta``.
- Actions: ``FORWARD``, ``TURN_LEFT``, ``TURN_RIGHT``.
- Transitions: ``forward`` slips sideways with probability ``slip_prob`` per side;
  walls and grid boundaries cause the agent to stay in place. Goal is absorbing.
- Rewards: forward step cost, turn cost, optional bump penalty, terminal goal reward.

This module is intentionally ROS-free so it can be unit-tested with plain pytest.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from maze_mdp.simulator import Maze


class Heading(IntEnum):
    """Cardinal heading enum. Order is meaningful: right turn = +1 mod 4."""

    N = 0
    E = 1
    S = 2
    W = 3


class Action(IntEnum):
    """Discrete action enum mapped 1-to-1 onto AlphaBot2 motion primitives."""

    FORWARD = 0
    TURN_LEFT = 1
    TURN_RIGHT = 2


# Heading -> (dr, dc) unit vector for ``forward`` displacement.
_HEADING_DELTA = np.array(
    [
        [-1, 0],  # N
        [0, 1],   # E
        [1, 0],   # S
        [0, -1],  # W
    ],
    dtype=np.int64,
)


@dataclass(frozen=True)
class MDPConfig:
    """
    Hyperparameters defining transition stochasticity and reward shape.

    Defaults match `docs/mdp_design.md`. All values are dimensionless except
    rewards (cost units of one forward step).
    """

    slip_prob: float = 0.1
    turn_fail_prob: float = 0.0
    forward_cost: float = 1.0
    turn_cost: float = 1.2
    bump_cost: float = 2.5
    goal_reward: float = 10.0
    gamma: float = 0.99

    def __post_init__(self) -> None:
        if not 0.0 <= self.slip_prob <= 0.5:
            raise ValueError('slip_prob must be in [0, 0.5]')
        if not 0.0 <= self.turn_fail_prob <= 1.0:
            raise ValueError('turn_fail_prob must be in [0, 1]')
        if not 0.0 < self.gamma < 1.0:
            raise ValueError('gamma must be in (0, 1)')


def encode_state(row: int, col: int, heading: int, n_cols: int) -> int:
    """Flatten ``(row, col, heading)`` into a scalar state index."""
    return int((row * n_cols + col) * 4 + heading)


def decode_state(s: int, n_cols: int) -> tuple[int, int, int]:
    """Inverse of :func:`encode_state`."""
    cell, heading = divmod(int(s), 4)
    row, col = divmod(cell, n_cols)
    return row, col, heading


class MDP:
    """
    Tabular maze MDP with dense ``P[S, A, S']`` and ``R[S, A, S']`` tensors.

    Parameters
    ----------
    maze:
        :class:`maze_mdp.simulator.Maze` describing the grid and goal cell.
    config:
        :class:`MDPConfig` (defaults to the values in ``docs/mdp_design.md``).

    """

    def __init__(self, maze: 'Maze', config: MDPConfig | None = None) -> None:
        self.maze = maze
        self.config = config or MDPConfig()
        self.n_states: int = int(maze.rows * maze.cols * 4)
        self.n_actions: int = len(Action)
        self.gamma: float = self.config.gamma
        self.P, self.R = self._build_tensors()
        self.terminal_states: np.ndarray = self._compute_terminal_states()

    # ------------------------------------------------------------------ utils
    def encode(self, row: int, col: int, heading: int) -> int:
        return encode_state(row, col, heading, self.maze.cols)

    def decode(self, s: int) -> tuple[int, int, int]:
        return decode_state(s, self.maze.cols)

    def is_terminal(self, s: int) -> bool:
        return bool(self.terminal_states[s])

    def expected_reward(self) -> np.ndarray:
        """Return ``Rbar[S, A] = sum_s' P(s'|s,a) R(s,a,s')`` (used by VI)."""
        return np.einsum('sat,sat->sa', self.P, self.R)

    # ----------------------------------------------------------------- build
    def _compute_terminal_states(self) -> np.ndarray:
        terminal = np.zeros(self.n_states, dtype=bool)
        gr, gc = self.maze.goal
        for h in range(4):
            terminal[self.encode(gr, gc, h)] = True
        return terminal

    def _build_tensors(self) -> tuple[np.ndarray, np.ndarray]:
        S, A = self.n_states, self.n_actions
        P = np.zeros((S, A, S), dtype=np.float64)
        R = np.zeros((S, A, S), dtype=np.float64)

        rows, cols = self.maze.rows, self.maze.cols
        walls = self.maze.walls  # bool[rows, cols]
        gr, gc = self.maze.goal
        cfg = self.config
        ps = cfg.slip_prob
        pt = cfg.turn_fail_prob

        for r in range(rows):
            for c in range(cols):
                if walls[r, c]:
                    # Wall cells are unreachable; leave their rows as a self-loop
                    # to keep P row-stochastic.
                    for h in range(4):
                        s = self.encode(r, c, h)
                        for a in range(A):
                            P[s, a, s] = 1.0
                    continue

                for h in range(4):
                    s = self.encode(r, c, h)

                    # Goal is absorbing.
                    if (r, c) == (gr, gc):
                        for a in range(A):
                            P[s, a, s] = 1.0
                        continue

                    # --- FORWARD: 0.8 intended, 0.1 slip-left, 0.1 slip-right ---
                    intended = (1.0 - 2.0 * ps, h)
                    slip_left = (ps, (h - 1) % 4)
                    slip_right = (ps, (h + 1) % 4)
                    for prob, move_h in (intended, slip_left, slip_right):
                        if prob == 0.0:
                            continue
                        nr, nc, bumped = self._apply_move(r, c, move_h)
                        ns = self.encode(nr, nc, h)  # heading unchanged
                        P[s, Action.FORWARD, ns] += prob
                        reward = cfg.goal_reward if (nr, nc) == (gr, gc) else (
                            -cfg.forward_cost - (cfg.bump_cost if bumped else 0.0)
                        )
                        # Weighted accumulation since multiple outcomes may hit ns.
                        R[s, Action.FORWARD, ns] = self._blend_reward(
                            R[s, Action.FORWARD, ns],
                            P[s, Action.FORWARD, ns] - prob,
                            reward,
                            prob,
                        )

                    # --- TURNS: rotate in place, possibly stall ---
                    for action, dh in ((Action.TURN_LEFT, -1), (Action.TURN_RIGHT, +1)):
                        ns_turn = self.encode(r, c, (h + dh) % 4)
                        ns_stay = s
                        if pt > 0.0:
                            P[s, action, ns_turn] += 1.0 - pt
                            P[s, action, ns_stay] += pt
                        else:
                            P[s, action, ns_turn] = 1.0
                        # Reward is independent of outcome cell for turns.
                        R[s, action, ns_turn] = -cfg.turn_cost
                        R[s, action, ns_stay] = -cfg.turn_cost

        # Numerical safety: rows must sum to 1.
        row_sums = P.sum(axis=2)
        if not np.allclose(row_sums, 1.0, atol=1e-10):
            raise RuntimeError('Transition tensor is not row-stochastic')
        return P, R

    @staticmethod
    def _blend_reward(
        prev_reward: float,
        prev_prob: float,
        new_reward: float,
        new_prob: float,
    ) -> float:
        """Probability-weighted average of rewards landing in the same s'."""
        total = prev_prob + new_prob
        if total <= 0.0:
            return 0.0
        return (prev_reward * prev_prob + new_reward * new_prob) / total

    def _apply_move(self, r: int, c: int, heading: int) -> tuple[int, int, bool]:
        """Return (new_r, new_c, bumped). Bumps stay in the current cell."""
        dr, dc = _HEADING_DELTA[heading]
        nr, nc = r + int(dr), c + int(dc)
        if not (0 <= nr < self.maze.rows and 0 <= nc < self.maze.cols):
            return r, c, True
        if self.maze.walls[nr, nc]:
            return r, c, True
        return nr, nc, False

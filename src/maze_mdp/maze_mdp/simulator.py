"""
Pure-Python grid-maze micro-simulator (ROS-free).

Provides:

- :class:`Maze` — immutable grid description (walls + goal).
- :class:`GridMaze` — Gym-like environment with ``reset()`` / ``step()``.
- A small library of test fixtures (3x3, 5x5 corridor, 7x7 loop) used both
  by the unit tests and by the experiments harness as default benchmarks.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from maze_mdp.mdp import MDP, MDPConfig, Action, Heading

# Sentinel cell values used in maze YAML / fixtures.
FREE = 0
WALL = 1
GOAL = 2


@dataclass(frozen=True)
class Maze:
    """Static grid description: walls layout + goal cell."""

    cells: np.ndarray  # int8[rows, cols], values in {FREE, WALL, GOAL}
    goal: tuple[int, int]

    def __post_init__(self) -> None:
        if self.cells.ndim != 2:
            raise ValueError('cells must be 2-D')
        gr, gc = self.goal
        if not (0 <= gr < self.rows and 0 <= gc < self.cols):
            raise ValueError('goal is outside the grid')
        if self.cells[gr, gc] == WALL:
            raise ValueError('goal cannot be a wall')

    @property
    def rows(self) -> int:
        return int(self.cells.shape[0])

    @property
    def cols(self) -> int:
        return int(self.cells.shape[1])

    @property
    def walls(self) -> np.ndarray:
        return self.cells == WALL

    @property
    def free_cells(self) -> list[tuple[int, int]]:
        rs, cs = np.where(self.cells != WALL)
        return list(zip(rs.tolist(), cs.tolist()))

    @classmethod
    def from_layout(cls, layout: list[str], goal: tuple[int, int]) -> 'Maze':
        """Build a :class:`Maze` from an ASCII layout (``.`` free, ``#`` wall)."""
        rows = len(layout)
        cols = len(layout[0])
        if any(len(row) != cols for row in layout):
            raise ValueError('layout rows must all have the same length')
        cells = np.zeros((rows, cols), dtype=np.int8)
        for r, line in enumerate(layout):
            for c, ch in enumerate(line):
                cells[r, c] = WALL if ch == '#' else FREE
        gr, gc = goal
        cells[gr, gc] = GOAL
        return cls(cells=cells, goal=(gr, gc))


class GridMaze:
    """
    Stochastic gridworld environment compatible with tabular RL.

    The dynamics are sampled directly from the MDP's transition tensor so the
    simulator and the planner share a single source of truth.
    """

    def __init__(
        self,
        maze: Maze,
        config: MDPConfig | None = None,
        rng: np.random.Generator | None = None,
        max_steps: int | None = None,
        start: tuple[int, int, int] | None = None,
    ) -> None:
        self.maze = maze
        self.mdp = MDP(maze, config or MDPConfig())
        self.rng = rng if rng is not None else np.random.default_rng()
        self.max_steps = max_steps if max_steps is not None else 10 * self.mdp.n_states
        self._start = start
        self._state: int | None = None
        self._step_count: int = 0

    # ----------------------------------------------------------- public API
    @property
    def n_states(self) -> int:
        return self.mdp.n_states

    @property
    def n_actions(self) -> int:
        return self.mdp.n_actions

    def seed(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)

    def reset(self) -> int:
        """Return the initial state. Uniform over free, non-goal cells if no start fixed."""
        if self._start is not None:
            r, c, h = self._start
            self._state = self.mdp.encode(r, c, h)
        else:
            free = [cell for cell in self.maze.free_cells if cell != self.maze.goal]
            idx = int(self.rng.integers(len(free)))
            r, c = free[idx]
            h = int(self.rng.integers(4))
            self._state = self.mdp.encode(r, c, h)
        self._step_count = 0
        return self._state

    def step(self, action: int) -> tuple[int, float, bool, dict]:
        if self._state is None:
            raise RuntimeError('Call reset() before step().')
        s = self._state
        probs = self.mdp.P[s, action]
        s_next = int(self.rng.choice(self.n_states, p=probs))
        reward = float(self.mdp.R[s, action, s_next])
        self._state = s_next
        self._step_count += 1
        terminated = self.mdp.is_terminal(s_next)
        truncated = self._step_count >= self.max_steps
        done = terminated or truncated
        info = {'terminated': terminated, 'truncated': truncated, 'steps': self._step_count}
        return s_next, reward, done, info

    @property
    def state(self) -> int:
        if self._state is None:
            raise RuntimeError('Environment has not been reset.')
        return self._state


# --------------------------------------------------------------------- fixtures
def fixture_3x3() -> Maze:
    """Tiny open 3x3 maze, goal at the bottom-right corner."""
    return Maze.from_layout(
        layout=[
            '...',
            '...',
            '...',
        ],
        goal=(2, 2),
    )


def fixture_5x5_corridor() -> Maze:
    """5x5 maze with a single wall corridor; canonical SARSA/QL benchmark."""
    return Maze.from_layout(
        layout=[
            '.....',
            '.###.',
            '.....',
            '.###.',
            '.....',
        ],
        goal=(4, 4),
    )


def fixture_7x7_loop() -> Maze:
    """7x7 maze containing a loop, exercises non-tree shortest-path policies."""
    # The inner block has a single opening at (3, 1) so every free cell is
    # reachable from every other one (no trap regions). The agent must choose
    # between two homotopy-distinct paths around the central island.
    return Maze.from_layout(
        layout=[
            '.......',
            '.#####.',
            '.#...#.',
            '...#.#.',
            '.#...#.',
            '.#####.',
            '.......',
        ],
        goal=(6, 6),
    )


def fixture_7x7_rooms() -> Maze:
    """7x7 maze mirroring the lab cardboard maze; goal at bottom-left."""
    # Mixed walls / open corridors, three homotopy-distinct paths from the
    # top-left region to the goal. Connected: every free cell reaches (6, 0).
    return Maze.from_layout(
        layout=[
            '.....#.',
            '##.#...',
            '...#.#.',
            '#.##.#.',
            '.....#.',
            '####...',
            '.....#.',
        ],
        goal=(6, 0),
    )


FIXTURES = {
    'fixture_3x3': fixture_3x3,
    'fixture_5x5_corridor': fixture_5x5_corridor,
    'fixture_7x7_loop': fixture_7x7_loop,
    'fixture_7x7_rooms': fixture_7x7_rooms,
}


__all__ = [
    'Maze',
    'GridMaze',
    'Action',
    'Heading',
    'FREE',
    'WALL',
    'GOAL',
    'FIXTURES',
    'fixture_3x3',
    'fixture_5x5_corridor',
    'fixture_7x7_loop',
    'fixture_7x7_rooms',
]

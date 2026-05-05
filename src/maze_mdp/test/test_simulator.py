"""Unit tests for the GridMaze micro-simulator."""

import numpy as np
import pytest

from maze_mdp.mdp import Action, Heading, MDPConfig
from maze_mdp.simulator import GridMaze, fixture_3x3


def test_reset_returns_valid_state():
    env = GridMaze(fixture_3x3(), rng=np.random.default_rng(0))
    s = env.reset()
    assert 0 <= s < env.n_states


def test_seed_reproducibility():
    env1 = GridMaze(fixture_3x3())
    env1.seed(42)
    env1.reset()
    traj1 = [env1.step(Action.FORWARD)[0] for _ in range(20)]

    env2 = GridMaze(fixture_3x3())
    env2.seed(42)
    env2.reset()
    traj2 = [env2.step(Action.FORWARD)[0] for _ in range(20)]

    assert traj1 == traj2


def test_goal_terminates():
    cfg = MDPConfig(slip_prob=0.0, turn_fail_prob=0.0)
    env = GridMaze(
        fixture_3x3(),
        config=cfg,
        rng=np.random.default_rng(0),
        start=(2, 1, Heading.E),
    )
    env.reset()
    _, _, done, info = env.step(Action.FORWARD)
    assert done
    assert info['terminated']


def test_step_before_reset_raises():
    env = GridMaze(fixture_3x3())
    with pytest.raises(RuntimeError):
        env.step(Action.FORWARD)

"""Unit tests for value iteration on small fixtures."""

import numpy as np

from maze_mdp.mdp import MDP, MDPConfig
from maze_mdp.simulator import fixture_3x3, fixture_5x5_corridor
from maze_mdp.value_iteration import value_iteration


def test_vi_converges_on_3x3():
    mdp = MDP(fixture_3x3(), MDPConfig(slip_prob=0.0, turn_fail_prob=0.0))
    V, pi, info = value_iteration(mdp, tol=1e-8)
    assert info.converged
    # Goal cells are absorbing with zero self-loop reward, so V == 0 there.
    gr, gc = mdp.maze.goal
    for h in range(4):
        assert V[mdp.encode(gr, gc, h)] == 0.0
    # Non-goal states have V bounded by goal_reward (the maximum one-step
    # payoff reachable in the MDP).
    others = V[~mdp.terminal_states]
    assert np.all(others <= mdp.config.goal_reward + 1e-9)
    # Far-from-goal states must be strictly worse than the immediate-goal
    # reward, since reaching the goal costs at least one step.
    assert (others < mdp.config.goal_reward).any()


def test_vi_bellman_error_monotone_nonincreasing():
    mdp = MDP(fixture_5x5_corridor())
    _, _, info = value_iteration(mdp, tol=1e-8)
    history = np.asarray(info.bellman_error_history)
    assert np.all(np.diff(history) <= 1e-12)


def test_vi_policy_stable_under_tighter_tolerance():
    mdp = MDP(fixture_3x3())
    _, pi_loose, _ = value_iteration(mdp, tol=1e-3)
    _, pi_tight, _ = value_iteration(mdp, tol=1e-9)
    # Tightening tolerance must not flip an action argmax in the optimal policy.
    assert np.array_equal(pi_loose, pi_tight)

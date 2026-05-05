"""Unit tests for the MDP transition / reward tensors."""

import numpy as np
import pytest

from maze_mdp.mdp import MDP, MDPConfig, Action, Heading, decode_state, encode_state
from maze_mdp.simulator import fixture_3x3, fixture_5x5_corridor


def test_state_encoding_roundtrip():
    n_cols = 5
    for r in range(4):
        for c in range(n_cols):
            for h in range(4):
                s = encode_state(r, c, h, n_cols)
                assert decode_state(s, n_cols) == (r, c, h)


def test_p_is_row_stochastic():
    mdp = MDP(fixture_5x5_corridor())
    sums = mdp.P.sum(axis=2)
    assert np.allclose(sums, 1.0, atol=1e-12)


def test_goal_is_absorbing():
    mdp = MDP(fixture_3x3())
    gr, gc = mdp.maze.goal
    for h in range(4):
        s = mdp.encode(gr, gc, h)
        assert mdp.is_terminal(s)
        for a in range(mdp.n_actions):
            assert mdp.P[s, a, s] == pytest.approx(1.0)


def test_deterministic_when_slip_zero():
    cfg = MDPConfig(slip_prob=0.0, turn_fail_prob=0.0)
    mdp = MDP(fixture_3x3(), cfg)
    # Forward from (0, 0, E) should land at (0, 1, E) deterministically.
    s = mdp.encode(0, 0, Heading.E)
    s_next = mdp.encode(0, 1, Heading.E)
    assert mdp.P[s, Action.FORWARD, s_next] == pytest.approx(1.0)


def test_forward_stays_when_blocked():
    cfg = MDPConfig(slip_prob=0.0)
    mdp = MDP(fixture_3x3(), cfg)
    # Facing N from top row -> bumping into boundary, must self-loop.
    s = mdp.encode(0, 0, Heading.N)
    assert mdp.P[s, Action.FORWARD, s] == pytest.approx(1.0)
    # Reward should include the bump penalty.
    assert mdp.R[s, Action.FORWARD, s] < -cfg.forward_cost


def test_turn_changes_heading_only():
    cfg = MDPConfig(slip_prob=0.0, turn_fail_prob=0.0)
    mdp = MDP(fixture_3x3(), cfg)
    s = mdp.encode(1, 1, Heading.N)
    s_left = mdp.encode(1, 1, Heading.W)
    s_right = mdp.encode(1, 1, Heading.E)
    assert mdp.P[s, Action.TURN_LEFT, s_left] == pytest.approx(1.0)
    assert mdp.P[s, Action.TURN_RIGHT, s_right] == pytest.approx(1.0)


def test_slip_distribution_matches_design():
    cfg = MDPConfig(slip_prob=0.1, turn_fail_prob=0.0)
    mdp = MDP(fixture_5x5_corridor(), cfg)
    s = mdp.encode(0, 0, Heading.E)  # plenty of room east + slips north/south
    row = mdp.P[s, Action.FORWARD]
    # 0.8 to intended forward, 0.1 to each slip outcome.
    nonzero_probs = sorted(row[row > 0].tolist())
    assert np.allclose(nonzero_probs, [0.1, 0.1, 0.8], atol=1e-12)

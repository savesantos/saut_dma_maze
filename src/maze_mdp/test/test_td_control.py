"""Convergence tests for SARSA and Q-Learning against VI on small fixtures."""

import numpy as np

from maze_mdp._td import TDConfig
from maze_mdp.mdp import MDP, MDPConfig
from maze_mdp.policy import greedy_policy, policy_value
from maze_mdp.qlearning import q_learning
from maze_mdp.sarsa import sarsa
from maze_mdp.simulator import GridMaze, fixture_3x3
from maze_mdp.value_iteration import value_iteration


def _train_and_score(algo_fn, seed=0):
    cfg_mdp = MDPConfig(slip_prob=0.0, turn_fail_prob=0.0)
    maze = fixture_3x3()
    env = GridMaze(maze, config=cfg_mdp, max_steps=200)
    # Budget sized for pessimistic Q init in _td.py (Q starts at
    # -forward_cost / (1-γ) = -100, so early greedy steps are noisier and
    # need more episodes to converge to V*).
    cfg = TDConfig(n_episodes=4000, alpha0=0.3, alpha_min=1e-3, alpha_tau=20.0,
                   epsilon0=1.0, epsilon_min=0.05, epsilon_anneal_frac=0.5)
    Q, info = algo_fn(env, cfg, seed=seed)
    pi = greedy_policy(Q)
    mdp = MDP(maze, cfg_mdp)
    V_pi = policy_value(mdp, pi)
    V_star, _, _ = value_iteration(mdp, tol=1e-9)
    # Compare on reachable, non-terminal states (the rest are unreachable walls
    # / absorbing goals where both methods trivially agree).
    mask = ~mdp.terminal_states
    err = float(np.max(np.abs(V_star[mask] - V_pi[mask])))
    return err, info


def test_qlearning_recovers_optimal_policy_on_3x3():
    err, _ = _train_and_score(q_learning, seed=0)
    assert err < 0.5


def test_sarsa_recovers_near_optimal_policy_on_3x3():
    err, _ = _train_and_score(sarsa, seed=0)
    # SARSA is on-policy with eps_min=0.05, so it converges to the eps-soft
    # optimum which is slightly worse than V*. Loose bound.
    assert err < 1.5


def test_returns_improve_over_training():
    cfg_mdp = MDPConfig(slip_prob=0.0)
    env = GridMaze(fixture_3x3(), config=cfg_mdp, max_steps=200)
    cfg = TDConfig(n_episodes=800, alpha0=0.3, alpha_tau=20.0)
    _, info = q_learning(env, cfg, seed=1)
    early = np.mean(info.episode_returns[:50])
    late = np.mean(info.episode_returns[-50:])
    assert late > early

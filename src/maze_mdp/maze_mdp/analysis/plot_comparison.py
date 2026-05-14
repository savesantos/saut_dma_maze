"""Comparative optimality plot: VI vs SARSA vs Q-Learning.

For each policy we compute the **mean policy value over all reachable,
non-terminal states** using closed-form policy evaluation
(:func:`maze_mdp.policy.policy_value`). We then report the *sub-optimality
gap*

.. math:: \\Delta = \\overline{V^*} - \\overline{V^\\pi}

per maze. VI sits at :math:`\\Delta = 0` by construction; RL methods rise
above it by however much their converged policy underperforms. Because
:math:`\\gamma < 1` and rewards are bounded, :math:`V^\\pi` is *always
finite* even for policies that contain looping substates -- which is exactly
why this metric is more informative than empirical steps-to-goal.

The right panel reports an empirical success rate under the *stochastic* env
to capture robustness to slip noise (complementary axis).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from maze_mdp.analysis import style
from maze_mdp.analysis.loaders import (
    load_deployment_runs,
    load_training_runs,
    mdp_config_from_runs,
)
from maze_mdp.policy import policy_value


def _valid_state_mask(mdp) -> np.ndarray:
    """Mask of non-terminal, non-wall states (where V^pi is meaningful)."""
    walls = mdp.maze.walls
    n_cols = mdp.maze.cols
    valid = np.ones(mdp.n_states, dtype=bool)
    for s in range(mdp.n_states):
        cell, _h = divmod(s, 4)
        r, c = divmod(cell, n_cols)
        if walls[r, c]:
            valid[s] = False
    valid &= ~mdp.terminal_states
    return valid


def _build_mdp(maze_name: str, mdp_config=None):
    from maze_mdp.mdp import MDP, MDPConfig
    from maze_mdp.simulator import FIXTURES
    return MDP(FIXTURES[maze_name](), mdp_config or MDPConfig())


def _vi_mean_value(maze_name: str, mdp_config=None) -> float:
    """Mean V* averaged over reachable non-terminal states."""
    from maze_mdp.value_iteration import value_iteration
    mdp = _build_mdp(maze_name, mdp_config)
    _V, pi_star, _ = value_iteration(mdp)
    V_star = policy_value(mdp, pi_star)
    return float(V_star[_valid_state_mask(mdp)].mean())


def _policy_mean_value(maze_name: str, pi, mdp_config=None) -> float:
    mdp = _build_mdp(maze_name, mdp_config)
    V = policy_value(mdp, pi)
    return float(V[_valid_state_mask(mdp)].mean())


_EVAL_SEED = 20260512  # shared with select_best_run for a single source of truth
_EVAL_EPISODES = 1000


def _empirical_success(
    maze_name: str,
    pi,
    n_episodes: int = _EVAL_EPISODES,
    seed: int = _EVAL_SEED,
    mdp_config=None,
) -> float:
    """Empirical success rate from random starts under the stochastic env.

    Uses a fixed evaluation seed so every trained policy is scored against
    the *same* sequence of (start cell, slip noise) draws. With
    ``n_episodes`` large enough (default 1000), the residual Monte-Carlo SE
    on each per-policy estimate is :math:`\\le \\tfrac{1}{2\\sqrt{n}} \\approx 1.6\\%`,
    so the variance reported in the bar plot reflects variability across
    the *trained policies* (i.e. across training seeds) rather than eval
    noise.
    """
    from maze_mdp.mdp import MDPConfig
    from maze_mdp.simulator import FIXTURES, GridMaze
    env = GridMaze(FIXTURES[maze_name](), config=mdp_config or MDPConfig(), max_steps=500)
    env.seed(seed)
    successes = 0
    for _ in range(n_episodes):
        s = env.reset()
        while True:
            s, _r, done, info = env.step(int(pi[s]))
            if done:
                if info['terminated']:
                    successes += 1
                break
    return successes / n_episodes


def plot(
    input_dir: Path,
    output_dir: Path,
    exclude_mazes: tuple[str, ...] = (),
) -> None:
    """Render the comparative plot to ``output_dir/comparison.{png,pdf}``."""
    style.apply()
    train_df = load_training_runs(input_dir / 'training')
    deploy_df = load_deployment_runs(input_dir / 'deployment')
    if train_df.empty:
        raise SystemExit('No training runs found.')
    if exclude_mazes:
        train_df = train_df[~train_df['maze'].isin(exclude_mazes)]
        if not deploy_df.empty:
            deploy_df = deploy_df[~deploy_df['maze'].isin(exclude_mazes)]

    mazes = sorted(train_df['maze'].unique())
    mdp_cfg = mdp_config_from_runs(input_dir / 'training')
    vi_value: dict[str, float] = {m: _vi_mean_value(m, mdp_cfg) for m in mazes}

    rows = []
    for _, run in train_df.iterrows():
        pi = run['policy'].get('pi')
        if pi is None:
            continue
        v_pi = _policy_mean_value(run['maze'], pi, mdp_cfg)
        gap = vi_value[run['maze']] - v_pi          # >= 0; VI ~ 0
        success = _empirical_success(run['maze'], pi, mdp_config=mdp_cfg)
        rows.append({
            'maze': run['maze'],
            'algo': run['algo'],
            'mean_value': v_pi,
            'subopt_gap': gap,
            'success_rate': success,
        })

    import pandas as pd
    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit('Empty comparison dataframe.')

    fig, (ax_s, ax_r) = plt.subplots(1, 2, figsize=(10, 3.5))
    algos = ['vi', 'sarsa', 'qlearning']
    width = 0.25
    x = np.arange(len(mazes))

    for i, algo in enumerate(algos):
        sub = df[df['algo'] == algo]
        gap_means = [sub[sub['maze'] == m]['subopt_gap'].mean() for m in mazes]
        gap_stds = [sub[sub['maze'] == m]['subopt_gap'].std(ddof=0) for m in mazes]
        success_means = [sub[sub['maze'] == m]['success_rate'].mean() for m in mazes]
        success_stds = [sub[sub['maze'] == m]['success_rate'].std(ddof=0) for m in mazes]
        offset = (i - 1) * width
        ax_s.bar(x + offset, gap_means, yerr=gap_stds, width=width,
                 color=style.ALGO_COLORS[algo], label=style.ALGO_LABELS[algo],
                 capsize=2.5, error_kw={'linewidth': 0.7})
        ax_r.bar(x + offset, success_means, yerr=success_stds, width=width,
                 color=style.ALGO_COLORS[algo], label=style.ALGO_LABELS[algo],
                 capsize=2.5, error_kw={'linewidth': 0.7})

    ax_s.axhline(0.0, color='black', linestyle='--', linewidth=0.7, alpha=0.7)
    ax_s.set_ylabel(r'$\overline{V^*} - \overline{V^\pi}$  (0 = optimal)')
    ax_r.set_ylabel('Success rate (stochastic env)')
    for ax in (ax_s, ax_r):
        ax.set_xticks(x)
        ax.set_xticklabels(mazes, rotation=20, ha='right')
    ax_r.set_ylim(0.0, 1.05)
    ax_s.legend(loc='upper left')

    output_dir.mkdir(parents=True, exist_ok=True)
    for ext in ('png', 'pdf'):
        fig.savefig(output_dir / f'comparison.{ext}', bbox_inches='tight')
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    """CLI: ``python -m maze_mdp.analysis.plot_comparison``."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, default=Path('data'))
    parser.add_argument('--output-dir', type=Path, default=Path('data/figures'))
    parser.add_argument(
        '--exclude-mazes', nargs='*', default=['fixture_3x3'],
        help='Mazes to omit from the plot (default: fixture_3x3, which is '
             'trivially saturated and consumes space without information).',
    )
    args = parser.parse_args(argv)
    plot(args.input_dir, args.output_dir, tuple(args.exclude_mazes))


if __name__ == '__main__':
    main()


# ``deploy_df`` is loaded but unused; keep loader call to surface missing-data
# errors early and stay consistent with sister scripts.
_ = load_deployment_runs

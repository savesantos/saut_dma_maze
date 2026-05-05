"""Comparative steps-to-goal and success-rate plot, VI vs SARSA vs Q-Learning."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from maze_mdp.analysis import style
from maze_mdp.analysis.loaders import load_deployment_runs, load_training_runs


def _evaluate_policy_in_sim(
    maze_name: str,
    pi,
    n_episodes: int = 50,
    seed: int = 7,
) -> tuple[float, float]:
    from maze_mdp.mdp import MDPConfig
    from maze_mdp.simulator import FIXTURES, GridMaze
    env = GridMaze(FIXTURES[maze_name](), config=MDPConfig(), max_steps=500)
    env.seed(seed)
    steps = []
    successes = 0
    for _ in range(n_episodes):
        s = env.reset()
        n = 0
        while True:
            s, _r, done, info = env.step(int(pi[s]))
            n += 1
            if done:
                if info['terminated']:
                    successes += 1
                steps.append(n)
                break
    return float(np.mean(steps)), successes / n_episodes


def plot(input_dir: Path, output_dir: Path) -> None:
    """Render the comparative plot to ``output_dir/comparison.{png,pdf}``."""
    style.apply()
    train_df = load_training_runs(input_dir / 'training')
    deploy_df = load_deployment_runs(input_dir / 'deployment')
    if train_df.empty:
        raise SystemExit('No training runs found.')

    rows = []
    for _, run in train_df.iterrows():
        pi = run['policy'].get('pi')
        if pi is None:
            continue
        steps, success = _evaluate_policy_in_sim(run['maze'], pi)
        rows.append({
            'maze': run['maze'],
            'algo': run['algo'],
            'source': 'sim',
            'steps_to_goal': steps,
            'success_rate': success,
        })
    if not deploy_df.empty:
        for _, run in deploy_df.iterrows():
            rows.append({
                'maze': run.get('maze'),
                'algo': run.get('algo'),
                'source': 'hw',
                'steps_to_goal': run.get('steps', np.nan),
                'success_rate': float(bool(run.get('success', False))),
            })

    import pandas as pd
    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit('Empty comparison dataframe.')

    fig, (ax_s, ax_r) = plt.subplots(1, 2, figsize=(10, 3.5))
    mazes = sorted(df['maze'].unique())
    algos = ['vi', 'sarsa', 'qlearning']
    width = 0.25
    x = np.arange(len(mazes))

    for i, algo in enumerate(algos):
        sub = df[df['algo'] == algo]
        steps_means = [sub[sub['maze'] == m]['steps_to_goal'].mean() for m in mazes]
        success_means = [sub[sub['maze'] == m]['success_rate'].mean() for m in mazes]
        offset = (i - 1) * width
        ax_s.bar(x + offset, steps_means, width=width,
                 color=style.ALGO_COLORS[algo], label=style.ALGO_LABELS[algo])
        ax_r.bar(x + offset, success_means, width=width,
                 color=style.ALGO_COLORS[algo], label=style.ALGO_LABELS[algo])

    for ax, ylabel in ((ax_s, 'Mean steps-to-goal'), (ax_r, 'Success rate')):
        ax.set_xticks(x)
        ax.set_xticklabels(mazes, rotation=20, ha='right')
        ax.set_ylabel(ylabel)
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
    args = parser.parse_args(argv)
    plot(args.input_dir, args.output_dir)


if __name__ == '__main__':
    main()

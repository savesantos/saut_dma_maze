"""Plot return vs. episode (mean ± std over seeds), one subplot per maze."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from maze_mdp.analysis import style
from maze_mdp.analysis.loaders import load_training_runs


def plot(input_dir: Path, output_dir: Path) -> None:
    """Render the convergence plot to ``output_dir/convergence.{png,pdf}``."""
    style.apply()
    df = load_training_runs(input_dir / 'training')
    if df.empty:
        raise SystemExit(f'No runs found under {input_dir / "training"}')
    rl = df[df['algo'].isin(['sarsa', 'qlearning'])].copy()
    if rl.empty:
        raise SystemExit('No SARSA / Q-Learning runs found.')

    mazes = sorted(rl['maze'].unique())
    fig, axes = plt.subplots(1, len(mazes), figsize=(4.5 * len(mazes), 3.5), sharey=True)
    if len(mazes) == 1:
        axes = [axes]

    for ax, maze in zip(axes, mazes):
        for algo in ('sarsa', 'qlearning'):
            sub = rl[(rl['maze'] == maze) & (rl['algo'] == algo)]
            if sub.empty:
                continue
            stacked = np.stack([s['return'].to_numpy() for s in sub['metrics']])
            mean = stacked.mean(axis=0)
            std = stacked.std(axis=0)
            x = np.arange(mean.size)
            ax.plot(x, mean, color=style.ALGO_COLORS[algo], label=style.ALGO_LABELS[algo])
            ax.fill_between(x, mean - std, mean + std,
                            color=style.ALGO_COLORS[algo], alpha=0.2)
        ax.set_title(maze)
        ax.set_xlabel('Episode')
    axes[0].set_ylabel('Episodic return')
    axes[-1].legend(loc='lower right')

    output_dir.mkdir(parents=True, exist_ok=True)
    for ext in ('png', 'pdf'):
        fig.savefig(output_dir / f'convergence.{ext}', bbox_inches='tight')
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    """CLI: ``python -m maze_mdp.analysis.plot_convergence``."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, default=Path('data'))
    parser.add_argument('--output-dir', type=Path, default=Path('data/figures'))
    args = parser.parse_args(argv)
    plot(args.input_dir, args.output_dir)


if __name__ == '__main__':
    main()

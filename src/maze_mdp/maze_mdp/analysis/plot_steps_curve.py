"""Plot steps-to-goal vs. episode (mean ± std over seeds) for SARSA / Q-Learning."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from maze_mdp.analysis import style
from maze_mdp.analysis.loaders import load_training_runs


def _smooth(x: np.ndarray, window: int) -> np.ndarray:
    """Centered moving average without zero-padding at the boundaries.

    ``np.convolve(..., mode='same')`` pads with zeros, which biases the
    first ``window/2`` and last ``window/2`` smoothed samples toward 0 and
    creates a spurious dip at the right edge of the curve. Here we
    compute the rolling mean from a cumulative sum and divide by the
    actual number of samples inside each window, so boundary points are
    just shorter averages of the data that *does* exist.
    """
    if window <= 1 or x.size < window:
        return x
    half = window // 2
    cumsum = np.concatenate(([0.0], np.cumsum(x, dtype=np.float64)))
    out = np.empty_like(x, dtype=np.float64)
    n = x.size
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        out[i] = (cumsum[hi] - cumsum[lo]) / (hi - lo)
    return out


def plot(input_dir: Path, output_dir: Path, smooth: int = 25) -> None:
    """Render ``output_dir/steps_curve.{png,pdf}``."""
    style.apply()
    df = load_training_runs(input_dir / 'training')
    if df.empty:
        raise SystemExit(f'No runs found under {input_dir / "training"}')
    rl = df[df['algo'].isin(['sarsa', 'qlearning'])].copy()
    if rl.empty:
        raise SystemExit('No SARSA / Q-Learning runs found.')

    mazes = sorted(rl['maze'].unique())
    fig, axes = plt.subplots(1, len(mazes), figsize=(4.5 * len(mazes), 3.5), sharey=False)
    if len(mazes) == 1:
        axes = [axes]

    for ax, maze in zip(axes, mazes):
        for algo in ('sarsa', 'qlearning'):
            sub = rl[(rl['maze'] == maze) & (rl['algo'] == algo)]
            if sub.empty:
                continue
            stacked = np.stack([
                _smooth(s['steps'].to_numpy(dtype=float), smooth)
                for s in sub['metrics']
            ])
            mean = stacked.mean(axis=0)
            std = stacked.std(axis=0)
            x = np.arange(mean.size)
            ax.plot(x, mean, color=style.ALGO_COLORS[algo], label=style.ALGO_LABELS[algo])
            ax.fill_between(x, mean - std, mean + std,
                            color=style.ALGO_COLORS[algo], alpha=0.2)
        ax.set_title(maze)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Steps-to-goal')
    axes[-1].legend(loc='upper right')

    output_dir.mkdir(parents=True, exist_ok=True)
    for ext in ('png', 'pdf'):
        fig.savefig(output_dir / f'steps_curve.{ext}', bbox_inches='tight')
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    """CLI: ``python -m maze_mdp.analysis.plot_steps_curve``."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, default=Path('data'))
    parser.add_argument('--output-dir', type=Path, default=Path('data/figures'))
    parser.add_argument('--smooth', type=int, default=25,
                        help='Moving-average window over episodes (default: 25).')
    args = parser.parse_args(argv)
    plot(args.input_dir, args.output_dir, smooth=args.smooth)


if __name__ == '__main__':
    main()

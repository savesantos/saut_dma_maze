"""Plot value heatmap + policy arrows per (maze, algo) pair."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from maze_mdp.analysis import style
from maze_mdp.analysis.loaders import load_training_runs
from maze_mdp.mdp import MDP, Action, MDPConfig
from maze_mdp.policy import policy_value
from maze_mdp.simulator import FIXTURES, WALL

_ARROW = {
    int(Action.FORWARD): (0.0, 0.4),
    int(Action.TURN_LEFT): (-0.3, 0.0),
    int(Action.TURN_RIGHT): (0.3, 0.0),
}


def _aggregate_value(maze_name: str, pi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    maze = FIXTURES[maze_name]()
    mdp = MDP(maze, MDPConfig())
    V = policy_value(mdp, pi)
    grid = np.full((maze.rows, maze.cols), np.nan)
    arrow = np.zeros((maze.rows, maze.cols), dtype=np.int64)
    for r in range(maze.rows):
        for c in range(maze.cols):
            if maze.cells[r, c] == WALL:
                continue
            # Average V over heading; record dominant action over headings.
            states = [mdp.encode(r, c, h) for h in range(4)]
            grid[r, c] = float(np.mean(V[states]))
            arrow[r, c] = int(np.bincount([pi[s] for s in states]).argmax())
    return grid, arrow


def plot(input_dir: Path, output_dir: Path) -> None:
    """Render policy/value heatmaps to ``output_dir/policy_heatmap_*.{png,pdf}``."""
    style.apply()
    df = load_training_runs(input_dir / 'training')
    if df.empty:
        raise SystemExit(f'No runs found under {input_dir / "training"}')

    # Pick the lowest-seed run per (maze, algo) for a single representative figure.
    df = df.sort_values('seed').drop_duplicates(['maze', 'algo'])

    mazes = sorted(df['maze'].unique())
    algos = ['vi', 'sarsa', 'qlearning']
    fig, axes = plt.subplots(
        len(algos), len(mazes),
        figsize=(3.5 * len(mazes), 3.0 * len(algos)),
        squeeze=False,
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, algo in enumerate(algos):
        for j, maze in enumerate(mazes):
            ax = axes[i][j]
            sub = df[(df['algo'] == algo) & (df['maze'] == maze)]
            if sub.empty:
                ax.axis('off')
                continue
            pi = sub.iloc[0]['policy']['pi']
            grid, arrows = _aggregate_value(maze, pi)
            im = ax.imshow(grid, cmap='viridis')
            for r in range(grid.shape[0]):
                for c in range(grid.shape[1]):
                    if np.isnan(grid[r, c]):
                        continue
                    dx, dy = _ARROW[int(arrows[r, c])]
                    ax.arrow(c, r, dx, dy, head_width=0.15, color='white', alpha=0.9)
            ax.set_title(f'{style.ALGO_LABELS[algo]} — {maze}', fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(output_dir / f'policy_heatmap.{ext}', bbox_inches='tight')
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    """CLI: ``python -m maze_mdp.analysis.plot_policy_heatmap``."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, default=Path('data'))
    parser.add_argument('--output-dir', type=Path, default=Path('data/figures'))
    args = parser.parse_args(argv)
    plot(args.input_dir, args.output_dir)


if __name__ == '__main__':
    main()

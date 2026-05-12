"""Plot value heatmap + per-heading policy arrows per (maze, algo) pair."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from maze_mdp.analysis import style
from maze_mdp.analysis.loaders import load_training_runs
from maze_mdp.analysis.select_best_run import load_selected
from maze_mdp.mdp import MDP, Action, Heading, MDPConfig
from maze_mdp.policy import policy_value
from maze_mdp.simulator import FIXTURES, WALL

# Heading sub-position inside a cell, in (dx, dy) image coords.
# imshow's y-axis grows downward, so North is -y.
_HEADING_OFFSET = {
    int(Heading.N): (0.0, -0.28),
    int(Heading.E): (0.28, 0.0),
    int(Heading.S): (0.0, 0.28),
    int(Heading.W): (-0.28, 0.0),
}

# Heading -> physical (dx, dy) unit vector (image coords).
_HEADING_DIR = {
    int(Heading.N): (0.0, -1.0),
    int(Heading.E): (1.0, 0.0),
    int(Heading.S): (0.0, 1.0),
    int(Heading.W): (-1.0, 0.0),
}

# Action -> color so the legend is readable even when arrows overlap.
_ACTION_COLOR = {
    int(Action.FORWARD): 'white',
    int(Action.TURN_LEFT): '#ffd166',
    int(Action.TURN_RIGHT): '#ef476f',
}


def _value_grid(maze_name: str, pi: np.ndarray) -> tuple[np.ndarray, MDP]:
    maze = FIXTURES[maze_name]()
    mdp = MDP(maze, MDPConfig())
    V = policy_value(mdp, pi)
    grid = np.full((maze.rows, maze.cols), np.nan)
    for r in range(maze.rows):
        for c in range(maze.cols):
            if maze.cells[r, c] == WALL:
                continue
            states = [mdp.encode(r, c, h) for h in range(4)]
            grid[r, c] = float(np.mean(V[states]))
    return grid, mdp


def _draw_action_arrow(ax, r: int, c: int, heading: int, action: int) -> None:
    """Draw a tiny arrow at (r, c) for the policy entry at heading ``heading``."""
    ox, oy = _HEADING_OFFSET[heading]
    if action == int(Action.FORWARD):
        dx, dy = _HEADING_DIR[heading]
    elif action == int(Action.TURN_LEFT):
        dx, dy = _HEADING_DIR[(heading - 1) % 4]
    else:  # TURN_RIGHT
        dx, dy = _HEADING_DIR[(heading + 1) % 4]
    scale = 0.18
    ax.arrow(
        c + ox, r + oy, dx * scale, dy * scale,
        head_width=0.06, head_length=0.06,
        length_includes_head=True,
        color=_ACTION_COLOR[action], alpha=0.95, linewidth=0.8,
    )


def plot(input_dir: Path, output_dir: Path) -> None:
    """Render policy/value heatmaps to ``output_dir/policy_heatmap.{png,pdf}``."""
    style.apply()
    df = load_training_runs(input_dir / 'training')
    if df.empty:
        raise SystemExit(f'No runs found under {input_dir / "training"}')
    # Pick the seed promoted by ``select_best_run`` so the heatmap matches
    # whatever ``compare_viz`` / deployment is running. Fall back to the
    # lowest seed only when no selection has been made yet.
    def _pick(group: 'pd.DataFrame') -> 'pd.Series':  # type: ignore[name-defined]
        algo = group['algo'].iloc[0]
        maze = group['maze'].iloc[0]
        record = load_selected(algo, maze, training_root=input_dir / 'training')
        if record is not None:
            seed = int(record['selected']['seed'])
            match = group[group['seed'] == seed]
            if not match.empty:
                return match.iloc[0]
        return group.sort_values('seed').iloc[0]
    df = df.groupby(['maze', 'algo'], group_keys=False).apply(_pick).reset_index(drop=True)

    mazes = sorted(df['maze'].unique())
    algos = ['vi', 'sarsa', 'qlearning']
    fig, axes = plt.subplots(
        len(algos), len(mazes),
        figsize=(3.7 * len(mazes), 3.2 * len(algos)),
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
            grid, mdp = _value_grid(maze, pi)
            im = ax.imshow(grid, cmap='viridis')
            goal = mdp.maze.goal
            for r in range(grid.shape[0]):
                for c in range(grid.shape[1]):
                    if np.isnan(grid[r, c]) or (r, c) == goal:
                        continue
                    for h in range(4):
                        s = mdp.encode(r, c, h)
                        _draw_action_arrow(ax, r, c, h, int(pi[s]))
            ax.scatter(
                [goal[1]], [goal[0]], marker='*', s=120,
                color='gold', edgecolors='black', linewidths=0.7, zorder=5,
            )
            ax.set_title(f'{style.ALGO_LABELS[algo]} — {maze}', fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    handles = [
        plt.Line2D([0], [0], color=_ACTION_COLOR[int(Action.FORWARD)], lw=2,
                   label='forward'),
        plt.Line2D([0], [0], color=_ACTION_COLOR[int(Action.TURN_LEFT)], lw=2,
                   label='turn left'),
        plt.Line2D([0], [0], color=_ACTION_COLOR[int(Action.TURN_RIGHT)], lw=2,
                   label='turn right'),
    ]
    legend = fig.legend(handles=handles, loc='lower center', ncol=3,
                        frameon=True, bbox_to_anchor=(0.5, -0.01))
    legend.get_frame().set_edgecolor('black')
    legend.get_frame().set_linewidth(0.8)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
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

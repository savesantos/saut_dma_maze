"""
Single-run trainer for VI / SARSA / Q-Learning.

CLI entry point ``train`` (registered in ``setup.py``) writes the canonical
artifact layout described in ``AGENTS.md`` under
``data/training/<algo>/<maze>/<run_id>/``:

- ``policy.npz``   — pi (and Q for RL, V for VI)
- ``params.yaml``  — full hyperparameters used
- ``metrics.csv``  — per-episode metrics (only for RL)
- ``summary.json`` — seed, wall-clock, convergence info, paths
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from maze_mdp._td import TDConfig
from maze_mdp.mdp import MDP, MDPConfig
from maze_mdp.policy import greedy_policy, save_policy
from maze_mdp.qlearning import q_learning
from maze_mdp.sarsa import sarsa
from maze_mdp.simulator import FIXTURES, GridMaze
from maze_mdp.value_iteration import value_iteration

ALGOS = ('vi', 'sarsa', 'qlearning')


def _make_run_id(seed: int) -> str:
    return f'{datetime.now().strftime("%Y%m%d-%H%M%S")}-seed{seed}'


def _resolve_maze(maze_name: str):
    if maze_name not in FIXTURES:
        raise ValueError(
            f'Unknown maze {maze_name!r}. Known: {sorted(FIXTURES)}'
        )
    return FIXTURES[maze_name]()


def _write_metrics_csv(path: Path, info) -> None:
    with path.open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['episode', 'return', 'steps', 'epsilon', 'mean_td_error'])
        for i, (ret, steps, eps, td) in enumerate(zip(
            info.episode_returns,
            info.episode_steps,
            info.episode_epsilon,
            info.episode_td_error,
        )):
            writer.writerow([i, ret, steps, eps, td])


def train_one(
    algo: str,
    maze_name: str,
    seed: int,
    out_root: Path,
    *,
    mdp_config: MDPConfig | None = None,
    td_config: TDConfig | None = None,
    vi_tol: float = 1e-6,
) -> dict[str, Any]:
    """Run a single training job, persist artifacts, and return a summary dict."""
    if algo not in ALGOS:
        raise ValueError(f'algo must be one of {ALGOS}, got {algo!r}')
    maze = _resolve_maze(maze_name)
    mdp_config = mdp_config or MDPConfig()
    td_config = td_config or TDConfig()
    run_id = _make_run_id(seed)
    run_dir = out_root / 'training' / algo / maze_name / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {
        'algo': algo,
        'maze': maze_name,
        'seed': int(seed),
        'run_id': run_id,
        'run_dir': str(run_dir),
    }
    params: dict[str, Any] = {
        'algo': algo,
        'maze': maze_name,
        'seed': int(seed),
        'mdp_config': asdict(mdp_config),
    }

    if algo == 'vi':
        mdp = MDP(maze, mdp_config)
        V, pi, info = value_iteration(mdp, tol=vi_tol)
        save_policy(run_dir / 'policy.npz', pi=pi, V=V, metadata={'algo': 'vi'})
        summary.update({
            'iterations': info.iterations,
            'converged': info.converged,
            'wall_clock_s': info.wall_clock_s,
        })
        params['vi_tol'] = vi_tol
    else:
        env = GridMaze(maze, config=mdp_config)
        algo_fn = sarsa if algo == 'sarsa' else q_learning
        Q, info = algo_fn(env, td_config, seed=seed)
        pi = greedy_policy(Q)
        save_policy(run_dir / 'policy.npz', pi=pi, Q=Q, metadata={'algo': algo})
        _write_metrics_csv(run_dir / 'metrics.csv', info)
        summary.update({
            'episodes': td_config.n_episodes,
            'final_return_mean_last50': float(np.mean(info.episode_returns[-50:])),
            'wall_clock_s': info.wall_clock_s,
        })
        params['td_config'] = asdict(td_config)

    (run_dir / 'params.yaml').write_text(yaml.safe_dump(params, sort_keys=True))
    (run_dir / 'summary.json').write_text(json.dumps(summary, indent=2))
    return summary


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Train a single policy and persist artifacts.'
    )
    parser.add_argument('--algo', choices=ALGOS, required=True)
    parser.add_argument('--maze', required=True, choices=sorted(FIXTURES))
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out', type=Path, default=Path('data'))
    parser.add_argument('--episodes', type=int, default=None)
    parser.add_argument('--gamma', type=float, default=None)
    parser.add_argument('--vi-tol', type=float, default=1e-6)
    return parser


def main(argv: list[str] | None = None) -> None:
    """CLI entry point ``train`` — see ``--help`` for usage."""
    args = _build_arg_parser().parse_args(argv)
    mdp_kwargs = {}
    if args.gamma is not None:
        mdp_kwargs['gamma'] = args.gamma
    mdp_config = MDPConfig(**mdp_kwargs) if mdp_kwargs else MDPConfig()
    td_kwargs = {}
    if args.episodes is not None:
        td_kwargs['n_episodes'] = args.episodes
    td_config = TDConfig(**td_kwargs) if td_kwargs else TDConfig()
    summary = train_one(
        algo=args.algo,
        maze_name=args.maze,
        seed=args.seed,
        out_root=args.out,
        mdp_config=mdp_config,
        td_config=td_config,
        vi_tol=args.vi_tol,
    )
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()

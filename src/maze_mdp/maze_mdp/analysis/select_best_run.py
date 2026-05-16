"""
Pick the best trained policy per (algo, maze) by held-out evaluation.

For every ``data/training/<algo>/<maze>/<run_id>/policy.npz`` we run ``K``
evaluation episodes through :class:`GridMaze` with a fixed evaluation seed
and the slip probability recorded in the training ``params.yaml``. The
policy that maximizes mean return (tiebreak: fewer mean steps) is written
to ``data/training/<algo>/<maze>/selected.json`` so downstream consumers
(plots, RViz launches, deployment) all dereference a single source of truth.

Example invocations::

    python -m maze_mdp.analysis.select_best_run --eval-episodes 200
    python -m maze_mdp.analysis.select_best_run --print qlearning fixture_7x7_loop
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from maze_mdp.mdp import MDPConfig
from maze_mdp.policy import greedy_policy, load_policy
from maze_mdp.simulator import FIXTURES, GridMaze


_DEFAULT_EVAL_SEED = 20260512
_DEFAULT_EVAL_EPISODES = 200


def _policy_pi(bundle: dict) -> np.ndarray:
    if 'pi' in bundle:
        return np.asarray(bundle['pi'], dtype=np.int64)
    if 'Q' in bundle:
        return greedy_policy(np.asarray(bundle['Q']))
    raise RuntimeError('Bundle has neither pi nor Q.')


def _mdp_config_from_params(params_path: Path) -> 'MDPConfig':
    """Build an :class:`MDPConfig` from a run's ``params.yaml``.

    Each archived scenario uses a different MDP (e.g. ``goal_reward``),
    so evaluation must mirror the training MDP for the reported
    ``mean_return`` to make sense across scenarios. ``success_rate`` is
    invariant under reward scaling, but the printed ranking keeps using
    return, so we keep this consistent.
    """
    from maze_mdp.mdp import MDPConfig
    if not params_path.exists():
        return MDPConfig()
    data = yaml.safe_load(params_path.read_text()) or {}
    mdp_cfg = data.get('mdp_config') or {}
    accepted = {
        'slip_prob', 'turn_fail_prob', 'forward_cost', 'turn_cost',
        'bump_cost', 'goal_reward', 'gamma',
    }
    return MDPConfig(**{k: v for k, v in mdp_cfg.items() if k in accepted})


def _slip_from_params(params_path: Path, fallback: float) -> float:
    if not params_path.exists():
        return fallback
    data = yaml.safe_load(params_path.read_text()) or {}
    return float(data.get('mdp_config', {}).get('slip_prob', fallback))


def evaluate_policy(
    pi: np.ndarray,
    maze_name: str,
    *,
    eval_seed: int = _DEFAULT_EVAL_SEED,
    n_episodes: int = _DEFAULT_EVAL_EPISODES,
    slip_prob: float = 0.1,
    mdp_config: 'MDPConfig | None' = None,
) -> dict:
    """Roll out ``pi`` for ``n_episodes`` and return aggregate metrics."""
    if maze_name not in FIXTURES:
        raise KeyError(f'Unknown maze fixture: {maze_name!r}')
    maze = FIXTURES[maze_name]()
    config = mdp_config if mdp_config is not None else MDPConfig(slip_prob=slip_prob)
    env = GridMaze(
        maze=maze,
        config=config,
        rng=np.random.default_rng(eval_seed),
    )
    returns = np.empty(n_episodes, dtype=np.float64)
    steps = np.empty(n_episodes, dtype=np.int64)
    successes = np.zeros(n_episodes, dtype=bool)
    for ep in range(n_episodes):
        s = env.reset()
        ep_return = 0.0
        while True:
            a = int(pi[s])
            s, r, done, info = env.step(a)
            ep_return += r
            if done:
                break
        returns[ep] = ep_return
        steps[ep] = int(info['steps'])
        successes[ep] = bool(info['terminated'])
    return {
        'mean_return': float(returns.mean()),
        'std_return': float(returns.std(ddof=0)),
        'mean_steps': float(steps.mean()),
        'success_rate': float(successes.mean()),
        'n_episodes': int(n_episodes),
        'eval_seed': int(eval_seed),
        'slip_prob': float(config.slip_prob),
    }


def _score_key(metrics: dict) -> tuple[float, float]:
    """Sort key: maximize mean_return, tiebreak by fewer mean_steps."""
    return (metrics['mean_return'], -metrics['mean_steps'])


def select_best_for(
    algo_dir: Path,
    maze_name: str,
    *,
    eval_seed: int,
    n_episodes: int,
) -> dict | None:
    """Pick the best run inside ``algo_dir/maze_name/`` and return the record."""
    maze_dir = algo_dir / maze_name
    run_dirs = sorted(p for p in maze_dir.iterdir() if p.is_dir())
    if not run_dirs:
        return None
    candidates: list[dict] = []
    for run_dir in run_dirs:
        policy_path = run_dir / 'policy.npz'
        if not policy_path.exists():
            continue
        bundle = load_policy(policy_path)
        pi = _policy_pi(bundle)
        params_path = run_dir / 'params.yaml'
        slip = _slip_from_params(params_path, fallback=0.1)
        mdp_cfg = _mdp_config_from_params(params_path)
        metrics = evaluate_policy(
            pi,
            maze_name,
            eval_seed=eval_seed,
            n_episodes=n_episodes,
            slip_prob=slip,
            mdp_config=mdp_cfg,
        )
        summary_path = run_dir / 'summary.json'
        summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
        candidates.append({
            'run_id': run_dir.name,
            'run_dir': str(run_dir),
            'policy_path': str(policy_path),
            'seed': int(summary.get('seed', -1)),
            'eval': metrics,
        })
    if not candidates:
        return None
    candidates.sort(key=lambda c: _score_key(c['eval']), reverse=True)
    best = candidates[0]
    return {
        'algo': algo_dir.name,
        'maze': maze_name,
        'selected': best,
        'all': candidates,
    }


def select_all(
    training_root: Path,
    *,
    eval_seed: int = _DEFAULT_EVAL_SEED,
    n_episodes: int = _DEFAULT_EVAL_EPISODES,
) -> list[dict]:
    """Run :func:`select_best_for` over every (algo, maze) pair under ``training_root``."""
    if not training_root.exists():
        raise SystemExit(f'No training data under {training_root}')
    records: list[dict] = []
    for algo_dir in sorted(p for p in training_root.iterdir() if p.is_dir()):
        for maze_dir in sorted(p for p in algo_dir.iterdir() if p.is_dir()):
            record = select_best_for(
                algo_dir,
                maze_dir.name,
                eval_seed=eval_seed,
                n_episodes=n_episodes,
            )
            if record is None:
                continue
            out_path = maze_dir / 'selected.json'
            out_path.write_text(json.dumps(record, indent=2))
            records.append(record)
    return records


def load_selected(
    algo: str,
    maze: str,
    training_root: Path = Path('data/training'),
) -> dict | None:
    """Return the ``selected.json`` record for ``(algo, maze)`` if present."""
    path = training_root / algo / maze / 'selected.json'
    if not path.exists():
        return None
    return json.loads(path.read_text())


def selected_policy_path(
    algo: str,
    maze: str,
    training_root: Path = Path('data/training'),
) -> Path | None:
    """Return the absolute policy path picked by :func:`select_all`."""
    record = load_selected(algo, maze, training_root)
    if record is None:
        return None
    return Path(record['selected']['policy_path']).resolve()


def main(argv: list[str] | None = None) -> None:
    """CLI: ``python -m maze_mdp.analysis.select_best_run``."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-root', type=Path, default=Path('data/training'))
    parser.add_argument('--eval-seed', type=int, default=_DEFAULT_EVAL_SEED)
    parser.add_argument('--eval-episodes', type=int, default=_DEFAULT_EVAL_EPISODES)
    parser.add_argument(
        '--print',
        nargs=2,
        metavar=('ALGO', 'MAZE'),
        help='Print the selected policy path for (algo, maze) and exit.',
    )
    args = parser.parse_args(argv)

    if args.print:
        algo, maze = args.print
        path = selected_policy_path(algo, maze, args.training_root)
        if path is None:
            raise SystemExit(
                f'No selected.json for {algo}/{maze}. '
                'Run `python -m maze_mdp.analysis.select_best_run` first.'
            )
        print(path)
        return

    records = select_all(
        args.training_root,
        eval_seed=args.eval_seed,
        n_episodes=args.eval_episodes,
    )
    print(f'Wrote selected.json for {len(records)} (algo, maze) pairs:')
    for rec in records:
        sel = rec['selected']
        ev = sel['eval']
        print(
            f"  {rec['algo']:9s} {rec['maze']:24s} "
            f"-> seed={sel['seed']:>2}  "
            f"return={ev['mean_return']:+.3f}  "
            f"steps={ev['mean_steps']:.1f}  "
            f"success={ev['success_rate']:.2%}"
        )


if __name__ == '__main__':
    main()

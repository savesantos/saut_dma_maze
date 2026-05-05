# Persisted Artifacts — Schema Reference

Every run writes a self-describing folder so the report's figures can be
regenerated from raw data alone.

## Training runs

`data/training/<algo>/<maze>/<run_id>/`

| File           | Type | Schema |
| -------------- | ---- | ------ |
| `policy.npz`   | NumPy archive | `pi: int64[S]`, optional `Q: float64[S, A]`, `V: float64[S]`, `metadata: object` (dict). |
| `params.yaml`  | YAML | `{algo, maze, seed, mdp_config, td_config?, vi_tol?}`. |
| `metrics.csv`  | CSV (RL only) | Header: `episode,return,steps,epsilon,mean_td_error`. One row per training episode. |
| `summary.json` | JSON | `{algo, maze, seed, run_id, run_dir, wall_clock_s, ...}` + algorithm-specific keys (`iterations` / `converged` for VI; `episodes` / `final_return_mean_last50` for RL). |

`run_id = YYYYMMDD-HHMMSS-seed<N>`.

## Deployment runs

`data/deployment/<maze>/<run_id>/`

| File              | Type | Schema |
| ----------------- | ---- | ------ |
| `bag/`            | rosbag2 | Topics: `/image/compressed`, `/robot_cell`, `/alphabot2/cmd_vel`, `/maze`. |
| `trajectory.csv`  | CSV | `timestamp,row,col,heading,action`. Extracted post-hoc. |
| `summary.json`    | JSON | `{algo, maze, run_id, success, steps, wall_clock_s, failure_mode?}`. |

## Reproducibility rules

- Every run records its `seed`. Runs without a seed are not acceptable for
  the comparative analysis.
- `data/` is gitignored; runs live only on developer machines and the lab
  storage. A small `data/fixtures/` reference run can be committed for
  regression tests.

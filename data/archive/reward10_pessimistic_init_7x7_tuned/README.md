# `goal_reward = 10.0` + pessimistic Q init + extended 7x7 budget

Combination of the two algorithm-side and schedule-side fixes
explored separately in `reward10_pessimistic_init/` and
`reward10_7x7_tuned/`. This is the "best result" archive — both
remedies applied at the same time.

## What changed vs. `reward10_pessimistic_init/`

Only [src/maze_bringup/config/sweeps/default.yaml](../../../src/maze_bringup/config/sweeps/default.yaml),
inside `td_config_overrides.fixture_7x7_loop`:

- `n_episodes`: 30 000 → **80 000**.
- `epsilon_anneal_frac`: 0.5 → **0.8** (slower ε decay).

Pessimistic Q init in [_td.py](../../../src/maze_mdp/maze_mdp/_td.py)
is preserved (`Q = -forward_cost / (1-γ) * np.ones(...)`,
i.e. `-100`). `MDPConfig`, the other two mazes, all seeds and the
remaining TD hyperparameters are unchanged.

## What was achieved

Read off [comparison.png](comparison.png), means over 5 seeds,
evaluated under the original stochastic dynamics. The y-axis on
the left subplot now tops out at **0.27** (units of value) — vs
**25** in the original `reward10/` run.

| metric                 | `reward10_pessimistic_init/` | this run (both fixes)        |
| ---------------------- | ---------------------------- | ---------------------------- |
| SARSA `V*` − `Vπ` 5x5  | ≈ 0.2                        | ≈ 0.2 (essentially the same) |
| SARSA success 5x5      | 1.00                         | 1.00                         |
| SARSA `V*` − `Vπ` 7x7  | ≈ 3.0                        | **≈ 0.02**                   |
| SARSA success 7x7      | 0.95                         | **1.00**                     |
| QL `V*` − `Vπ` 7x7     | ≈ 0.0                        | ≈ 0.0 (unchanged)            |
| QL success 7x7         | 1.00                         | 1.00                         |

Versus the original baseline:

| metric                 | `reward10/` (zeros, 30k) | this run (both fixes, 80k) |
| ---------------------- | ------------------------ | -------------------------- |
| SARSA `V*` − `Vπ` 5x5  | ≈ 2.8                    | **≈ 0.2** (~14× better)    |
| SARSA `V*` − `Vπ` 7x7  | ≈ 24.5                   | **≈ 0.02** (~1000× better) |
| SARSA success 7x7      | 0.75                     | **1.00**                   |

Every algorithm reaches the goal on every rollout for every
maze. SARSA's residual gap is ≈ 0.18 on 5x5 (the genuine
ε-soft optimum gap, intrinsic to on-policy bootstrapping with
`epsilon_min = 0.05`) and ≈ 0.02 on 7x7 — both well within the
seed-to-seed standard deviation.

## What was confirmed by combining the fixes

The previous archive `reward10_pessimistic_init/` left an
ambiguity: was SARSA's residual ≈ 3 gap on 7x7 the *intrinsic*
ε-soft optimum, or just under-training at 30 k episodes? This
combined run answers it: with 80 k episodes and slower ε decay
**on top of** pessimistic init, the gap drops to ≈ 0.02 — so
the 30 k version was still under-trained, and the true ε-soft
gap on 7x7 is essentially nil for this MDP.

The 5x5 SARSA gap (~0.2), in contrast, is *unaffected* by the
extra budget. That's the genuine on-policy ε-soft optimum
correction at `epsilon_min = 0.05`: SARSA evaluates `Q^πε`, so
its greedy projection is slightly worse than `V*` even at
infinity. To close it further you would need to anneal ε to 0
(at the cost of losing convergence guarantees in the tabular
case unless `α` also goes to 0).

## What is still suboptimal

- **SARSA on 5x5 still has ≈ 0.2 gap.** Intrinsic ε-soft optimum;
  not a tuning problem. Worth flagging in the report as a
  *property* of SARSA rather than an artifact.
- **Unreachable interior cells of `fixture_7x7_loop`** still show
  arbitrary policy arrows in the heatmap. Cosmetic; a one-line
  mask in [plot_policy_heatmap.py](../../../src/maze_mdp/maze_mdp/analysis/plot_policy_heatmap.py)
  would clean it up before publication.
- **2.7× longer wall-clock** vs `reward10_pessimistic_init/`
  because of the 80 k 7x7 budget. For weekly experimentation
  the 30 k pessimistic-init config is better; for the final
  report figures, this combined-fix config is the one to use.

## Reproducing

Everything needed to re-run this scenario is pinned in the archive:

- `sweep.yaml` — exact sweep configuration consumed by the trainer.
- `training/<algo>/<maze>/<run_id>/` — `policy.npz`, `params.yaml`,
  `metrics.csv`, `summary.json` for every (algo, maze, seed) triple,
  plus `selected.json` identifying the policy picked by
  `select_best_run` for downstream consumers.
- `figures/` — PNG + PDF versions of all four canonical plots.

To regenerate the figures from the archived policies (no retraining):

```bash
cd /home/salva/saut_dma_maze
rm -rf data/training data/figures
cp -a data/archive/reward10_pessimistic_init_7x7_tuned/training data/
PYTHONPATH=src/maze_mdp bash scripts/make_all_figures.sh
```

To retrain end-to-end from the pinned sweep config:

```bash
cd /home/salva/saut_dma_maze
bash scripts/rerun_archive.sh reward10_pessimistic_init_7x7_tuned
```

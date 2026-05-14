# `goal_reward = 10.0` + pessimistic Q init

Same MDP and sweep schedule as `reward10/` (zeros Q init, 30 k episodes
on 7x7, ε annealed over 50 % of training). Only the **algorithm-side**
initialisation of `Q` was changed in `_td.py`. This is the apples-to-
apples test for "does pessimistic init alone fix the SARSA gap?".

## What changed vs. `reward10/`

In [src/maze_mdp/maze_mdp/_td.py](../../../src/maze_mdp/maze_mdp/_td.py),
inside `td_control`:

```python
q_init = -env.mdp.config.forward_cost / (1.0 - gamma)   # = -100 with defaults
Q = np.full((env.n_states, env.n_actions), q_init, dtype=np.float64)
```

Replaces `Q = np.zeros(...)`. The value `-forward_cost / (1-γ)` is the
worst-case discounted return assuming the agent only ever pays forward
costs and never reaches the goal, so it is a true lower bound on `Q*`.

Goal cells are absorbing terminals; the TD loop sets `bootstrap = 0.0`
on `done`, so the pessimistic Q values at terminal states are never
read. They stay at `-100` but do not contaminate any update.

Sweep config and `MDPConfig` are **unchanged** vs. `reward10/`:

- 7x7 budget back to 30 000 episodes (vs 80 000 in `reward10_7x7_tuned/`).
- `epsilon_anneal_frac` back to 0.5 (vs 0.8 in `reward10_7x7_tuned/`).
- All other hyperparameters and seeds identical to `reward10/`.

The companion test `test/test_td_control.py::test_qlearning_recovers_
optimal_policy_on_3x3` was updated from 1500 to 4000 episodes — with
pessimistic init the early greedy steps are noisier, so the convergence
test needs more episodes to be robust. This is a test-budget change,
not an algorithm change.

## What was achieved

Read off [comparison.png](comparison.png), means over 5 seeds, evaluated
under the original stochastic dynamics:

| metric                 | `reward10/` (zeros init) | this run (pessimistic init) |
| ---------------------- | ------------------------ | --------------------------- |
| SARSA `V*` − `Vπ` 5x5  | ≈ 2.8                    | **≈ 0.2**                   |
| SARSA success 5x5      | ≈ 0.96                   | **1.00**                    |
| SARSA `V*` − `Vπ` 7x7  | ≈ 24.5                   | **≈ 3.0**                   |
| SARSA success 7x7      | 0.75                     | **0.95**                    |
| QL `V*` − `Vπ` 7x7     | ≈ 3.1                    | **≈ 0.0**                   |
| QL success 7x7         | 0.97                     | **1.00**                    |
| QL on 3x3 / 5x5        | optimal                  | optimal (unchanged)         |

Versus the alternative fix `reward10_7x7_tuned/` (80 k episodes,
ε anneal 0.8) on the 7x7 loop:

| metric           | `reward10_7x7_tuned/` | this run (pessimistic init) |
| ---------------- | --------------------- | --------------------------- |
| SARSA `V*` − `Vπ` | ≈ 4.0                 | **≈ 3.0**                   |
| SARSA success    | 0.94                  | **0.95**                    |
| episode budget   | 80 000                | **30 000**                  |

So pessimistic init alone, at the *original* 30 k budget, matches or
beats the tuned-budget result on 7x7 SARSA, AND fixes the residual
~2.8 gap on 5x5 SARSA that budget tuning did not touch. The
algorithm-side change is more efficient and more general than the
schedule tuning.

## Why it works

The previous pathology (see `reward0_baseline/README.md` and the
discussion of `reward10/`) was that ε-greedy at rarely-visited states
preferred the **less-tried** action because zero-initialised Q is
optimistic relative to the negative step costs that get back-propagated
into visited entries. Pessimistic init (`-100`) flips this: any
visited entry whose updates have moved Q above `-100` is now
*better* than an unvisited entry, so the argmax stops being biased
toward unexplored actions at corner / under-sampled states.

`forward_cost / (1-γ)` is the natural choice because it is a tight
lower bound on the true value function (the worst infinite-horizon
discounted return assuming only step costs are paid). Anything more
pessimistic still works; anything less pessimistic risks not being a
valid lower bound and re-introducing the bias.

## What is still suboptimal

- **SARSA on 7x7 still has `V*-Vπ` ≈ 3 and success 0.95.** This
  appears to be the genuine on-policy ε-soft optimum, not a
  pathology — SARSA's bootstrap target `Q[s', a']` averages over
  the ε-soft behaviour policy, so the converged greedy policy is
  evaluated on `V_πε`, not `V*`. With `epsilon_min = 0.05` and a
  large maze where most cells have a 5 % chance of a wrong-action
  excursion per visit, a small residual gap is intrinsic. The
  4× drop in confidence-band width (vs `reward10_7x7_tuned/`)
  suggests run-to-run variance is mostly resolved.
- **Unreachable interior cells of `fixture_7x7_loop`** still show
  arbitrary policy arrows in the heatmap. Cosmetic; mask them in
  the plot before the report.
- **Convergence curves on 7x7** ([convergence.png](convergence.png))
  start much further below zero than in `reward10/` because the
  early greedy estimate is `-100` everywhere; the curves take ~5 k
  episodes to climb out of the pessimistic regime. This is the
  expected cost of pessimistic init and does not prevent
  convergence — it just shifts the y-axis.

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
cp -a data/archive/reward10_pessimistic_init/training data/
PYTHONPATH=src/maze_mdp bash scripts/make_all_figures.sh
```

To retrain end-to-end from the pinned sweep config:

```bash
cd /home/salva/saut_dma_maze
bash scripts/rerun_archive.sh reward10_pessimistic_init
```

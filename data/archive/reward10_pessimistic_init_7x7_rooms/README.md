# `fixture_7x7_rooms` with the "best" config (pessimistic init + tuned 7x7 schedule)

New maze added on 2026-05-16, mirroring the cardboard 7x7 maze used
in the lab (three irregular wall clusters, free perimeter except a
wall at `(_, 5)` on several rows, goal at the bottom-left `(6, 0)`).
Layout:

```
. . . . . # .
# # . # . . .
. . . # . # .
# . # # . # .
. . . . . # .
# # # # . . .
G . . . . # .
```

## Config

This archive runs **only** the new maze with the exact same
hyperparameters as the project-best scenario
[`reward10_pessimistic_init_7x7_tuned/`](../reward10_pessimistic_init_7x7_tuned/README.md)
applied to the 7x7 case:

- Pessimistic Q init (`Q = -forward_cost / (1-γ) * 1`, i.e. `-100`).
- `n_episodes = 80 000`, `epsilon_anneal_frac = 0.8`,
  `epsilon_min = 0.05`, `alpha0 = 0.1`, `alpha_min = 0.001`,
  `alpha_tau = 50`.
- `MDPConfig`: `gamma = 0.99`, `slip_prob = 0.1`,
  `goal_reward = 10.0`.
- 5 seeds × {VI, SARSA, Q-Learning} = 15 runs.

See [sweep.yaml](sweep.yaml) for the pinned config and
[../../../src/maze_bringup/config/sweeps/archive/reward10_pessimistic_init_7x7_rooms.yaml](../../../src/maze_bringup/config/sweeps/archive/reward10_pessimistic_init_7x7_rooms.yaml)
for the source.

## Results

Mean over 5 seeds, 200 evaluation rollouts per seed under the
training MDP dynamics (`slip_prob = 0.1`).

| algorithm  | mean return | mean steps | success rate |
| ---------- | ----------- | ---------- | ------------ |
| VI         | −9.18       | 15.1       | 100 %        |
| SARSA      | −9.10       | 15.0       | 100 %        |
| Q-Learning | −9.18       | 15.1       | 100 %        |

Best-per-algo seeds selected for the heatmap (see
`training/<algo>/fixture_7x7_rooms/selected.json`): VI seed 0,
SARSA seed 1, Q-Learning seed 0. All three converge to the same
optimal path topology (down the central col 4 corridor, then
left along the bottom row to `(6, 0)`).

Optimality gap `V* − V^π` (averaged over free cells):

- VI: ≈ 0 (deterministic, by construction).
- Q-Learning: ≈ 0.002 — essentially `V*`.
- SARSA: ≈ 0.026 — the intrinsic ε-soft residual at
  `epsilon_min = 0.05`. Same property as on
  `fixture_5x5_corridor` / `fixture_7x7_loop` in the
  `reward10_pessimistic_init_7x7_tuned` archive, not a tuning issue.

## Figures

- [convergence.png](convergence.png) — episode-return curves for
  SARSA and Q-Learning, mean ± std over 5 seeds. Both stabilize
  near the optimum around episode ~40 k.
- [steps_curve.png](steps_curve.png) — steps-to-goal vs. episode.
- [policy_heatmap.png](policy_heatmap.png) — final greedy policy
  + state value overlay for the best seed of each algorithm.
- [comparison.png](comparison.png) — bar plot of
  `V* − V^π` and success rate, VI vs. SARSA vs. Q-Learning.

## Reproduce

```bash
bash scripts/rerun_archive.sh reward10_pessimistic_init_7x7_rooms
```

Wall-clock ≈ 10 min on a laptop CPU (SARSA ~7 min, Q-Learning
~10 min for 5 × 80 k episodes; VI is sub-second).

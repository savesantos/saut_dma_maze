# MDP design

Specification of the Markov Decision Process used by Value Iteration, SARSA and Q-Learning in this project.
All choices below are implemented in [src/maze_mdp/maze_mdp/mdp.py](../src/maze_mdp/maze_mdp/mdp.py) (to be scaffolded) and consumed ROS-free by the micro-simulator.

---

## 1. State space $S$

$$s = (r, c, \theta), \qquad \theta \in \{N, E, S, W\}$$

- Discrete cell $(r, c)$ on a 4-connected grid plus a 4-valued heading.
- $|S| = 4 \cdot R \cdot C$, scales linearly with maze size.
- Flat encoding: `s = (r * C + c) * 4 + theta_idx` with `N=0, E=1, S=2, W=3`.
- Walls are *not* states; they are blocked transitions.
- Goal cell is a single absorbing state.

Rationale: matches the AlphaBot2's differential-drive reality (turning has a cost), is fiducial-localizable (camera + ArUco/AprilTag → discrete cell + coarse heading), and stays small enough for tabular VI/SARSA/Q-Learning at any reasonable maze size.
8 headings or 8-connected grids are deferred until the maze spec includes diagonal passages.

## 2. Action space $A$

$$\mathcal{A} = \{\texttt{forward}, \texttt{turn\_left}, \texttt{turn\_right}\}, \qquad |A| = 3$$

- `forward`: move one cell in direction $\theta$ if no wall, else stay.
- `turn_left`: $\theta \leftarrow (\theta - 1) \bmod 4$, cell unchanged.
- `turn_right`: $\theta \leftarrow (\theta + 1) \bmod 4$, cell unchanged.

Encoding: `FORWARD=0, TURN_LEFT=1, TURN_RIGHT=2`.

Rationale: maps 1-to-1 to AlphaBot2 primitives on `/alphabot2/cmd_vel`. `backward` is excluded (no rear sensors → unsafe; two turns already produce a 180° flip). `stay` / `noop` is excluded (creates trivial absorbing loops in RL). Holonomic `{go_N, go_E, go_S, go_W}` is excluded (would hide turn cost from the planner and contradict our heading-augmented state).

## 3. Transition model $P(s' \mid s, a)$

Slip model with two parameters:

- $p_s = 0.1$ — per-side probability that `forward` slips sideways (translation only, no rotation).
  Outcome distribution: $0.8$ intended forward, $0.1$ slip-left, $0.1$ slip-right.
- $p_t = 0.0$ — probability that a turn fails to rotate (motor stall).

Boundary rules apply to every outcome (intended or slipped):

- Move into a wall or off-grid → **stay** in current cell, heading unchanged.
- Goal cell is **absorbing**: $P(s_\text{goal} \mid s_\text{goal}, a) = 1$ for all $a$, $R = 0$.

Storage: dense `np.ndarray` of shape `(|S|, |A|, |S|)`, sparse in practice (≤ 3 non-zeros per row).

Rationale: $p_s = 0.1$ matches the canonical AIMA 4×3 gridworld (Russell & Norvig, 4th ed. 2020, Ch. 17 §17.1, Fig. 17.1: 0.8 intended / 0.1 each right-angle perturbation); $p_t = 0$ reflects empirically near-deterministic in-place turns. Both are exposed as `MDPConfig` parameters so a Week-5 sweep or hardware-estimated values can override them without touching algorithm code.

## 4. Reward function $R(s, a, s')$

$$
R(s, a, s') =
\begin{cases}
+R_\text{goal} & s' = s_\text{goal} \\
-c_\text{fwd} - c_\text{bump} \cdot \mathbb{1}[s' = s] & a = \texttt{forward} \\
-c_\text{turn} & a \in \{\texttt{turn\_left}, \texttt{turn\_right\}}
\end{cases}
$$

Defaults:

| Symbol            | Value | Role                                                                   |
| ----------------- | ----- | ---------------------------------------------------------------------- |
| $c_\text{fwd}$    | 1.0   | Unit step cost (anchor).                                               |
| $c_\text{turn}$   | 1.2   | 20% premium over forward — smooth policies, no long detours.           |
| $c_\text{bump}$   | 2.5   | Forward bump penalty; corridor (2 walls) ≈ half a step.                |
| $R_\text{goal}$   | 0.0   | Absorbing goal; discounting alone yields shortest path.                |

Storage: `R` of shape `(|S|, |A|, |S|)`, so the simulator can return the actually-sampled reward to RL while VI computes $\bar R(s,a) = \sum_{s'} P(s'\mid s,a)\,R(s,a,s')$ on the fly.

Rationale (sized by expected bump cost per `forward` at $p_s = 0.1$):

| Cell geometry              | $\mathbb{E}[\text{bump}]$ at $c_\text{bump}=2.5$ |
| -------------------------- | ------------------------------------------------ |
| Open (no adjacent walls)   | 0                                                |
| Single-side wall           | 0.25                                             |
| Two-side wall (corridor)   | 0.5                                              |
| Forward into known wall    | 2.0                                              |

Below 2.5 the wall-awareness disappears; above ~5 it overrides path length and the agent takes paranoid detours. 2.5 anchors "1-cell corridor ≈ half step", appropriate for typical project mazes that are mostly corridors.
Differentiated forward/turn costs are the right lever for "smoother turn-taking" without growing the state space (preferred over moving to 8 headings).
Dense distance-to-goal shaping is rejected: it requires knowing the maze topology, contradicts the model-free premise of SARSA/Q-Learning, and risks changing the optimal policy unless done as potential-based shaping.

## 5. Discount factor $\gamma$

**$\gamma = 0.99$** (effective horizon $1/(1-\gamma) = 100$).

The maze size is left open in this project, so $\gamma$ is chosen to remain valid as the maze grows.
Floor rule of thumb: the goal's value should still be ≥10% of its on-goal value at the maze's diameter $D$, i.e. $\gamma^D \gtrsim 0.1$, giving $\gamma \gtrsim 0.1^{1/D}$. $\gamma = 0.99$ clears this for $D$ up to ~230 actions, comfortably more than any maze we expect.

Three forces pull on $\gamma$:

- **VI convergence speed.** Bellman backup contracts by $\gamma$. Iterations to tolerance $\epsilon$ scale as $\log\epsilon / \log\gamma$. At $\gamma=0.99$, ~700 sweeps to $10^{-3}$ — still milliseconds at our $|S|$.
- **RL sample complexity.** Tabular Q-Learning sample complexity scales as $|S||A| / (1-\gamma)^3$ in the worst case (in practice closer to linear in the horizon). $\gamma = 0.99$ is the largest value that keeps episode budgets reasonable.
- **Reward landscape.** With step cost $c$ and discount $\gamma$, the value at distance $k$ is $V^*(s_k) = -c (1-\gamma^k)/(1-\gamma)$, with local gradient $c \gamma^k$. Lower $\gamma$ flattens this gradient far from the goal — RL cannot distinguish nearby states through sample noise. Higher $\gamma$ keeps the gradient crisp at the cost of slower learning.

Drop to $\gamma = 0.95$ only if RL learning curves are too slow to fit the training budget on small mazes.

## 6. Episode setup

| Item                      | Choice                                                                  |
| ------------------------- | ----------------------------------------------------------------------- |
| Initial state $\mu_0$     | Fixed `(start_cell, start_heading)` for VI evaluation; uniform over free cells for RL training. Both configurable. |
| Goal                      | Single absorbing cell.                                                  |
| Episode termination       | Goal reached, **or** step limit $T_\text{max}$ hit.                     |
| Maze representation       | Cell-based: `cells: np.ndarray[R, C]` with `0=free, 1=wall, 2=goal`.    |
| Random seed               | Single `rng: np.random.Generator` threaded through simulator and RL.    |

Maze test fixtures (a 3×3, a 5×5 with corridor, a small loop) live in [src/maze_mdp/maze_mdp/simulator.py](../src/maze_mdp/maze_mdp/simulator.py) so unit tests have deterministic ground truth.

## 7. Algorithm hyperparameters

Defaults usable across maze sizes; tune only if learning curves misbehave.

| Parameter           | Value                                              | Notes                                                                                              |
| ------------------- | -------------------------------------------------- | -------------------------------------------------------------------------------------------------- |
| VI tolerance        | $\epsilon = 10^{-6}$                               | Sup-norm Bellman error. Tight enough to never flip an action max; loose enough to converge fast.   |
| Episode step cap    | $T_\text{max} = 10 \cdot \|S\|$                    | Safety net: ≥2× the expected return time of a random walk on the state graph.                      |
| Learning rate       | $\alpha_0 = 0.1$, decayed to $\alpha_\text{min} \approx 10^{-3}$ over training | Mirrors production deep-learning practice: start at 0.1 for fast initial progress, decay toward zero so updates shrink as $Q$ converges. Recommended schedule: per-(s,a) visit-count decay $\alpha_t = \alpha_0 / (1 + N(s,a)/\tau)$ with $\tau = 50$ — satisfies Robbins-Monro ($\sum\alpha=\infty,\sum\alpha^2<\infty$) for tabular convergence. Constant $\alpha = 0.1$ is acceptable only for early dev / sanity checks. |
| Exploration         | $\epsilon$-greedy with decay; **schedule TBD** (likely $\epsilon_0 = 1.0 \to \epsilon_\text{min} = 0.05$, exponential or visit-count-based decay) | Final schedule pending: depends on maze size and chosen episode budget. Constraint: $\epsilon_\text{min} > 0$ — under stochastic dynamics, zero exploration can leave parts of the Q-table unvisited. Decision deferred until first SARSA/Q-Learning runs are profiled on representative mazes. |
| Episodes per run    | $N_\text{episodes} \approx k \cdot \|S\| \|A\| / (1-\gamma)$ with $k \in [5, 20]$ | Sized from worst-case tabular-RL sample complexity. Maze size is unknown, so we keep this as a formula tied to $\|S\|$, $\|A\|$, and $\gamma$ rather than a fixed number. At $\gamma = 0.99$, $\|A\| = 3$: $N \approx 1500 \cdot \|S\|$ to $6000 \cdot \|S\|$. Run ≥5 seeds for error bars. Increase $k$ if learning curves haven't plateaued. |

## 8. Configuration

```python
@dataclass(frozen=True)
class MDPConfig:
    # transitions
    slip_prob: float = 0.1        # P(slip sideways | forward), per side
    turn_fail_prob: float = 0.0   # P(rotation fails | turn_*)
    # rewards
    forward_cost: float = 1.0
    turn_cost: float = 1.2
    bump_cost: float = 2.5
    goal_reward: float = 0.0
    # discounting
    gamma: float = 0.99
```

## 9. Out of scope (deferred)

- 8 headings / 8-connected grid (only if maze gains diagonal passages).
- `backward` action (no rear sensors on AlphaBot2).
- Continuous-pose state and function approximation (project is tabular).
- Distance-to-goal reward shaping (breaks model-free premise).
- Multi-goal policies / per-cell reward maps (not in spec).
- Empirical $\hat P$ from real-robot rollouts (Week-5 stretch goal; defaults are textbook).

## 10. References

- Russell, S. & Norvig, P. *Artificial Intelligence: A Modern Approach*, 4th ed., 2020 — Ch. 17 §17.1, Fig. 17.1: 4×3 gridworld MDP with 0.8 / 0.1 / 0.1 transition model.
- Sutton, R. S. & Barto, A. G. *Reinforcement Learning: An Introduction*, 2nd ed., 2018 — VI, SARSA, Q-Learning, tabular convergence.
- Ng, A. Y., Harada, D. & Russell, S. *Policy invariance under reward transformations: Theory and application to reward shaping*, ICML 1999, pp. 278–287 — potential-based shaping.

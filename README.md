### **`Value Iteration`** on FrozenLake 2×2 and 5×5
```markdown 
# Value Iteration on FrozenLake 2×2 and 5×5
A minimal, dependency-light implementation of **value iteration** — the
classic dynamic-programming *planning* algorithm from reinforcement
learning — on a hand-coded 2×2 and 5×5 FrozenLake grid.
Only Python + NumPy. No Gym, no Gymnasium.

> Part of `planningRL_methods_on_frozenlake_2x2_and_5x5`.
> Companion branches: `policy-iteration`, `truncated-policy-iteration`.

---

## Why "planning", not "learning"?

**Planning** assumes you already have a perfect model of the environment:
the reward function `R(s, a, s')` and the transition function
`T(s, a) → s'`. Given that model, value iteration computes the optimal
value function by repeatedly applying the **Bellman optimality operator**:
```
$V_{k+1}(s) = max_a [ R(s, a, s') + γ · V_k(s') ]$
```markdown

That is different from **learning**, where the model is unknown and values
have to be estimated from sampled experience (Q-learning, SARSA, PPO…).

---
```
## Algorithm

1. Initialise $V(s) = 0$ for every state $s ∈ S$.
2. Repeat until $max_s |V_new(s) − V(s)| < θ$:
   - for each state, compute $Q(s, a) = R(s, a, s') + γ·V(s')$ for every action\
   - set $V(s) = max_a Q(s, a)$
3. Extract the greedy policy: $π(s) = argmax_a Q(s, a)$.

Hyperparameters: $γ = 0.9$, convergence threshold $θ = 1e-10$.

### Elementwise form\

For each state `s`, one sweep performs:\
$V_{k+1}(s) = max_a [ Σ_r p(r|s,a)·r + γ · Σ_{s'} p(s'|s,a) · V_k(s') ]$\
└──────────────────── Q_k(s, a) ───────────────────────┘


In this code the model is deterministic, so the sums over `r` and `s'`
collapse to a single term — `T(s, a)` returns one next state and one
reward.

### Connection to policy iteration

Value iteration performs **one Bellman-*optimality* update per sweep**.
Policy iteration alternates **many Bellman-*expectation* updates** (policy
evaluation) with **one greedy update** (policy improvement). Value
iteration is the "cheap-per-step, many-steps" extreme; policy iteration is
the "expensive-per-step, few-steps" extreme. Truncated policy iteration
sits between the two.

---

## The 5×5 environment

![5×5 FrozenLake layout](images/frozenlake5x5_layout.png)

- **Goal** (red) : `s18 = (4, 3)` → reward `+1`
- **Pits** (blue) : `s7, s8, s13, s17, s19, s22` → reward `−10`
- **Bumping into a wall** → reward `−1` (agent stays in place)
- **Safe move** → reward `0`
- **Actions** : `↑ = 1`, `→ = 2`, `↓ = 3`, `← = 4`, `○ = 5` (stay)
- **Discount factor** : $γ = 0.9$

## The 2×2 environment
s2 = pit (−1)\
s4 = goal (+1)\
$γ = 0.9$\
s1 | s2 \
----+----\ 
s3 | s4 


Because `s4` is *not terminal* in this toy version, the agent keeps
collecting `+1` and `V(s4)` converges to `1 / (1 − γ) = 10` rather than
`1`. It's a useful first sanity check; the 5×5 grid is where the
algorithms get interesting.

---

## Results

![Optimal policy on the 5×5 grid](../main/images/frozenlake5x5_optimal_policy.png)

The optimal policy walks the agent toward `s18` while avoiding the pits.
Since each sweep is a single Bellman-optimality update, the number of
sweeps needed to reach the `1e-10` threshold is relatively large — but
each sweep costs almost nothing.

### Convergence

Because value iteration applies a **contraction** with modulus `γ`, the
error decays geometrically:\
$‖V_k − V‖_∞ ≤ γ^k · ‖V_0 − V‖_∞$


With $γ = 0.9$ and $θ = 1e-10$, the number of sweeps required is roughly\
$k ≈ log(θ) / log(γ) ≈ 23 / 0.105 ≈ 220$ sweep


which matches the observed count in practice (a few hundred sweeps,
depending on the starting $V_0$ and the tolerance).

---

## Run it

```bash
python valueIteration5x5.py
python valueIteration.py #for 2x2 environment

```
Requires only Python 3.8+ and NumPy.

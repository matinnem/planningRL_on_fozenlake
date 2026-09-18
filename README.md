

```markdown
# Truncated Policy Iteration on FrozenLake 2×2 and 5×5

A minimal, dependency-light implementation of **truncated policy
iteration** — the middle point between value iteration and full policy
iteration — on a hand-coded 5×5 FrozenLake grid (and a 2×2 toy version).
Only Python + NumPy.

> Part of `planningRL_methods_on_frozenlake_2x2_and_5x5`.
> Companion branches: `value-iteration`, `policy-iteration`.

---

## Why "planning", not "learning"?

**Planning** assumes you already have a perfect model of the environment:
`T(s, a) → s'` and `R(s, a, s')`. Given the model, you can compute the
optimal value function and policy *without interacting with the
environment*. This file sits squarely on the planning side.

---
```
## Algorithm

Truncated policy iteration follows the **same two-step loop** as policy
iteration:

1. **Policy evaluation.** Given $π_k$, iteratively compute $v_{π_k}$ from
   the Bellman equation\
   $v_{π_k} = r_{π_k} + γ · P_{π_k} · v_{π_k}$

   
2. **Policy improvement.**\
$π_{k+1} = arg max_π ( r_π + γ · P_π · v_{π_k} )$


The **only** difference from full policy iteration is that in step 1 the
inner loop is stopped after a *fixed* number of sweeps — here\
$j_{trunc} = 30$


instead of being run until $v_{π_k}$ converges to machine precision.\

### Where it sits in the family

| Algorithm                    | Inner sweeps per outer step   |\
|------------------------------|-----------------------------|\
| Value iteration              | 1 Bellman-optimality update |\
| Truncated policy iteration   | $j_{trunc} = 30$              |\
| Full policy iteration        | until convergence             |

- Value iteration at the leftmost extreme is the cheapest per outer step
  but needs the most outer steps.
- Full policy iteration at the rightmost extreme needs the fewest outer
  steps but pays for a fully-converged evaluation every time.
- **Truncated policy iteration** sits between them, and shows that a
  *perfectly* evaluated $v_{π_k}$ is not required for the improvement step
  to still produce the correct greedy policy. That is the conceptual seed
  for many modern actor–critic methods, where the critic is only ever
  partially trained between two actor updates.

### Elementwise form (identical to policy iteration)

**Policy evaluation** — in truncated form, `j` runs from `0` to $j_{trunc-1}$:\
$v_{π_k}^{(j+1)}(s) = Σ_a π_k(a|s) ·[ Σ_r p(r|s,a)·r+ γ · Σ_{s'} p(s'|s,a) · v_{π_k}^{(j)}(s') ]$ for all $s ∈ S$, j = 0, 1, ..., $j_{trunc − 1}$


**Policy improvement:**\
$π_{k+1}(s) = arg max_π Σ_a π(a|s) ·( Σ_r p(r|s,a)·r + γ · Σ_{s'} p(s'|s,a) · v_{π_k}(s') )$\
└────────────────── $q_{π_k}(s, a)$ ─────────────────┘


Let $a_{k(s)}$`*` = $argmax_{a}$ $q_{π_k}(s, a)$. Then the greedy policy is\
$π_{k+1}(a|s) = 1 if a == a_{k(s)}$`*`, else 0



---

## The 5×5 environment

![5×5 FrozenLake layout](../main/images/frozenlake5x5_layout.png)

- pits (blue) : `s7, s8, s13, s17, s19, s22` → reward `−10`
- goal (red)  : `s18` → reward `+1`
- bumping into a wall → `−1`
- safe move → `0`
- `γ = 0.9`

---

## Results

![Optimal policy on the 5×5 grid](../main/images/frozenlake5x5_optimal_policy.png)

Because each policy-evaluation step is capped at 30 sweeps, each outer
iteration is much cheaper than in full policy iteration — and yet the
optimal policy is still reached. In practice truncated policy iteration
often requires **fewer total Bellman updates** than either of its two
extremes.

---

## Run it

```bash
python truncatedPolicyIteration5x5.py
```
Requires only Python 3.8+ and NumPy.

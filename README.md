### **`policy-iteration`**
```markdown
# Policy Iteration on FrozenLake 2×2 and 5×5

A minimal, dependency-light implementation of **policy iteration** — the
other classical *planning* algorithm of reinforcement learning — on a
hand-coded 2×2 and 5×5 FrozenLake grid. Only Python + NumPy.

> Part of `planningRL_methods_on_frozenlake_2x2_and_5x5`.
> Companion branches: `value-iteration`, `truncated-policy-iteration`.

---

## Why "planning", not "learning"?

**Planning** assumes you already have a perfect model of the environment:
`T(s, a) → s'` and `R(s, a, s')`. Given the model, you can compute the
optimal value function and policy *without interacting with the
environment*.

That is different from **learning**, where the model is unknown and values
must be estimated from sampled experience (Q-learning, SARSA, PPO…).

---
```
## Algorithm

Policy iteration is *not* a direct solver of the Bellman optimality
equation. Instead it alternates between two steps:

1. **Policy evaluation.** Given a policy $π_k$, compute its state value
   $v_{π_k}$ by solving the Bellman equation:\
$v_{π_k} = r_{π_k} + γ · P_{π_k} · v_{π_k}$


where $r_{π_k}$ and $P_{π_k}$ come from the system model.

2. **Policy improvement.** Using $v_{π_k}$, produce a better policy:\
$π_{k+1} = arg max_π ( r_π + γ · P_π · v_{π_k} )$



Repeat until the value (or policy) stops changing.

### Elementwise form

**Policy evaluation** solves $v_{π_k} = r_{π_k} + γ·P_{π_k}·v_{π_k}$
iteratively, one state at a time:

$v_{π_k}^{(j+1)}(s) = Σ_a π_k(a|s) ·[ Σ_r p(r|s,a)·r + γ · Σ_{s'} p(s'|s,a) · v_{π_k}^{(j)}(s') ]$  for all s ∈ S, j = 0, 1, 2, ...


**Policy improvement** computes, for each state,\
$π_{k+1}(s) = arg max_π Σ_a π(a|s) · ( Σ_r p(r|s,a)·r + γ · Σ_{s'} p(s'|s,a) · v_{π_k}(s') )$\
└─────────────── $q_{π_k}(s, a)$ ──────────────────┘

Let $a_{k(s)}$`*` = $argmax_a$ $q_{π_k}(s, a)$. Then the greedy policy is\
$π_{k+1}(a|s) = 1$ if $a == a_{k(s)}$`*`, else 0


### Connection to value iteration

Value iteration performs **one Bellman-optimality update per sweep**.
Policy iteration performs **many Bellman-expectation updates** (the inner
evaluation) followed by **one greedy step**. Both converge to the same
optimum — policy iteration often in far fewer outer iterations, at the cost
of the inner evaluation loop.

The idea behind policy iteration is widely used in modern RL — for instance
in actor–critic methods, where the "critic" plays the role of the policy
evaluation step and the "actor" plays the role of the policy improvement
step.

---

## The 5×5 environment

![5×5 FrozenLake layout](../main/images/frozenlake5x5_layout.png)

- pits (blue) : `s7, s8, s13, s17, s19, s22` → reward `−10`
- goal (red)  : `s18` → reward `+1`
- bumping into a wall → `−1`
- safe move → `0`

## The 2×2 environment
s2 = pit (−1), s4 = goal (+1)\
s1 | s2\
----+----\
s3 | s4


---

## Results

![Optimal policy on the 5×5 grid](../main/images/frozenlake5x5_optimal_policy.png)

Starting from a trivial policy (stay everywhere), policy iteration converges
in very few **outer** iterations. Each outer iteration contains a *policy
evaluation* loop that runs to convergence, so the total cost is larger than
it looks — but the number of policy improvements (outer steps) is tiny
because each greedy update is much better informed.

---

## Run it

```bash
python policyIteration.py  # for 2x2 environment
python policyIteration5x5.py
```
Requires only Python 3.8+ and NumPy.


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

## Algorithm

Policy iteration is *not* a direct solver of the Bellman optimality
equation. Instead it alternates between two steps:

1. **Policy evaluation.** Given a policy `π_k`, compute its state value
   `v_{π_k}` by solving the Bellman equation:\
v_{π_k} = r_{π_k} + γ · P_{π_k} · v_{π_k}


where `r_{π_k}` and `P_{π_k}` come from the system model.

2. **Policy improvement.** Using `v_{π_k}`, produce a better policy:\
π_{k+1} = arg max_π ( r_π + γ · P_π · v_{π_k} )\



Repeat until the value (or policy) stops changing.

### Elementwise form

**Policy evaluation** solves `v_{π_k} = r_{π_k} + γ·P_{π_k}·v_{π_k}`
iteratively, one state at a time:\

v_{π_k}^{(j+1)}(s) = Σ_a π_k(a|s) ·
[ Σ_r p(r|s,a)·r

    γ · Σ_{s'} p(s'|s,a) · v_{π_k}^{(j)}(s') ]  for all s ∈ S, j = 0, 1, 2, ...


**Policy improvement** computes, for each state,



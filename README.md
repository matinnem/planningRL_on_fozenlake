
### 5.3 `README.md` on **`policy-iteration`**

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
   `v_{π_k}` by solving the Bellman equation:

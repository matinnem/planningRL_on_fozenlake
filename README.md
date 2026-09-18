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

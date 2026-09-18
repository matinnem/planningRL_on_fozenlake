"""
Common environment and helpers for 5x5 FrozenLake planning.
    - state_space : dict, 's1'..'s25' → [row, col] (1-indexed)
    - action_space: dict, arrow symbol → action code 1..5
    - reward      : R(state1, action, state2)
    - transition  : deterministic T(state1, action) → (state2, reward)
    - q           : one-step look-ahead Q value
    - key_of       : inverse lookup, [row, col] → 'sN'
    - render_box  : pretty-prints the 5x5 value/policy grid
"""
import numpy as np

# ---------------------------------------------------------------- environment
# 5x5 grid; every cell is a state 's1'..'s25' with (row, col) coordinates.
state_space = {'s1': [1,1], 's2': [1,2], 's3': [1,3], 's4': [1,4], 's5': [1,5],
               's6': [2,1], 's7': [2,2], 's8': [2,3], 's9': [2,4], 's10': [2,5],
               's11': [3,1], 's12': [3,2], 's13': [3,3], 's14': [3,4], 's15': [3,5],
               's16': [4,1], 's17': [4,2], 's18': [4,3], 's19': [4,4], 's20': [4,5],
               's21': [5,1], 's22': [5,2], 's23': [5,3], 's24': [5,4], 's25': [5,5]}

# Action codes: 1=Up, 2=Right, 3=Down, 4=Left, 5=Stay.
action_space = {"↑": 1, "→": 2, "↓": 3, "←": 4, "○": 5}
# Pits:  [2,2], [2,3], [3,3], [4,2], [4,4], [5,2]
# Goal:  [4,3]


def reward(state1, action1, state2):
    """Reward for the transition (state1, action1) → state2."""
    pits  = [[2,2], [2,3], [3,3], [4,2], [4,4], [5,2]]
    goals = [[4,3]]
    if state1 == state2:
        if action1 == 5:                        # stay action
            if state2 in pits:
                return -10                      # staying on a pit is still bad
            if state2 in goals:
                return 1                        # staying on the goal is good
            return 0                            # staying on a safe cell is neutral
        else:
            return -1                           # tried to move but bumped into a wall
    else:
        if state2 in pits:
            return -10                          # fell into a pit
        elif state2 in goals:
            return 1                            # reached the goal
        else:
            return 0                            # normal safe move


def transition(state1, action1):
    """
    Deterministic transition: from position `state1` and action `action1`,
    return (next_position, reward). Walls keep the agent in place; the grid
    is 5x5 and coordinates are 1-indexed.
    """
    state2 = [0, 0]
    if action1 == 5:
        state2 = state1                         # stay: no movement
    else:
        # -------- top edge: cannot go up ----------------------------------
        if state1[0] == 1:
            if action1 == 1:
                state2 = state1
            elif action1 == 2:
                state2 = state1 if state1[1] == 5 else [state1[0], state1[1] + 1]
            elif action1 == 4:
                state2 = state1 if state1[1] == 1 else [state1[0], state1[1] - 1]
            else:
                state2 = [state1[0] + 1, state1[1]]
        # -------- bottom edge: cannot go down -----------------------------
        elif state1[0] == 5:
            if action1 == 3:
                state2 = state1
            elif action1 == 2:
                state2 = state1 if state1[1] == 5 else [state1[0], state1[1] + 1]
            elif action1 == 4:
                state2 = state1 if state1[1] == 1 else [state1[0], state1[1] - 1]
            else:
                state2 = [state1[0] - 1, state1[1]]
        # -------- left edge: cannot go left -------------------------------
        elif state1[1] == 1:
            if action1 == 4:
                state2 = state1
            elif action1 == 1:
                state2 = [state1[0] - 1, state1[1]]
            elif action1 == 3:
                state2 = [state1[0] + 1, state1[1]]
            else:
                state2 = [state1[0], state1[1] + 1]
        # -------- right edge: cannot go right -----------------------------
        elif state1[1] == 5:
            if action1 == 2:
                state2 = state1
            elif action1 == 1:
                state2 = [state1[0] - 1, state1[1]]
            elif action1 == 3:
                state2 = [state1[0] + 1, state1[1]]
            else:
                state2 = [state1[0], state1[1] - 1]
        # -------- interior: normal move -----------------------------------
        else:
            if action1 == 1:
                state2 = [state1[0] - 1, state1[1]]
            elif action1 == 2:
                state2 = [state1[0], state1[1] + 1]
            elif action1 == 3:
                state2 = [state1[0] + 1, state1[1]]
            else:
                state2 = [state1[0], state1[1] - 1]
    return state2, reward(state1, action1, state2)


def q(state1, action1, v, gamma=0.9):
    """One-step look-ahead value: R + γ·V(s')."""
    state2, r = transition(state1, action1)
    return r + gamma * v


def key_of(val):
    """Inverse lookup: [row, col] → 'sN'. Returns None if not found."""
    for k, v in state_space.items():
        if v == val:
            return k
    return None


# ---------------------------------------------------------------- pretty print
def render_box(v_pi, pi, title=None):
    """Pretty-print the 5x5 grid as aligned boxes (cell width W = 11)."""
    W = 11
    if title:
        bar = "─" * max(1, 55 - len(title))
        print(f"\n  ── {title} {bar}")

    top = "  ╔" + ("═" * W + "╦") * 4 + "═" * W + "╗"
    mid = "  ╠" + ("═" * W + "╬") * 4 + "═" * W + "╣"
    bot = "  ╚" + ("═" * W + "╩") * 4 + "═" * W + "╝"

    inv   = {v: k for k, v in action_space.items()}   # code → arrow
    PITS  = {(2,2), (2,3), (3,3), (4,2), (4,4), (5,2)}
    GOALS = {(4,3)}

    def cell(row, col):
        s = f"s{row * 5 + col + 1}"
        v = float(v_pi[s])
        a = inv[int(pi[s])]
        # annotate pits / goal with a suffix in the name
        if   (row + 1, col + 1) in PITS:  s += "·P"
        elif (row + 1, col + 1) in GOALS: s += "·G"
        return (f"{s:^{W}}",
                f"{('v = ' + format(v, '>7.2f')):^{W}}",
                f"{a:^{W}}")

    grid = [[cell(r, c) for c in range(5)] for r in range(5)]

    print(top)
    for r, row in enumerate(grid):
        for line in range(3):
            print("  ║" + "║".join(row[c][line] for c in range(5)) + "║")
        if r < 4:
            print(mid)
    print(bot)



# ---------------------------------------------------------------- algorithm
# Truncated policy iteration is the same two-step loop as policy iteration,
# except policy evaluation is cut off after a FIXED number of inner sweeps
# (j_trunc = 30) instead of being run until convergence. This places it
# between value iteration (1 update per outer step) and full policy iteration
# (unbounded inner sweeps).
pi_s   = {i: 5 for i in state_space}
v_pi_s = {key: 0.0 for key in pi_s}
diff1  = 1000
k = 0
gamma = 0.9

print("═" * 60)
print("  Truncated Policy Iteration  •  FrozenLake 5x5  (Planning)")
print("═" * 60)
print(f"  Discount factor γ   : {gamma}")
print(f"  Convergence Δ       : 1e-10")
print(f"  Inner-sweep cap     : 30")
print(f"  Total iterations    : (pending)")

render_box(v_pi_s, pi_s, title=f"Initial guess  (k = {k})")

while diff1 >= 1e-10:
    # ---- (1) Truncated policy evaluation: exactly j_trunc inner sweeps -----
    v_pi_js = {key: 0.0 for key in pi_s}
    j = 0
    j_trunc = 30
    while j < j_trunc:
        for s_first in state_space:
            action = pi_s[s_first]
            num_s_first = state_space[s_first]
            num_s_next, r = transition(num_s_first, action)
            s_next = key_of(num_s_next)
            v_temp = r + gamma * v_pi_js[s_next]
            v_pi_js[s_first] = v_temp
        j += 1
    diff_vector1 = [np.abs(v_pi_s[s] - v_pi_js[s]) for s in state_space]
    v_pi_s = v_pi_js.copy()
    diff1  = np.max(diff_vector1)
    k += 1

    # ---- (2) Policy improvement (identical to full policy iteration) -------
    for s_first in state_space:
        q_values = []
        for sign, action in action_space.items():
            num_s_first = state_space[s_first]
            num_s_next, _ = transition(num_s_first, action)
            s_next = key_of(num_s_next)
            q_val = float(q(num_s_first, action, v_pi_s[s_next], gamma))
            q_values.append(q_val)
        a_max = np.argmax(q_values) + 1
        pi_s[s_first] = int(a_max)

    print(f"\n  ▸ Policy evaluation finished  (inner sweeps j = {j})")
    render_box(v_pi_s, pi_s, title=f"After policy improvement  (k = {k})")

print()
print("═" * 60)
print(f"  Converged after {k} truncated-policy-iteration steps")
print("═" * 60)
render_box(v_pi_s, pi_s, title="Final optimal policy & state values")
print()

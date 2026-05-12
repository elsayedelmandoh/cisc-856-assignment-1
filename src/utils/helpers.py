"""helper functions for value iteration and policy iteration algorithms, plus visualization utilities."""

import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8-whitegrid')
from pathlib import Path

from .actions import Action

# solver 2a: value iteration
def value_iteration(gw, discount, tolerance=0.1):
    """
    perform value iteration to compute optimal state values
    bellman optimality: V*(s) = max_a Σ_{s'} P(s'|s,a) [R(s') + γ V*(s')]
    
    Args:
        gw (GridWorld): The GridWorld environment to solve.
        discount (float): Discount factor for future rewards (typically 0.75 or 0.95).
        tolerance (float): Convergence threshold; iteration stops when max value change < tolerance.
    
    Returns:
        int: Number of iterations performed before convergence.
    """
    iteration = 0
    convergence_history = []

    while True:
        gw.create_next_values()
        delta = 0.0

        for state in gw.get_states():
            old_val = gw.get_value(state)
            best_value = float('-inf')

            for action in gw.get_actions(state):
                action_value = 0.0
                for t in gw.get_transitions(state, action):
                    ns, prob = t["state"], t["prob"]
                    r = gw.get_reward(ns)
                    future = 0.0 if gw.is_terminal(ns) else gw.get_value(ns)
                    action_value += prob * (r + discount * future)

                best_value = max(best_value, action_value)

            gw.set_value(state, best_value)
            delta = max(delta, abs(best_value - old_val))

        gw.set_next_values()
        iteration += 1
        convergence_history.append(delta)

        if delta < tolerance:
            break

    return iteration, convergence_history


# solver 2b: policy iteration
def policy_iteration(gw, discount, tolerance=0.1):
    """
    alternates between:
        1. policy evaluation  - iterate V under fixed policy until convergence
        2. policy improvement - greedy update of policy using current V
    stops when policy is stable (no changes in improvement step)

    perform policy iteration to compute optimal state values and policy.
    
    alternates between policy evaluation (computing state values for current policy) 
    and policy improvement (updating policy to greedy actions) until convergence.
    
    args:
        gw (GridWorld): the gridWorld environment to solve.
        discount (float): discount factor for future rewards (typically 0.75 or 0.95).
        tolerance (float): convergence threshold for value updates within policy evaluation.
    
    returns:
        tuple: (iterations: int, policy: dict) where policy maps state indices to Action objects.
    """
    states = gw.get_states()
    policy = {s: Action.up for s in states} 
    outer_iteration = 0

    while True:
        while True:
            gw.create_next_values()
            delta = 0.0

            for state in states:
                old_val = gw.get_value(state)
                action = policy[state]
                value = 0.0
                for t in gw.get_transitions(state, action):
                    ns, prob = t["state"], t["prob"]
                    r = gw.get_reward(ns)
                    future = 0.0 if gw.is_terminal(ns) else gw.get_value(ns)
                    value += prob * (r + discount * future)

                gw.set_value(state, value)
                delta = max(delta, abs(value - old_val))

            gw.set_next_values()
            if delta < tolerance:
                break

        policy_stable = True
        for state in states:
            old_action = policy[state]
            best_action, best_val = None, float('-inf')

            for action in gw.get_actions(state):
                val = 0.0
                for t in gw.get_transitions(state, action):
                    ns, prob = t["state"], t["prob"]
                    r = gw.get_reward(ns)
                    future = 0.0 if gw.is_terminal(ns) else gw.get_value(ns)
                    val += prob * (r + discount * future)

                if val > best_val:
                    best_val, best_action = val, action

            policy[state] = best_action
            if best_action != old_action:
                policy_stable = False

        outer_iteration += 1
        if policy_stable:
            break

    return outer_iteration, policy


def plot_values(gw, title="state values"):
    """heatmap of state values with annotations."""
    grid = np.zeros((gw._height, gw._width))
    mask = np.zeros((gw._height, gw._width), dtype=bool)

    for r in range(gw._height):
        for c in range(gw._width):
            cell = r * gw._width + c
            if cell in gw._blocked_cells:
                mask[r, c] = True
            elif cell == gw._goal_cell:
                grid[r, c] = gw._goal_value
            elif cell in gw._danger_cells:
                grid[r, c] = gw._danger_value
            else:
                grid[r, c] = gw.get_value(cell)

    fig, ax = plt.subplots(figsize=(6, 6))
    cmap = plt.cm.RdYlGn
    im = ax.imshow(grid, cmap=cmap, vmin=-6, vmax=6)

    for r in range(gw._height):
        for c in range(gw._width):
            cell = r * gw._width + c
            if cell in gw._blocked_cells:
                ax.add_patch(plt.Rectangle((c-0.5, r-0.5), 1, 1, color='black'))
            elif cell == gw._goal_cell:
                ax.text(c, r, "G\n+5.0", ha='center', va='center', fontsize=10, color='white', fontweight='bold')
            elif cell in gw._danger_cells:
                ax.text(c, r, f"F\n-5.0", ha='center', va='center', fontsize=10, color='white')
            else:
                ax.text(c, r, f"{gw.get_value(cell):.2f}", ha='center', va='center', fontsize=9)

    plt.colorbar(im, ax=ax)
    ax.set_title(title)
    ax.set_xticks(range(gw._width))
    ax.set_yticks(range(gw._height))
    plt.tight_layout()
    
    # ensure results directory exists
    Path("docs/02-results").mkdir(parents=True, exist_ok=True)
    plt.savefig(f"docs/02-results/{title.replace(' ', '_')}.png", dpi=150)
    plt.show()


def plot_policy(gw, policy, title="optimal policy"):
    """arrow visualization of the extracted policy."""
    arrow_map = {
        Action.up: (0, -0.3),
        Action.down: (0, 0.3),
        Action.left: (-0.3, 0),
        Action.right: (0.3, 0),
    }

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(-0.5, gw._width - 0.5)
    ax.set_ylim(-0.5, gw._height - 0.5)
    ax.set_aspect('equal')
    ax.invert_yaxis()

    for r in range(gw._height):
        for c in range(gw._width):
            cell = r * gw._width + c
            if cell in gw._blocked_cells:
                ax.add_patch(plt.Rectangle((c-0.5, r-0.5), 1, 1, color='black'))
            elif cell == gw._goal_cell:
                ax.add_patch(plt.Rectangle((c-0.5, r-0.5), 1, 1, color='green', alpha=0.5))
                ax.text(c, r, "G", ha='center', va='center', fontsize=12, color='white', fontweight='bold')
            elif cell in gw._danger_cells:
                ax.add_patch(plt.Rectangle((c-0.5, r-0.5), 1, 1, color='red', alpha=0.5))
                ax.text(c, r, "F", ha='center', va='center', fontsize=12, color='white')
            else:
                if cell in policy:
                    dx, dy = arrow_map[policy[cell]]
                    ax.annotate("", xy=(c+dx, r+dy), xytext=(c, r),
                                arrowprops=dict(arrowstyle="->", color='steelblue', lw=2))

    ax.set_xticks(range(gw._width))
    ax.set_yticks(range(gw._height))
    ax.grid(True, alpha=0.3)
    ax.set_title(title)
    plt.tight_layout()
    
    # ensure results directory exists
    Path("docs/02-results").mkdir(parents=True, exist_ok=True)
    plt.savefig(f"docs/02-results/{title.replace(' ', '_')}.png", dpi=150)
    plt.show()

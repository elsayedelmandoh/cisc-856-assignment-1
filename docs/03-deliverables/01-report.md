# implementing and analyzing a basic rl algorithm

elsayed elmandouh - 20596379 - cisc 856 assignment 1 - reinforcement learning

## 1. objective

this report documents the implementation and analysis of a 5x5 gridworld mdp solved using three approaches: linear system solving, value iteration, and policy iteration.

## 2. environment

### 2.1 environment parameters

| parameter | value |
|-----------|-------|
| grid size | 5x5 (25 cells total) |
| goal state | cell 14 (reward +5.0) |
| danger states | cells 2, 18, 21 (reward -5.0) |
| blocked cells | cells 6, 7, 11, 12 |
| valid states | 17 |
| actions | up, right, down, left |

### 2.2 mdp components

**state space**: cells 0-24 (row-major), excluding blocked cells 6,7,11,12.

**action space**: four moves; hitting wall/blocked = stay in place.

**transition model**: probability (1-noise) moves intended direction; probability noise moves random.

**reward function**: goal +5.0 (terminal), danger -5.0 (terminal), others 0.0.

## 3. implementation

### 3.1 gridworld class

the `gridworld` class implements: `__init__`, `_state_from_action`, `is_terminal`, `get_states`, `get_actions`, `get_reward`, `get_transitions`, `solve_linear_system`.

### 3.2 key details

- state indexing: row-major
- noise=0.0: deterministic; noise=0.2: 0.8 intended, 0.067 each other

## 4. methods

### 4.1 linear system solver

v = r + γ × p × v → ax = b where a = i - γp/4, b = expected rewards. solved via `np.linalg.solve()`.

### 4.2 value iteration

v(s) ← max_a Σ p(s'|s,a)[r(s') + γv(s')]. iterate until delta < tolerance.

### 4.3 policy iteration

alternates: policy evaluation (v = r + γpv) and policy improvement (π(s) = argmax_a q(s,a)).

## 5. results

### 5.1 linear system solver

**γ = 0.95:**
```
 -2.69  -3.60  -5.00  -1.64  -0.00
 -2.35   ----   ----  -0.00   1.64
 -2.51   ----   ----  -0.00   goal
 -3.19  -3.91  -4.10  -5.00  -0.57
 -3.82  -5.00  -3.99  -3.45  -1.82
```

**γ = 0.75:**
```
 -0.78  -2.24  -5.00  -1.54  -0.00
 -0.38   ----   ----  -0.00   1.54
 -0.48   ----   ----  -0.00   goal
 -1.22  -2.45  -2.73  -5.00  -0.17
 -2.37  -5.00  -2.71  -2.34  -0.75
```

### 5.2 value iteration

| configuration | iterations |
|--------------|------------|
| γ=0.95, noise=0.0 | 12 |
| γ=0.75, noise=0.0 | 12 |
| γ=0.95, noise=0.2 | 15 |
| γ=0.75, noise=0.2 | 7 |

**γ=0.95, noise=0.0:**
```
  3.15   2.99  -5.00   4.51   4.75
  3.32   ----   ----   4.75   5.00
  3.49   ----   ----   5.00   goal
  3.68   3.87   4.07  -5.00   5.00
  3.49  -5.00   4.29   4.51   4.75
```

![value iteration γ=0.95 noise=0.0](docs/02-results/vi_gamma0.95_noise0.0.png)

**γ=0.95, noise=0.2:**
```
  0.42  -0.10  -5.00   3.60   4.51
  0.56   ----   ----   4.49   4.88
  0.65   ----   ----   4.22   goal
  0.72   0.83   1.40  -5.00   4.17
  0.23  -5.00   2.09   2.90   3.84
```

![value iteration γ=0.95 noise=0.2](docs/02-results/vi_gamma0.95_noise0.2.png)

### 5.3 policy iteration

**γ=0.95, noise=0.0 (10 iterations):**
```
  3.15   2.99  -5.00   4.51   4.75
  3.32   ----   ----   4.75   5.00
  3.49   ----   ----   5.00   goal
  3.68   3.87   4.07  -5.00   5.00
  3.49  -5.00   4.29   4.51   4.75
```

![policy iteration](docs/02-results/pi_gamma0.95_noise0.0.png)
![optimal policy](docs/02-results/optimal_policy_gamma0.95.png)

### 5.4 optimal policy map

| state | action | rationale |
|-------|--------|-----------|
| 0 | down | toward goal |
| 1 | left | avoid blocked |
| 3,8,13 | right | approach goal |
| 4,9,19,24 | down/up | reach goal |

## 6. convergence analysis

| algorithm | config | iterations |
|-----------|--------|------------|
| linear solver | γ=0.95 | 1 |
| linear solver | γ=0.75 | 1 |
| value iteration | γ=0.95, noise=0.0 | 12 |
| value iteration | γ=0.75, noise=0.0 | 12 |
| value iteration | γ=0.95, noise=0.2 | 15 |
| value iteration | γ=0.75, noise=0.2 | 7 |
| policy iteration | γ=0.95, noise=0.0 | 10 |

**factors affecting convergence:**
- discount factor γ: higher γ → more iterations
- noise: increases uncertainty, more iterations
- grid topology: blocked cells affect value propagation

## 7. algorithm comparison

**linear solver**: closed-form, deterministic only, random policy baseline.

**value iteration**: simple, works for stochastic, max over actions each update.

**policy iteration**: fewer outer iterations, explicit policy at each step.

all three produce correct results; vi and pi reach identical optimal policy.

![linear solver γ=0.95](docs/02-results/linear_solver_gamma0.95_noise0.0.png)


**references**: sutton & barto (2018) reinforcement learning: an introduction - https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf
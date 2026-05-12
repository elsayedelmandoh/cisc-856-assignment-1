import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8-whitegrid')
from .actions import Action, action_to_offset


class GridWorld:

    def __init__(self, height, width, goal, goal_value=5.0, danger=[], danger_value=-5.0, blocked=[], noise=0.0):
        """
        Initialize the GridWorld environment.
        Creates a gridworld like MDP
         - height (int): Number of rows
         - width (int): Number of columns
         - goal (int): Index number of goal cell
         - goal_value (float): Reward given for goal cell
         - danger (list of int): Indices of cells marked as danger
         - danger_value (float): Reward given for danger cell
         - blocked (list of int): Indices of cells marked as blocked (can't enter)
         - noise (float): probability of resulting state not being what was expected
        """

        self._width = width
        self._height = height
        self._grid_values = [0 for _ in range(height * width)] # Initialize state values.
        self._goal_value = goal_value
        self._danger_value = danger_value
        self._goal_cell = goal
        self._danger_cells = danger
        self._blocked_cells = blocked
        self._noise = noise # Noise level in the environment.
        assert noise >= 0 and noise < 1 # Ensure valid noise value.
        self.create_next_values() # Initialize the next state values.


    def reset(self):
        """
        Reset the state values to their initial state.
        """
        self._grid_values = [0 for _ in range(self._height * self._width)]
        self.create_next_values()


    def _inbounds(self, state):
        """
        Check if a state index is within the grid boundaries.
        """
        return state >= 0 and state < self._width * self._height

    def _inbounds_rc(self, state_r, state_c):
        """
        Check if row and column indices are within the grid boundaries.
        """
        return state_r >= 0 and state_r < self._height and state_c >= 0 and state_c < self._width

    def _state_to_rc(self, state):
        """
        Convert a state index to row and column indices.
        """
        return state // self._width, state % self._width

    def _state_from_action(self, state, action):
        """
        returns the resulting state after taking action from state
        if the move goes out of bounds or into a blocked cell, the agent stays
        """
        r, c = self._state_to_rc(state)
        dr, dc = action_to_offset[action]
        nr, nc = r + dr, c + dc

        if not self._inbounds_rc(nr, nc):
            return state
        
        new_state = nr * self._width + nc
        
        if new_state in self._blocked_cells:
            return state
        
        return new_state

    def is_terminal(self, state):
        """
        Returns true if a state is terminal (goal, or danger)
        """
        return state == self._goal_cell or state in self._danger_cells

    def get_states(self):
        """
        gets all non-terminal non-blocked states — where bellman updates happen
        """
        return [
        s for s in range(self._height * self._width)
        if s not in self._blocked_cells and not self.is_terminal(s)
        ]

    def get_actions(self, state):
        """
        returns a list of valid actions given the current state
        all 4 actions available everywhere; transition model handles walls
        """
        return list(Action) 


    def get_reward(self, state):
        """
        get the reward for being in the current state
        """
        assert self._inbounds(state)
        # Reward is non-zero for danger or goal
        if state == self._goal_cell:
            return self._goal_value
        elif state in self._danger_cells:
            return self._danger_value
        return 0.0


    def get_transitions(self, state, action):
        """
        get a list of transitions as a result of attempting the action in the current state
        each item in the list is a dictionary, containing the probability of reaching that state and the state itself
        stochastic transition model
        returns list of {state, prob} dicts
        intended direction: prob = 1 - noise
        other 3 directions: prob = noise / 3 each
        probabilities accumulate for states reachable via multiple actions
        """
        transitions = {}
        
        if self._noise == 0.0:
            next_state = self._state_from_action(state, action)
            transitions[next_state] = transitions.get(next_state, 0) + 1.0
        
        else:
            intended = self._state_from_action(state, action)
            transitions[intended] = transitions.get(intended, 0) + (1 - self._noise)
        
            other_actions = [a for a in Action if a != action]
            noise_per = self._noise / len(other_actions)
            for other_action in other_actions:
                ns = self._state_from_action(state, other_action)
                transitions[ns] = transitions.get(ns, 0) + noise_per

        return [{"state": s, "prob": p} for s, p in transitions.items()]


    def get_value(self, state):
        """
        Get the current value of the state
        """
        assert self._inbounds(state)
        return self._grid_values[state]

    def create_next_values(self):
        """
        creates a temporary storage for state value updating
        if this is not used, then asynchronous updating may result in unexpected results
        to use properly, run this at the start of each iteration
        snapshot current values; updates go here until set_next_values() commits them
        """
        self._next_values = list(self._grid_values)

    def set_next_values(self):
        """
        set the state values from the temporary copied values
        to use properly, run this at the end of each iteration
        commit the buffered updates, synchronous bellman sweep
        """
        self._grid_values = list(self._next_values)

    def set_value(self, state, value):
        """
        set the value of the state into the temporary copy
        this value will not update into main storage until self.set_next_values() is called
        write to the buffer (not live until set_next_values is called
        """
        assert self._inbounds(state)
        self._next_values[state] = value

    def solve_linear_system(self, discount_factor=1.0):
        """
        solve the gridworld using a system of linear equations
        :param discount_factor: The discount factor for future rewards
        solves V = R + γ * P * V  as a linear system Ax = b
        under a uniform random policy (prob 1/4 per action)
        only valid for deterministic environments (noise=0)
        """
        states = self.get_states()
        n = len(states)
        state_to_idx = {s: i for i, s in enumerate(states)}

        A = np.zeros((n, n))
        b = np.zeros(n)

        for i, state in enumerate(states):
            actions = self.get_actions(state)
            num_actions = len(actions)
            
            A[i][i] = 1.0  
            
            for action in actions:
                transitions = self.get_transitions(state, action)
                for t in transitions:
                    ns, prob = t["state"], t["prob"]
                    weight = (1.0 / num_actions) * prob  
                    
                    b[i] += weight * self.get_reward(ns)
                    
                    if not self.is_terminal(ns) and ns in state_to_idx:
                        A[i][state_to_idx[ns]] -= discount_factor * weight

        x = np.linalg.solve(A, b)

        for i, state in enumerate(states):
            self._grid_values[state] = x[i]

        return self


    def __str__(self):
        """
        Pretty print the state values
        """
        out_str = ""
        for r in range(self._height):
            for c in range(self._width):
                cell = r * self._width + c
                if cell in self._blocked_cells:
                    out_str += "{:>6}".format("----")
                elif cell == self._goal_cell:
                    out_str += "{:>6}".format("GOAL")
                elif cell in self._danger_cells:
                    out_str += "{:>6.2f}".format(self._danger_value)
                else:
                    out_str += "{:>6.2f}".format(self._grid_values[cell])
                out_str += " "
            out_str += "\n"
        return out_str


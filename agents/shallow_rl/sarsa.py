"""
SARSA Agent

Implements tabular SARSA (on-policy) for the relationship dynamics environment.
Uses discrete state space (discretized emotion/trust/conflict).
"""

from typing import Dict, Tuple
import numpy as np
from collections import defaultdict

from personality.personality_policy import PersonalityPolicy, PersonalityType
from environment.action_feasibility import ActionFeasibility


class SarsaAgent:
    """
    SARSA agent for relationship dynamics simulator.

    Uses tabular SARSA (on-policy TD learning) with discretized state space.
    Similar to Q-learning but uses the actual next action taken (on-policy)
    instead of the maximum Q-value.
    """

    def __init__(
        self,
        num_actions: int = 10,
        state_bins: int = 5,
        learning_rate: float = 0.1,
        discount_factor: float = 0.95,
        epsilon: float = 0.1,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01,
        personality: PersonalityType = PersonalityType.NEUTRAL,
    ):
        """
        Initialize SARSA agent.

        Args:
            num_actions: Number of discrete actions (default: 10)
            state_bins: Number of bins per state dimension for discretization (default: 5)
            learning_rate: SARSA learning rate (default: 0.1)
            discount_factor: Discount factor gamma (default: 0.95)
            epsilon: Initial exploration rate (default: 0.1)
            epsilon_decay: Epsilon decay rate per episode (default: 0.995)
            epsilon_min: Minimum epsilon value (default: 0.01)
            personality: Personality type for this agent
        """
        self.num_actions = num_actions
        self.state_bins = state_bins
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        # Initialize Q-table: state -> action -> Q-value
        # State is now (emotion_bin, trust_bin, conflict_bin, calmness_bin)
        self.q_table: Dict[Tuple[int, int, int, int], np.ndarray] = defaultdict(
            lambda: np.zeros(num_actions)
        )

        # Personality policy
        self.personality = PersonalityPolicy(personality)

        # Action feasibility (for calmness-based action selection)
        self.action_feasibility = ActionFeasibility()

        # Training statistics
        self.training_stats = {
            "episodes": 0,
            "total_reward": 0.0,
            "action_counts": np.zeros(num_actions),
        }

    def discretize_state(self, state: np.ndarray) -> Tuple[int, int, int, int]:
        """
        Discretize continuous state into discrete bins.
        State now includes calmness: [emotion, trust, conflict, calmness]

        Args:
            state: Continuous state vector [emotion, trust, conflict, calmness]

        Returns:
            Discretized state tuple (emotion_bin, trust_bin, conflict_bin, calmness_bin)
        """
        # Apply personality-based state perception (only to first 3 dimensions)
        perceived_state = self.personality.modify_state_perception(state[:3])

        # Discretize each dimension
        emotion, trust, conflict = perceived_state[:3]
        calmness = state[3] if len(state) > 3 else 0.5

        emotion_bin = np.digitize(emotion, np.linspace(-1, 1, self.state_bins)) - 1
        emotion_bin = np.clip(emotion_bin, 0, self.state_bins - 1)

        trust_bin = np.digitize(trust, np.linspace(0, 1, self.state_bins)) - 1
        trust_bin = np.clip(trust_bin, 0, self.state_bins - 1)

        conflict_bin = np.digitize(conflict, np.linspace(0, 1, self.state_bins)) - 1
        conflict_bin = np.clip(conflict_bin, 0, self.state_bins - 1)

        calmness_bin = np.digitize(calmness, np.linspace(0, 1, self.state_bins)) - 1
        calmness_bin = np.clip(calmness_bin, 0, self.state_bins - 1)

        return (emotion_bin, trust_bin, conflict_bin, calmness_bin)

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        Select action using epsilon-greedy policy with personality bias and calmness feasibility.

        Args:
            state: Current state vector [emotion, trust, conflict, calmness]
            training: Whether in training mode (affects exploration)

        Returns:
            Selected action index
        """
        discrete_state = self.discretize_state(state)
        q_values = self.q_table[discrete_state].copy()

        # Apply personality-based action bias
        for action_idx in range(self.num_actions):
            q_values[action_idx] += self.personality.get_action_bias(action_idx)

        # Get calmness from state
        calmness = state[3] if len(state) > 3 else 0.5

        # Apply action feasibility based on calmness
        q_values = self.action_feasibility.modify_q_values(q_values, calmness)

        # Epsilon-greedy exploration
        if training and np.random.random() < self.epsilon:
            action = np.random.randint(self.num_actions)
        else:
            # Break ties randomly
            action = np.random.choice(np.where(q_values == q_values.max())[0])

        return action

    def update(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        next_action: int,
        done: bool,
    ):
        """
        Update Q-table using SARSA update rule (on-policy).

        Q(s, a) = Q(s, a) + α * [r + γ * Q(s', a') - Q(s, a)]

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state after action
            next_action: Next action to be taken (on-policy)
            done: Whether episode terminated
        """
        discrete_state = self.discretize_state(state)
        discrete_next_state = self.discretize_state(next_state)

        # Current Q-value
        current_q = self.q_table[discrete_state][action]

        # Next state Q-value (for the actual next action, not max)
        if done:
            next_q = 0.0
        else:
            next_q = self.q_table[discrete_next_state][next_action]
            # Apply personality bias
            next_q += self.personality.get_action_bias(next_action)

        # SARSA update
        target = reward + self.discount_factor * next_q
        self.q_table[discrete_state][action] += self.learning_rate * (
            target - current_q
        )

        # Update statistics
        self.training_stats["action_counts"][action] += 1

    def decay_epsilon(self):
        """Decay exploration rate after episode."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def get_policy(self) -> Dict[Tuple[int, int, int, int], int]:
        """
        Get deterministic policy from Q-table.

        Returns:
            Dictionary mapping states to greedy actions
        """
        policy = {}
        for state, q_values in self.q_table.items():
            # Apply personality bias
            biased_q = q_values.copy()
            for action_idx in range(self.num_actions):
                biased_q[action_idx] += self.personality.get_action_bias(action_idx)

            # Apply feasibility (approximate calmness from state bin)
            # calmness_bin is the 4th element of state tuple
            calmness = (state[3] + 0.5) / self.state_bins  # Approximate calmness value
            biased_q = self.action_feasibility.modify_q_values(biased_q, calmness)

            policy[state] = np.argmax(biased_q)
        return policy

    def save(self, filepath: str):
        """Save Q-table to file."""
        import pickle

        with open(filepath, "wb") as f:
            pickle.dump(
                {
                    "q_table": dict(self.q_table),
                    "epsilon": self.epsilon,
                    "training_stats": self.training_stats,
                },
                f,
            )

    def load(self, filepath: str):
        """Load Q-table from file."""
        import pickle

        with open(filepath, "rb") as f:
            data = pickle.load(f)
            self.q_table = defaultdict(
                lambda: np.zeros(self.num_actions), data["q_table"]
            )
            self.epsilon = data["epsilon"]
            self.training_stats = data["training_stats"]

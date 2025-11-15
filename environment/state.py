"""
State Space Definition

Defines the relationship state representation, including:
1. Core relationship metrics (emotion, trust, conflict)
2. Internal states (calmness for each agent)
3. Dialogue history for deep RL agents
"""

from typing import List
import numpy as np


class RelationshipState:
    """
    Represents the state of the relationship communication environment.

    Core State Components:
    - emotion_level: Current emotional valence [-1, 1] (negative to positive)
    - trust_level: Relationship trust level [0, 1]
    - conflict_intensity: Intensity of current conflict [0, 1]

    Internal States (per agent):
    - calmness: Self-regulation level [0, 1] (dynamic, changes during dialogue)
      Lower → more prone to negative actions (BLAME, DEFENSIVE, WITHDRAW)
      Higher → more capable of positive actions (APOLOGIZE, EMPATHIZE)
    - irritability: Trait reactivity [0, 1] (fixed personality trait)
      Higher → calmness drops more from negative events, recovers slower

    Dialogue History (for Deep RL):
    - action_history: Sequence of recent actions taken
    - history_length: Maximum length of history to maintain
    """

    def __init__(
        self,
        emotion_level: float = 0.0,
        trust_level: float = 0.5,
        conflict_intensity: float = 0.5,
        calmness_a: float = 0.4,
        calmness_b: float = 0.4,
        irritability_a: float = 0.4,
        irritability_b: float = 0.4,
        history_length: int = 10,
    ):
        """
        Initialize relationship state.

        Args:
            emotion_level: Initial emotion level (default: 0.0, neutral)
            trust_level: Initial trust level (default: 0.5, moderate)
            conflict_intensity: Initial conflict intensity (default: 0.5, moderate)
            calmness_a: Initial calmness for agent A (default: 0.4, moderate)
            calmness_b: Initial calmness for agent B (default: 0.4, moderate)
            irritability_a: Irritability trait for agent A (default: 0.4, moderate)
            irritability_b: Irritability trait for agent B (default: 0.4, moderate)
            history_length: Maximum length of action history to maintain
        """
        # Core relationship metrics
        self.emotion_level = np.clip(emotion_level, -1.0, 1.0)
        self.trust_level = np.clip(trust_level, 0.0, 1.0)
        self.conflict_intensity = np.clip(conflict_intensity, 0.0, 1.0)

        # Internal states (calmness is dynamic, irritability is fixed)
        self.calmness_a = np.clip(calmness_a, 0.0, 1.0)
        self.calmness_b = np.clip(calmness_b, 0.0, 1.0)
        self.irritability_a = np.clip(irritability_a, 0.0, 1.0)
        self.irritability_b = np.clip(irritability_b, 0.0, 1.0)

        # Dialogue history
        self.history_length = history_length
        self.action_history: List[int] = []  # Store action indices

    def get_core_state(self) -> np.ndarray:
        """
        Get core state vector for shallow RL (tabular methods).

        Returns:
            numpy array of shape (3,) containing [emotion, trust, conflict]
        """
        return np.array(
            [self.emotion_level, self.trust_level, self.conflict_intensity],
            dtype=np.float32,
        )

    def get_core_state_with_calmness(self, agent_id: int = 0) -> np.ndarray:
        """
        Get core state vector including calmness for the specified agent.

        Args:
            agent_id: Agent ID (0 for A, 1 for B)

        Returns:
            numpy array of shape (4,) containing [emotion, trust, conflict, calmness]
        """
        calmness = self.calmness_a if agent_id == 0 else self.calmness_b
        return np.array(
            [self.emotion_level, self.trust_level, self.conflict_intensity, calmness],
            dtype=np.float32,
        )

    def get_full_state(self, agent_id: int = 0) -> np.ndarray:
        """
        Get full state vector including dialogue history and calmness for deep RL.

        Args:
            agent_id: Agent ID (0 for A, 1 for B)

        Returns:
            numpy array of shape (4 + history_length,) containing
            core state + calmness + action history
        """
        # Core state with calmness
        core = self.get_core_state_with_calmness(agent_id)

        # Encode action history (one-hot or padding)
        # If history is shorter than history_length, pad with -1
        history_encoding = np.full(self.history_length, -1, dtype=np.float32)
        if self.action_history:
            recent_history = self.action_history[-self.history_length :]
            history_encoding[: len(recent_history)] = recent_history

        return np.concatenate([core, history_encoding])

    def add_action(self, action_idx: int):
        """
        Add an action to the dialogue history.

        Args:
            action_idx: Index of the action taken
        """
        self.action_history.append(action_idx)
        # Keep only recent history
        if len(self.action_history) > self.history_length:
            self.action_history = self.action_history[-self.history_length :]

    def get_discretized_state(self, bins: int = 5, agent_id: int = 0) -> tuple:
        """
        Discretize state for tabular RL methods.

        Args:
            bins: Number of bins per dimension (default: 5)
            agent_id: Agent ID (0 for A, 1 for B)

        Returns:
            Tuple of discretized indices (emotion_bin, trust_bin, conflict_bin, calmness_bin)
        """
        # Discretize each dimension into bins
        emotion_bin = np.digitize(self.emotion_level, np.linspace(-1, 1, bins)) - 1
        emotion_bin = np.clip(emotion_bin, 0, bins - 1)

        trust_bin = np.digitize(self.trust_level, np.linspace(0, 1, bins)) - 1
        trust_bin = np.clip(trust_bin, 0, bins - 1)

        conflict_bin = np.digitize(self.conflict_intensity, np.linspace(0, 1, bins)) - 1
        conflict_bin = np.clip(conflict_bin, 0, bins - 1)

        calmness = self.calmness_a if agent_id == 0 else self.calmness_b
        calmness_bin = np.digitize(calmness, np.linspace(0, 1, bins)) - 1
        calmness_bin = np.clip(calmness_bin, 0, bins - 1)

        return (emotion_bin, trust_bin, conflict_bin, calmness_bin)

    def copy(self) -> "RelationshipState":
        """Create a deep copy of the state."""
        new_state = RelationshipState(
            emotion_level=self.emotion_level,
            trust_level=self.trust_level,
            conflict_intensity=self.conflict_intensity,
            calmness_a=self.calmness_a,
            calmness_b=self.calmness_b,
            irritability_a=self.irritability_a,
            irritability_b=self.irritability_b,
            history_length=self.history_length,
        )
        new_state.action_history = self.action_history.copy()
        return new_state

    def get_calmness(self, agent_id: int) -> float:
        """Get calmness for specified agent."""
        return self.calmness_a if agent_id == 0 else self.calmness_b

    def get_irritability(self, agent_id: int) -> float:
        """Get irritability trait for specified agent."""
        return self.irritability_a if agent_id == 0 else self.irritability_b

    def update_calmness(self, agent_id: int, delta: float, recovery_rate: float = 0.02):
        """
        Update calmness for specified agent.

        Args:
            agent_id: Agent ID (0 for A, 1 for B)
            delta: Change in calmness from action (can be negative for negative actions)
            recovery_rate: Automatic recovery rate per step (always positive)
        """
        # Apply delta from action first, then recovery
        # Recovery helps calmness naturally increase over time
        if agent_id == 0:
            self.calmness_a = np.clip(self.calmness_a + delta, 0.0, 1.0)
            # Apply recovery after clipping
            self.calmness_a = np.clip(self.calmness_a + recovery_rate, 0.0, 1.0)
        else:
            self.calmness_b = np.clip(self.calmness_b + delta, 0.0, 1.0)
            # Apply recovery after clipping
            self.calmness_b = np.clip(self.calmness_b + recovery_rate, 0.0, 1.0)

    def __repr__(self):
        return (
            f"RelationshipState("
            f"emotion={self.emotion_level:.2f}, "
            f"trust={self.trust_level:.2f}, "
            f"conflict={self.conflict_intensity:.2f}, "
            f"calm_a={self.calmness_a:.2f}, "
            f"calm_b={self.calmness_b:.2f}, "
            f"history_len={len(self.action_history)})"
        )

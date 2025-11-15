"""
State Space Definition

Defines the relationship state representation, including:
1. Core relationship metrics (emotion, trust, conflict)
2. Dialogue history for deep RL agents
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

    Dialogue History (for Deep RL):
    - action_history: Sequence of recent actions taken
    - history_length: Maximum length of history to maintain
    """

    def __init__(
        self,
        emotion_level: float = 0.0,
        trust_level: float = 0.5,
        conflict_intensity: float = 0.5,
        history_length: int = 10,
    ):
        """
        Initialize relationship state.

        Args:
            emotion_level: Initial emotion level (default: 0.0, neutral)
            trust_level: Initial trust level (default: 0.5, moderate)
            conflict_intensity: Initial conflict intensity (default: 0.5, moderate)
            history_length: Maximum length of action history to maintain
        """
        # Core relationship metrics
        self.emotion_level = np.clip(emotion_level, -1.0, 1.0)
        self.trust_level = np.clip(trust_level, 0.0, 1.0)
        self.conflict_intensity = np.clip(conflict_intensity, 0.0, 1.0)

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

    def get_full_state(self) -> np.ndarray:
        """
        Get full state vector including dialogue history for deep RL.

        Returns:
            numpy array of shape (3 + history_length,) containing
            core state + one-hot encoded action history
        """
        # Core state
        core = self.get_core_state()

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

    def get_discretized_state(self, bins: int = 5) -> tuple:
        """
        Discretize state for tabular RL methods.

        Args:
            bins: Number of bins per dimension (default: 5)

        Returns:
            Tuple of discretized indices (emotion_bin, trust_bin, conflict_bin)
        """
        # Discretize each dimension into bins
        emotion_bin = np.digitize(self.emotion_level, np.linspace(-1, 1, bins)) - 1
        emotion_bin = np.clip(emotion_bin, 0, bins - 1)

        trust_bin = np.digitize(self.trust_level, np.linspace(0, 1, bins)) - 1
        trust_bin = np.clip(trust_bin, 0, bins - 1)

        conflict_bin = np.digitize(self.conflict_intensity, np.linspace(0, 1, bins)) - 1
        conflict_bin = np.clip(conflict_bin, 0, bins - 1)

        return (emotion_bin, trust_bin, conflict_bin)

    def copy(self) -> "RelationshipState":
        """Create a deep copy of the state."""
        new_state = RelationshipState(
            emotion_level=self.emotion_level,
            trust_level=self.trust_level,
            conflict_intensity=self.conflict_intensity,
            history_length=self.history_length,
        )
        new_state.action_history = self.action_history.copy()
        return new_state

    def __repr__(self):
        return (
            f"RelationshipState("
            f"emotion={self.emotion_level:.2f}, "
            f"trust={self.trust_level:.2f}, "
            f"conflict={self.conflict_intensity:.2f}, "
            f"history_len={len(self.action_history)})"
        )

"""
Personality Policy Definitions

Defines different personality types that influence agent behavior:
- Impulsive: More likely to use blame/defensive actions
- Sensitive: More reactive to negative states, trust more volatile
- Avoidant: Tends to withdraw or change topic
- Neutral: Balanced behavior
"""

from enum import Enum
from typing import Dict
import numpy as np


class PersonalityType(Enum):
    """Personality types that affect agent behavior and state perception."""

    IMPULSIVE = "impulsive"  # Tendency to blame/defensive actions
    SENSITIVE = "sensitive"  # More reactive, trust more volatile
    AVOIDANT = "avoidant"  # Tendency to withdraw/change topic
    NEUTRAL = "neutral"  # Balanced behavior


class PersonalityPolicy:
    """
    Defines personality-based behavior modifications.

    Each personality type affects:
    1. Action selection bias (preferences for certain actions)
    2. State perception (how agent interprets relationship state)
    3. Transition effects (how actions affect state - handled in TransitionModel)
    """

    def __init__(self, personality_type: PersonalityType = PersonalityType.NEUTRAL):
        """
        Initialize personality policy.

        Args:
            personality_type: Type of personality
        """
        self.personality_type = personality_type
        self._action_biases = self._initialize_action_biases()
        self._state_perception_modifier = self._initialize_state_perception()

    def _initialize_action_biases(self) -> Dict[int, float]:
        """
        Initialize action selection biases based on personality.

        Returns:
            Dictionary mapping action indices to bias values
        """
        biases = {i: 0.0 for i in range(10)}  # 10 action types

        if self.personality_type == PersonalityType.IMPULSIVE:
            # Impulsive: higher probability for blame, defensive
            biases[7] = 0.3  # DEFENSIVE
            biases[8] = 0.4  # BLAME
            biases[0] = -0.2  # Less likely to APOLOGIZE

        elif self.personality_type == PersonalityType.SENSITIVE:
            # Sensitive: more likely to empathize, reassure, or withdraw
            biases[1] = 0.2  # EMPATHIZE
            biases[3] = 0.2  # REASSURE
            biases[9] = 0.3  # WITHDRAW (under stress)
            biases[8] = -0.1  # Less likely to BLAME

        elif self.personality_type == PersonalityType.AVOIDANT:
            # Avoidant: higher probability for withdraw, change topic
            biases[9] = 0.4  # WITHDRAW
            biases[6] = 0.3  # CHANGE_TOPIC
            biases[0] = -0.2  # Less likely to APOLOGIZE
            biases[1] = -0.1  # Less likely to EMPATHIZE

        # Neutral: no bias

        return biases

    def _initialize_state_perception(self) -> Dict[str, float]:
        """
        Initialize state perception modifiers.

        Some personalities may perceive states differently (e.g., sensitive
        may perceive conflict as more intense).

        Returns:
            Dictionary with perception modifiers for emotion, trust, conflict
        """
        if self.personality_type == PersonalityType.SENSITIVE:
            return {
                "emotion_multiplier": 1.2,  # Perceives emotions more intensely
                "trust_multiplier": 1.3,  # Trust changes more volatile
                "conflict_multiplier": 1.2,  # Perceives conflict as more intense
            }
        elif self.personality_type == PersonalityType.AVOIDANT:
            return {
                "emotion_multiplier": 0.9,  # Downplays emotions
                "trust_multiplier": 0.8,  # Less sensitive to trust changes
                "conflict_multiplier": 1.1,  # Slightly amplifies conflict perception
            }
        else:
            return {
                "emotion_multiplier": 1.0,
                "trust_multiplier": 1.0,
                "conflict_multiplier": 1.0,
            }

    def get_action_bias(self, action_idx: int) -> float:
        """
        Get action selection bias for a given action.

        Args:
            action_idx: Index of action (0-9)

        Returns:
            Bias value (added to action logits/Q-values)
        """
        return self._action_biases.get(action_idx, 0.0)

    def modify_state_perception(self, state: np.ndarray) -> np.ndarray:
        """
        Modify perceived state based on personality.

        Args:
            state: Original state vector [emotion, trust, conflict]

        Returns:
            Modified state vector
        """
        modifiers = self._state_perception_modifier
        modified_state = state.copy()

        if len(modified_state) >= 3:
            modified_state[0] *= modifiers["emotion_multiplier"]
            modified_state[1] *= modifiers["trust_multiplier"]
            modified_state[2] *= modifiers["conflict_multiplier"]

        return modified_state

    def get_personality_name(self) -> str:
        """Get personality type name as string."""
        return self.personality_type.value

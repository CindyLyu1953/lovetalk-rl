"""
Transition Model

Defines how actions affect relationship state changes (emotion, trust, conflict).
Values are calibrated based on psychological models (Gottman, NVC) and can be
calibrated using real dialogue data (DailyDialog, EmpatheticDialogues).
"""

from typing import Dict, Tuple
import numpy as np
from .actions import ActionType


class TransitionModel:
    """
    Models state transitions based on actions taken by agents.

    Each action type has associated effects on:
    - emotion_level: Change in emotional valence
    - trust_level: Change in trust
    - conflict_intensity: Change in conflict intensity

    Values are calibrated based on:
    1. Gottman's Four Horsemen model (negative actions)
    2. Nonviolent Communication (NVC) model (positive actions)
    3. Emotion Regulation & Repair Research
    4. Real dialogue data calibration (optional)
    """

    def __init__(self):
        """
        Initialize transition model with default action effects.
        These values can be calibrated using real dialogue data.
        """
        # Action effects: (delta_emotion, delta_trust, delta_conflict)
        # Positive actions typically increase emotion/trust and decrease conflict
        # Negative actions typically decrease emotion/trust and increase conflict

        self.action_effects: Dict[ActionType, Tuple[float, float, float]] = {
            # Positive actions (NVC-based)
            ActionType.APOLOGIZE: (
                0.15,
                0.20,
                -0.15,
            ),  # Strong trust boost, conflict reduction
            ActionType.EMPATHIZE: (0.12, 0.15, -0.12),  # Emotion and trust improvement
            ActionType.EXPLAIN: (0.05, 0.08, -0.08),  # Moderate positive effect
            ActionType.REASSURE: (0.10, 0.12, -0.10),  # Emotion improvement
            ActionType.SUGGEST_SOLUTION: (0.08, 0.10, -0.15),  # Conflict reduction
            ActionType.ASK_FOR_NEEDS: (0.06, 0.10, -0.08),  # Trust building
            # Neutral actions
            ActionType.CHANGE_TOPIC: (
                0.0,
                -0.02,
                0.05,
            ),  # Slight conflict increase, trust decrease
            # Negative actions (Gottman's Four Horsemen)
            ActionType.DEFENSIVE: (-0.10, -0.08, 0.12),  # Increases conflict
            ActionType.BLAME: (-0.15, -0.15, 0.18),  # Strong negative effect
            ActionType.WITHDRAW: (-0.12, -0.12, 0.10),  # Emotion and trust decrease
        }

        # Personality modifiers (can be adjusted per agent)
        self.default_personality_modifier = {
            "impulsive": 1.3,  # Amplifies all effects
            "sensitive": 1.2,  # Amplifies negative effects more
            "avoidant": 0.8,  # Reduces all effects
            "neutral": 1.0,
        }

    def compute_transition(
        self, action: ActionType, personality_type: str = "neutral"
    ) -> Tuple[float, float, float]:
        """
        Compute state transition given an action and personality type.

        Args:
            action: The action taken
            personality_type: Personality type affecting action impact

        Returns:
            Tuple of (delta_emotion, delta_trust, delta_conflict)
        """
        base_effect = self.action_effects[action]
        modifier = self.default_personality_modifier.get(personality_type, 1.0)

        # Apply personality modifier
        delta_emotion, delta_trust, delta_conflict = [
            effect * modifier for effect in base_effect
        ]

        # Add small random noise to model uncertainty (optional)
        noise_scale = 0.02
        delta_emotion += np.random.normal(0, noise_scale)
        delta_trust += np.random.normal(0, noise_scale)
        delta_conflict += np.random.normal(0, noise_scale)

        return delta_emotion, delta_trust, delta_conflict

    def update_state(
        self,
        current_state: "RelationshipState",
        action: ActionType,
        personality_type: str = "neutral",
    ) -> "RelationshipState":
        """
        Update relationship state based on action.

        Args:
            current_state: Current relationship state
            action: Action taken
            personality_type: Personality type affecting transition

        Returns:
            New relationship state after transition
        """
        delta_emotion, delta_trust, delta_conflict = self.compute_transition(
            action, personality_type
        )

        # Create new state with updated values
        new_state = current_state.copy()
        new_state.emotion_level = np.clip(
            new_state.emotion_level + delta_emotion, -1.0, 1.0
        )
        new_state.trust_level = np.clip(new_state.trust_level + delta_trust, 0.0, 1.0)
        new_state.conflict_intensity = np.clip(
            new_state.conflict_intensity + delta_conflict, 0.0, 1.0
        )

        # Add action to history
        new_state.add_action(action.value)

        return new_state

    def calibrate_from_data(self, calibration_data: Dict):
        """
        Calibrate transition model from real dialogue data.

        This method would update action_effects based on statistical analysis
        of real dialogue datasets (DailyDialog, EmpatheticDialogues).

        Args:
            calibration_data: Dictionary containing calibrated action effects
                Format: {ActionType: (delta_emotion, delta_trust, delta_conflict), ...}
        """
        for action_type, effects in calibration_data.items():
            if action_type in self.action_effects:
                self.action_effects[action_type] = effects

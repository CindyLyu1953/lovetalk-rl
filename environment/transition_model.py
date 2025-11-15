"""
Transition Model

Defines how actions affect relationship state changes (emotion, trust, calmness).
Values are calibrated based on psychological models (Gottman, NVC) and real data.
Updated to include calmness effects based on the revised transition table.
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
    - calmness: Change in calmness (per-agent internal state)

    Values are calibrated based on:
    1. Updated transition table with calmness effects
    2. Gottman's Four Horsemen model (negative actions)
    3. Nonviolent Communication (NVC) model (positive actions)
    4. Emotion Regulation & Repair Research

    Key insights from updated table:
    - EMPATHIZE: Most effective for emotion repair, but doesn't directly increase trust
    - EXPLAIN/SUGGEST_SOLUTION: Strong trust builders, moderate emotion effects
    - APOLOGIZE/ASK_FOR_NEEDS: Balanced effects, not extreme
    - BLAME: Most damaging to emotion
    - WITHDRAW: Most damaging to trust
    """

    def __init__(self):
        """
        Initialize transition model with updated action effects including calmness.
        Values based on the revised transition table.
        """
        # Action effects: (delta_emotion, delta_trust, delta_calmness)
        # Updated according to the new transition table

        self.action_effects: Dict[ActionType, Tuple[float, float, float]] = {
            # Positive actions (NVC-based)
            ActionType.APOLOGIZE: (
                0.25,
                0.35,
                0.18,
            ),  # Taking responsibility → trust rises significantly; moderate emotion boost
            ActionType.EMPATHIZE: (
                0.45,
                0.12,
                0.30,
            ),  # Most effective emotional repair, but doesn't necessarily increase trust
            ActionType.EXPLAIN: (
                0.12,
                0.45,
                0.08,
            ),  # Clarifying facts → strongly increases trust, but helps little with emotion
            ActionType.REASSURE: (
                0.40,
                0.15,
                0.25,
            ),  # Reassuring speech → emotional stability + slight trust increase
            ActionType.SUGGEST_SOLUTION: (
                0.15,
                0.40,
                0.12,
            ),  # Problem-solving oriented → trust significantly increases, emotion slightly improves
            ActionType.ASK_FOR_NEEDS: (
                0.28,
                0.28,
                0.20,
            ),  # Dual improvement, moderately balanced action, not extreme
            # Neutral actions
            ActionType.CHANGE_TOPIC: (
                -0.10,
                -0.15,
                -0.08,
            ),  # Other party might be unhappy, but not serious; slight emotion and trust decrease
            # Negative actions (Gottman's Four Horsemen)
            # Fixed: Reduced negative action impacts to prevent immediate termination
            # Scaled down by ~0.6 to give agents more room to recover
            ActionType.DEFENSIVE: (
                -0.25,  # Was -0.40, reduced to prevent immediate termination
                -0.20,  # Was -0.35, reduced
                -0.15,  # Was -0.25, reduced
            ),  # Defensive behavior → emotion triggered, trust decreases, calmness drops quickly
            ActionType.BLAME: (
                -0.33,  # Was -0.55, reduced to prevent immediate termination
                -0.27,  # Was -0.45, reduced
                -0.21,  # Was -0.35, reduced
            ),  # Explicit accusation → most severe aggressive speech, all three items significantly decrease
            ActionType.WITHDRAW: (
                -0.27,  # Was -0.45, reduced to prevent immediate termination
                -0.30,  # Was -0.50, reduced
                -0.18,  # Was -0.30, reduced
            ),  # Cold war/Silence → strongly reduces trust, also makes atmosphere worse
        }

        # Calmness effects on action feasibility are handled in ActionFeasibility module

    def compute_transition(
        self, action: ActionType, irritability: float = 0.4
    ) -> Tuple[float, float, float]:
        """
        Compute state transition given an action and agent's irritability trait.

        Args:
            action: The action taken
            irritability: Agent's irritability trait [0, 1]
                         Affects how calmness changes from actions

        Returns:
            Tuple of (delta_emotion, delta_trust, delta_calmness)
        """
        base_effect = self.action_effects[action]
        delta_emotion, delta_trust, delta_calmness = base_effect

        # Irritability affects calmness changes:
        # - Negative actions → calmness drops more for high irritability
        # - Positive actions → calmness rises less for high irritability
        if delta_calmness < 0:
            # Negative action: apply irritability multiplier (more drop)
            delta_calmness *= 1 + irritability
        else:
            # Positive action: apply irritability multiplier (less rise)
            delta_calmness *= 1 - irritability

        # Add small random noise to model uncertainty (optional)
        noise_scale = 0.02
        delta_emotion += np.random.normal(0, noise_scale)
        delta_trust += np.random.normal(0, noise_scale)
        delta_calmness += np.random.normal(0, noise_scale * 0.5)

        return delta_emotion, delta_trust, delta_calmness

    def update_state(
        self,
        current_state: "RelationshipState",
        action: ActionType,
        agent_id: int,
        recovery_rate: float = 0.02,
    ) -> "RelationshipState":
        """
        Update relationship state based on action.

        Args:
            current_state: Current relationship state
            action: Action taken
            agent_id: Agent ID (0 for A, 1 for B)
            recovery_rate: Automatic calmness recovery rate per step

        Returns:
            New relationship state after transition
        """
        # Get agent's irritability trait
        irritability = current_state.get_irritability(agent_id)

        # Compute transitions
        delta_emotion, delta_trust, delta_calmness = self.compute_transition(
            action, irritability
        )

        # Create new state with updated values
        new_state = current_state.copy()

        # Update relationship metrics
        new_state.emotion_level = np.clip(
            new_state.emotion_level + delta_emotion, -1.0, 1.0
        )
        new_state.trust_level = np.clip(new_state.trust_level + delta_trust, 0.0, 1.0)

        # Update conflict intensity (derived from emotion and trust)
        # Conflict increases when emotion is negative and trust is low
        conflict_from_emotion = max(0, -new_state.emotion_level) * 0.5
        conflict_from_trust = (1.0 - new_state.trust_level) * 0.5
        new_state.conflict_intensity = np.clip(
            conflict_from_emotion + conflict_from_trust, 0.0, 1.0
        )

        # Update calmness for the agent who took the action
        new_state.update_calmness(agent_id, delta_calmness, recovery_rate)

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
                Format: {ActionType: (delta_emotion, delta_trust, delta_calmness), ...}
        """
        for action_type, effects in calibration_data.items():
            if action_type in self.action_effects:
                self.action_effects[action_type] = effects

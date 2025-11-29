"""
Transition Model

Defines how actions affect relationship state changes (emotion, trust, calmness).
Values are calibrated based on psychological models (Gottman, NVC) and real data.
Updated to include calmness effects based on the revised transition table.
"""

from typing import Dict, Tuple
import numpy as np
from .actions import ActionType
from personality.personality_policy import PersonalityType


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
        # Legacy single-value effects (kept for reference)
        self.base_action_effects: Dict[ActionType, Tuple[float, float, float]] = {
            ActionType.APOLOGIZE: (0.25, 0.35, 0.18),
            ActionType.EMPATHIZE: (0.45, 0.12, 0.30),
            ActionType.EXPLAIN: (0.12, 0.45, 0.08),
            ActionType.REASSURE: (0.40, 0.15, 0.25),
            ActionType.SUGGEST_SOLUTION: (0.15, 0.40, 0.12),
            ActionType.ASK_FOR_NEEDS: (0.28, 0.28, 0.20),
            ActionType.CHANGE_TOPIC: (-0.10, -0.15, -0.08),
            ActionType.DEFENSIVE: (-0.25, -0.20, -0.15),
            ActionType.BLAME: (-0.33, -0.27, -0.21),
            ActionType.WITHDRAW: (-0.27, -0.30, -0.18),
        }

        # Personality-specific action effect RANGES (min, max) for (emotion, trust, calmness)
        # Values adopted from experimental tables (Neuroticism, Agreeableness, Conscientiousness, Avoidant, Baseline)
        # Format: {PersonalityType: {ActionType: ((e_min,e_max),(t_min,t_max),(c_min,c_max)), ...}, ...}
        self.personality_action_ranges: Dict[PersonalityType, Dict[ActionType, Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]]] = {
            # Baseline / neutral
            PersonalityType.NEUTRAL: {
                ActionType.APOLOGIZE: ((0.34, 0.42), (0.45, 0.53), (0.28, 0.36)),
                ActionType.EMPATHIZE: ((0.43, 0.49), (0.22, 0.27), (0.40, 0.49)),
                ActionType.EXPLAIN: ((0.40, 0.49), (0.28, 0.33), (0.36, 0.45)),
                ActionType.REASSURE: ((0.33, 0.40), (0.40, 0.48), (0.28, 0.35)),
                ActionType.SUGGEST_SOLUTION: ((0.24, 0.30), (0.48, 0.60), (0.23, 0.29)),
                ActionType.ASK_FOR_NEEDS: ((0.28, 0.36), (0.42, 0.53), (0.29, 0.38)),
                ActionType.CHANGE_TOPIC: ((0.00, -0.04), ( -0.01, -0.07), (0.03, 0. -0.02)),
                ActionType.DEFENSIVE: ((-0.31, -0.39), (-0.21, -0.29), (-0.15, -0.23)),
                ActionType.BLAME: ((-0.41, -0.50), (-0.30, -0.40), (-0.18, -0.28)),
                ActionType.WITHDRAW: ((-0.28, -0.38), (-0.34, -0.46), (-0.09, -0.20)),
            },
            # Neuroticism
            PersonalityType.NEUROTIC: {
                ActionType.APOLOGIZE: ((0.32, 0.39), (0.41, 0.50), (0.23, 0.30)),
                ActionType.EMPATHIZE: ((0.44, 0.53), (0.21, 0.28), (0.33, 0.41)),
                ActionType.EXPLAIN: ((0.29, 0.36), (0.36, 0.43), (0.21, 0.27)),
                ActionType.REASSURE: ((0.41, 0.52), (0.24, 0.30), (0.29, 0.36)),
                ActionType.SUGGEST_SOLUTION: ((0.21, 0.27), (0.43, 0.54), (0.19, 0.25)),
                ActionType.ASK_FOR_NEEDS: ((0.26, 0.34), (0.38, 0.50), (0.24, 0.31)),
                ActionType.CHANGE_TOPIC: ((-0.03, -0.08), (-0.05, -0.12), (0.01, -0.04)),
                ActionType.DEFENSIVE: ((-0.35, -0.47), (-0.26, -0.38), (-0.18, -0.29)),
                ActionType.BLAME: ((-0.43, -0.56), (-0.31, -0.45), (-0.21, -0.35)),
                ActionType.WITHDRAW: ((-0.31, -0.42), (-0.37, -0.52), (-0.11, -0.22)),
            },
            # Agreeableness
            PersonalityType.AGREEABLE: {
                ActionType.APOLOGIZE: ((0.36, 0.45), (0.49, 0.60), (0.33, 0.41)),
                ActionType.EMPATHIZE: ((0.47, 0.58), (0.25, 0.33), (0.42, 0.53)),
                ActionType.EXPLAIN: ((0.35, 0.44), (0.44, 0.54), (0.31, 0.39)),
                ActionType.REASSURE: ((0.44, 0.55), (0.30, 0.37), (0.39, 0.48)),
                ActionType.SUGGEST_SOLUTION: ((0.26, 0.33), (0.52, 0.64), (0.28, 0.35)),
                ActionType.ASK_FOR_NEEDS: ((0.30, 0.38), (0.47, 0.60), (0.33, 0.41)),
                ActionType.CHANGE_TOPIC: ((0.02, -0.02), (0.01, -0.04), (0.05, 0.00)),
                ActionType.DEFENSIVE: ((-0.26, -0.37), (-0.20, -0.33), (-0.11, -0.21)),
                ActionType.BLAME: ((-0.31, -0.45), (-0.26, -0.40), (-0.14, -0.27)),
                ActionType.WITHDRAW: ((-0.19, -0.30), (-0.32, -0.47), (-0.07, -0.16)),
            },
            # Conscientiousness
            PersonalityType.CONSCIENTIOUS: {
                ActionType.APOLOGIZE: ((0.33, 0.41), (0.48, 0.57), (0.30, 0.37)),
                ActionType.EMPATHIZE: ((0.37, 0.45), (0.20, 0.25), (0.27, 0.35)),
                ActionType.EXPLAIN: ((0.35, 0.45), (0.46, 0.58), (0.33, 0.41)),
                ActionType.REASSURE: ((0.39, 0.48), (0.23, 0.29), (0.31, 0.38)),
                ActionType.SUGGEST_SOLUTION: ((0.27, 0.34), (0.53, 0.66), (0.29, 0.37)),
                ActionType.ASK_FOR_NEEDS: ((0.29, 0.37), (0.45, 0.56), (0.31, 0.39)),
                ActionType.CHANGE_TOPIC: ((0.01, -0.03), (-0.03, -0.10), (0.03, -0.01)),
                ActionType.DEFENSIVE: ((-0.34, -0.46), (-0.28, -0.41), (-0.17, -0.28)),
                ActionType.BLAME: ((-0.37, -0.50), (-0.32, -0.45), (-0.18, -0.31)),
                ActionType.WITHDRAW: ((-0.24, -0.36), (-0.38, -0.51), (-0.07, -0.18)),
            },
            # Avoidant
            PersonalityType.AVOIDANT: {
                ActionType.APOLOGIZE: ((0.27, 0.34), (0.34, 0.44), (0.22, 0.28)),
                ActionType.EMPATHIZE: ((0.24, 0.31), (0.18, 0.24), (0.23, 0.29)),
                ActionType.EXPLAIN: ((0.26, 0.33), (0.34, 0.43), (0.25, 0.31)),
                ActionType.REASSURE: ((0.23, 0.30), (0.20, 0.27), (0.24, 0.29)),
                ActionType.SUGGEST_SOLUTION: ((0.26, 0.32), (0.38, 0.49), (0.25, 0.32)),
                ActionType.ASK_FOR_NEEDS: ((0.20, 0.27), (0.19, 0.26), (0.18, 0.24)),
                ActionType.CHANGE_TOPIC: ((0.05, 0.02), (0.04, 0.00), (0.07, 0.03)),
                ActionType.DEFENSIVE: ((-0.29, -0.41), (-0.26, -0.38), (-0.12, -0.22)),
                ActionType.BLAME: ((-0.36, -0.48), (-0.29, -0.44), (-0.15, -0.27)),
                ActionType.WITHDRAW: ((-0.37, -0.52), (-0.43, -0.61), (-0.08, -0.17)),
            },
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
        # Legacy fallback (if called without personality/rng) - sample around base effects
        delta_emotion, delta_trust, delta_calmness = self.base_action_effects.get(action, (0.0, 0.0, 0.0))
        # Apply small gaussian noise for legacy behavior
        noise_scale = 0.02
        delta_emotion += np.random.normal(0, noise_scale)
        delta_trust += np.random.normal(0, noise_scale)
        delta_calmness += np.random.normal(0, noise_scale * 0.5)
        return delta_emotion, delta_trust, delta_calmness

    def _compute_midpoint(self, low: float, high: float) -> float:
        """
        Compute the midpoint (average) of [low, high] interval.
        
        UPGRADED: Changed from Beta sampling to simple average to reduce noise
        and make transitions more predictable for learning.
        
        Args:
            low: Lower bound of interval
            high: Upper bound of interval
        
        Returns:
            Midpoint (average) of the interval
        """
        return (low + high) / 2.0

    def compute_transition_with_personality(
        self,
        action: ActionType,
        personality: PersonalityType,
        rng: np.random.Generator,
        irritability: float = 0.4,
    ) -> Tuple[float, float, float]:
        """
        Compute state transition given an action and an agent's personality using
        personality-specific ranges.
        
        UPGRADED: Changed from Beta sampling to midpoint (average) to reduce noise
        and make learning more stable.
        """
        # Use personality ranges if available, otherwise fallback to neutral ranges
        ranges_for_person = self.personality_action_ranges.get(personality, self.personality_action_ranges.get(PersonalityType.NEUTRAL))
        if ranges_for_person is None or action not in ranges_for_person:
            # Fallback to base_effects with small noise
            return self.compute_transition(action, irritability)

        (e_low, e_high), (t_low, t_high), (c_low, c_high) = ranges_for_person[action]

        # UPGRADED: Use midpoint (average) instead of sampling
        delta_emotion = self._compute_midpoint(e_low, e_high)
        delta_trust = self._compute_midpoint(t_low, t_high)
        delta_calmness = self._compute_midpoint(c_low, c_high)

        # Adjust calmness change by irritability: negative changes amplified, positive reduced
        if delta_calmness < 0:
            delta_calmness *= 1 + irritability
        else:
            delta_calmness *= 1 - irritability

        # REMOVED: Gaussian noise (deterministic transitions for better learning)
        # noise_scale = 0.02
        # delta_emotion += rng.normal(0, noise_scale)
        # delta_trust += rng.normal(0, noise_scale)
        # delta_calmness += rng.normal(0, noise_scale * 0.5)

        return float(delta_emotion), float(delta_trust), float(delta_calmness)

    def update_state(
        self,
        current_state: "RelationshipState",
        action: ActionType,
        agent_id: int,
        recovery_rate: float = 0.02,
        personality: PersonalityType = PersonalityType.NEUTRAL,
        rng: np.random.Generator = None,
        cross_agent_calmness_factor: float = 0.6,
    ) -> "RelationshipState":
        """
        Update relationship state based on action.
        
        Now updates calmness for BOTH agents:
        - Agent who took the action: full effect (delta_calmness)
        - Other agent: partial effect (delta_calmness * cross_agent_calmness_factor)
        
        This allows positive actions from one agent to help the other agent recover
        from low calmness, breaking negative feedback loops.

        Args:
            current_state: Current relationship state
            action: Action taken
            agent_id: Agent ID (0 for A, 1 for B)
            recovery_rate: Automatic calmness recovery rate per step
            personality: Personality of the agent taking the action
            rng: Random number generator for sampling
            cross_agent_calmness_factor: Multiplier for how much the other agent's
                                        calmness is affected (default: 0.6, meaning
                                        60% of the effect)

        Returns:
            New relationship state after transition
        """
        # Get agent's irritability trait
        irritability = current_state.get_irritability(agent_id)
        # Ensure RNG
        if rng is None:
            rng = np.random.default_rng()

        # Compute transitions using personality-aware sampling
        try:
            # Accept either PersonalityType or string
            if isinstance(personality, str):
                try:
                    personality_enum = PersonalityType(personality)
                except Exception:
                    personality_enum = PersonalityType.NEUTRAL
            else:
                personality_enum = personality

            delta_emotion, delta_trust, delta_calmness = self.compute_transition_with_personality(
                action, personality_enum, rng, irritability
            )
        except Exception:
            # Fallback to legacy deterministic transition if anything fails
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

        # Update calmness for BOTH agents
        # 1. Agent who took the action: full effect
        new_state.update_calmness(agent_id, delta_calmness, recovery_rate)
        
        # 2. Other agent: partial effect (reduced by cross_agent_calmness_factor)
        # This allows one agent's positive actions to help the other recover
        other_agent_id = 1 - agent_id
        other_agent_delta_calmness = delta_calmness * cross_agent_calmness_factor
        # Note: Other agent's recovery is also applied in update_calmness
        new_state.update_calmness(other_agent_id, other_agent_delta_calmness, recovery_rate)

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

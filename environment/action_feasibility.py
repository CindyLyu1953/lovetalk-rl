"""
Action Feasibility Module

Implements action feasibility mechanism based on calmness level.
This models how internal self-regulation state affects the ability to
choose certain actions during conflict.
"""

from typing import Dict
import numpy as np
from .actions import ActionType, NUM_ACTIONS


class ActionFeasibility:
    """
    Computes action feasibility weights based on calmness level.

    Core idea:
    Actions have different "emotional difficulty" levels. When calmness is low,
    negative actions (BLAME, DEFENSIVE, WITHDRAW) become more feasible,
    while positive actions (APOLOGIZE, EMPATHIZE) become nearly impossible.

    Formula:
    feasibility(a) = exp(α * calmness - β * difficulty(a))

    The modified policy becomes:
    π'(a) = π(a) * feasibility(a) (normalized)
    """

    def __init__(self, alpha: float = 1.0, beta: float = 1.0):
        """
        Initialize action feasibility calculator.

        Args:
            alpha: Weight for calmness in feasibility calculation (default: 1.0)
            beta: Weight for action difficulty (default: 1.0)
        """
        self.alpha = alpha
        self.beta = beta

        # Define emotional difficulty for each action
        # Higher difficulty → harder to perform when calmness is low
        # Negative difficulty → easier when calmness is low (negative actions)
        self.action_difficulty: Dict[ActionType, float] = {
            # Positive actions (require high calmness)
            ActionType.APOLOGIZE: +2.0,  # Requires high self-regulation
            ActionType.EMPATHIZE: +2.0,  # Requires high self-regulation
            ActionType.REASSURE: +1.5,  # Moderate difficulty
            ActionType.SUGGEST_SOLUTION: +1.0,  # Moderate difficulty
            # Balanced actions
            ActionType.EXPLAIN: +1.0,  # Moderate difficulty
            ActionType.ASK_FOR_NEEDS: +1.0,  # Moderate difficulty
            # Neutral actions
            ActionType.CHANGE_TOPIC: 0.0,  # Neutral difficulty
            # Negative actions (easier when calmness is low)
            ActionType.WITHDRAW: -1.0,  # Natural when upset (avoidant behavior)
            ActionType.DEFENSIVE: -2.0,  # Easy to fall into when agitated
            ActionType.BLAME: -2.0,  # Easiest when very upset
        }

    def compute_feasibility(self, calmness: float) -> np.ndarray:
        """
        Compute feasibility weights for all actions given calmness level.

        Args:
            calmness: Current calmness level [0, 1]

        Returns:
            Array of feasibility weights for each action (shape: (NUM_ACTIONS,))
        """
        feasibility = np.zeros(NUM_ACTIONS, dtype=np.float32)

        for action_idx in range(NUM_ACTIONS):
            action_type = ActionType(action_idx)
            difficulty = self.action_difficulty[action_type]

            # Compute feasibility: exp(α * calmness - β * difficulty)
            feasibility[action_idx] = np.exp(
                self.alpha * calmness - self.beta * difficulty
            )

        return feasibility

    def modify_policy(self, action_probs: np.ndarray, calmness: float) -> np.ndarray:
        """
        Modify policy probabilities based on calmness feasibility.

        Args:
            action_probs: Original action probabilities (shape: (NUM_ACTIONS,))
            calmness: Current calmness level [0, 1]

        Returns:
            Modified and normalized action probabilities
        """
        # Compute feasibility weights
        feasibility = self.compute_feasibility(calmness)

        # Apply feasibility to policy
        modified_probs = action_probs * feasibility

        # Normalize to valid probability distribution
        prob_sum = modified_probs.sum()
        if prob_sum > 1e-8:
            modified_probs = modified_probs / prob_sum
        else:
            # Fallback to uniform if all probabilities are near zero
            modified_probs = np.ones_like(action_probs) / NUM_ACTIONS

        return modified_probs

    def modify_q_values(self, q_values: np.ndarray, calmness: float) -> np.ndarray:
        """
        Modify Q-values based on calmness feasibility.

        This adds feasibility as a bias to Q-values, making them more
        realistic given the agent's current emotional state.

        Args:
            q_values: Original Q-values (shape: (NUM_ACTIONS,))
            calmness: Current calmness level [0, 1]

        Returns:
            Modified Q-values with feasibility bias
        """
        # Compute feasibility weights
        feasibility = self.compute_feasibility(calmness)

        # Add feasibility as bias (scale by temperature-like factor)
        # Higher feasibility → positive bias, lower → negative bias
        feasibility_bias = (feasibility - feasibility.mean()) * 0.5

        modified_q = q_values + feasibility_bias

        return modified_q

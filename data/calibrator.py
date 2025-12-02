"""
Transition Model Calibrator

Calibrates transition model parameters using real dialogue data.
"""

from typing import Dict, Tuple
from environment.transition_model import TransitionModel
from environment.actions import ActionType
from .data_loader import DataLoader


class TransitionCalibrator:
    """
    Calibrates transition model based on real dialogue data.

    Uses statistical analysis of dialogue datasets (DailyDialog, EmpatheticDialogues)
    to estimate action effects on emotion, trust, and conflict.
    """

    def __init__(self):
        """Initialize transition calibrator."""
        self.data_loader = DataLoader()
        self.transition_model = TransitionModel()

    def calibrate_from_dailydialog(
        self, dataset_path: str
    ) -> Dict[ActionType, Tuple[float, float, float]]:
        """
        Calibrate transition model from DailyDialog dataset.

        Args:
            dataset_path: Path to DailyDialog dataset

        Returns:
            Dictionary mapping action types to calibrated effects
        """
        # Load dataset
        dialogues = self.data_loader.load_dailydialog(dataset_path)

        # Extract action effects
        action_effects = self.data_loader.extract_action_effects(dialogues)

        # Update transition model
        for action_type, effects in action_effects.items():
            action_enum = ActionType(action_type)
            self.transition_model.action_effects[action_enum] = effects

        return action_effects

    def calibrate_from_empathetic_dialogues(
        self, dataset_path: str
    ) -> Dict[ActionType, Tuple[float, float, float]]:
        """
        Calibrate transition model from EmpatheticDialogues dataset.

        Focuses on empathy-related actions (EMPATHIZE, REASSURE).

        Args:
            dataset_path: Path to EmpatheticDialogues dataset

        Returns:
            Dictionary mapping action types to calibrated effects
        """
        # Load dataset
        dialogues = self.data_loader.load_empathetic_dialogues(dataset_path)

        # Extract action effects (similar to DailyDialog but focused on empathy)
        action_effects = self.data_loader.extract_action_effects(dialogues)

        # Update transition model
        for action_type, effects in action_effects.items():
            action_enum = ActionType(action_type)
            self.transition_model.action_effects[action_enum] = effects

        return action_effects

    def get_calibrated_model(self) -> TransitionModel:
        """Get calibrated transition model."""
        return self.transition_model

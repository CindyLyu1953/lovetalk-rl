"""
Training Package

Contains training and evaluation utilities for RL agents in the
relationship dynamics simulator.
"""

from .trainer import MultiAgentTrainer
from .evaluator import Evaluator

__all__ = ["MultiAgentTrainer", "Evaluator"]

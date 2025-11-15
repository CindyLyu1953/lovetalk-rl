"""
Shallow RL Agents Package

Implements tabular RL methods for discrete state spaces:
- Q-learning
- SARSA
"""

from .q_learning import QLearningAgent
from .sarsa import SarsaAgent

__all__ = ["QLearningAgent", "SarsaAgent"]

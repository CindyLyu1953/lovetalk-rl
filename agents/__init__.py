"""
Agents Package

Contains implementations of various RL agents:
- Shallow RL: Q-learning, SARSA
- Deep RL: DQN, PPO
"""

from .shallow_rl.q_learning import QLearningAgent
from .shallow_rl.sarsa import SarsaAgent

__all__ = ["QLearningAgent", "SarsaAgent"]

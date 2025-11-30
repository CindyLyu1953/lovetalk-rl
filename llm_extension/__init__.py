"""
LLM Extension Module for LoveTalk-RL

This module provides LLM-based dialogue rendering capabilities.
It is completely isolated from RL training and does not affect:
- RL state transitions
- Reward computation
- Policy learning
- Observations

It ONLY converts semantic action labels into natural language utterances.
"""

from .dialogue_renderer import DialogueRenderer

__all__ = ["DialogueRenderer"]


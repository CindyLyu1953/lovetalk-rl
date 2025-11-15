"""
Action Space Definition

Defines the discrete communication actions available to agents in the
relationship dynamics simulator. Actions are based on psychological models
including Gottman's Four Horsemen and Nonviolent Communication (NVC).
"""

from enum import Enum


class ActionType(Enum):
    """
    Discrete communication action types with psychological grounding.

    Positive Actions (NVC-based):
    - APOLOGIZE: Taking responsibility and apologizing
    - EMPATHIZE: Expressing understanding and empathy
    - EXPLAIN: Calmly explaining facts without blame
    - REASSURE: Providing emotional comfort and reassurance
    - SUGGEST_SOLUTION: Proposing constructive solutions
    - ASK_FOR_NEEDS: Inquiring about partner's needs and feelings

    Neutral Actions:
    - CHANGE_TOPIC: Shifting conversation topic (can be positive or negative)

    Negative Actions (Gottman's Four Horsemen):
    - DEFENSIVE: Self-defense and justification (contributes to defensiveness)
    - BLAME: Blaming the partner (contributes to criticism/contempt)
    - WITHDRAW: Silent treatment or avoidance (contributes to stonewalling)
    """

    APOLOGIZE = 0
    EMPATHIZE = 1
    EXPLAIN = 2
    REASSURE = 3
    SUGGEST_SOLUTION = 4
    ASK_FOR_NEEDS = 5
    CHANGE_TOPIC = 6
    DEFENSIVE = 7
    BLAME = 8
    WITHDRAW = 9


class Action:
    """
    Represents a communication action in the environment.

    Attributes:
        action_type: The type of communication action
        agent_id: ID of the agent taking this action (0 for A, 1 for B)
    """

    def __init__(self, action_type: ActionType, agent_id: int):
        self.action_type = action_type
        self.agent_id = agent_id

    def __repr__(self):
        return f"Action(agent={self.agent_id}, type={self.action_type.name})"

    def __eq__(self, other):
        if not isinstance(other, Action):
            return False
        return self.action_type == other.action_type and self.agent_id == other.agent_id


# Action categories for reward shaping
POSITIVE_ACTIONS = {
    ActionType.APOLOGIZE,
    ActionType.EMPATHIZE,
    ActionType.EXPLAIN,
    ActionType.REASSURE,
    ActionType.SUGGEST_SOLUTION,
    ActionType.ASK_FOR_NEEDS,
}

NEGATIVE_ACTIONS = {
    ActionType.DEFENSIVE,
    ActionType.BLAME,
    ActionType.WITHDRAW,
}

NEUTRAL_ACTIONS = {
    ActionType.CHANGE_TOPIC,
}

# Total number of discrete actions
NUM_ACTIONS = len(ActionType)

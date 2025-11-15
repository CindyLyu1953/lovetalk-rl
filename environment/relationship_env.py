"""
Relationship Communication Environment

Implements the main turn-based two-agent communication environment for
simulating relationship dynamics and conflict resolution.
"""

from typing import Dict, Tuple, Optional
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .state import RelationshipState
from .actions import ActionType, NUM_ACTIONS
from .transition_model import TransitionModel
from .action_feasibility import ActionFeasibility


class RelationshipEnv(gym.Env):
    """
    Turn-based two-agent relationship communication environment.

    Interaction Flow:
    1. Partner A takes an action -> environment updates state
    2. Partner B takes an action -> environment updates state
    3. Repeat until episode ends

    The environment maintains relationship state (emotion, trust, conflict)
    and updates it based on actions taken by both agents.
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        max_episode_steps: int = 20,
        history_length: int = 10,
        use_history: bool = False,
        initial_emotion: float = -0.3,
        initial_trust: float = 0.5,
        initial_calmness_a: float = 0.4,
        initial_calmness_b: float = 0.4,
        irritability_a: float = 0.4,
        irritability_b: float = 0.4,
        recovery_rate: float = 0.02,
        reward_weights: Dict[str, float] = None,
        feasibility_alpha: float = 1.0,
        feasibility_beta: float = 1.0,
    ):
        """
        Initialize relationship communication environment.

        Args:
            max_episode_steps: Maximum number of steps per episode (default: 20)
            history_length: Length of action history for deep RL (default: 10)
            use_history: Whether to include action history in state (default: False)
            initial_emotion: Initial emotion level (default: -0.3, slightly negative)
            initial_trust: Initial trust level (default: 0.5, moderate)
            initial_calmness_a: Initial calmness for agent A (default: 0.4, moderate)
            initial_calmness_b: Initial calmness for agent B (default: 0.4, moderate)
            irritability_a: Irritability trait for agent A (default: 0.4, moderate)
            irritability_b: Irritability trait for agent B (default: 0.4, moderate)
            recovery_rate: Automatic calmness recovery rate per step (default: 0.02)
            reward_weights: Weights for reward components (default: balanced)
            feasibility_alpha: Weight for calmness in action feasibility (default: 1.0)
            feasibility_beta: Weight for action difficulty (default: 1.0)
        """
        super().__init__()

        self.max_episode_steps = max_episode_steps
        self.history_length = history_length
        self.use_history = use_history
        self.recovery_rate = recovery_rate

        # Initialize transition model and action feasibility
        self.transition_model = TransitionModel()
        self.action_feasibility = ActionFeasibility(
            alpha=feasibility_alpha, beta=feasibility_beta
        )

        # Reward weights (default: balanced)
        self.reward_weights = reward_weights or {
            "emotion": 1.0,
            "trust": 1.0,
            "action_bonus": 0.1,
        }

        # Episode tracking
        self.current_step = 0
        self.current_agent = 0  # 0 for A, 1 for B
        self.state: Optional[RelationshipState] = None
        self.initial_state: Optional[RelationshipState] = None
        self.termination_reason: Optional[str] = None  # SUCCESS, FAILURE, NEUTRAL

        # Initial state parameters
        self.initial_emotion = initial_emotion
        self.initial_trust = initial_trust
        self.initial_calmness_a = initial_calmness_a
        self.initial_calmness_b = initial_calmness_b
        self.irritability_a = irritability_a
        self.irritability_b = irritability_b

        # Define action space (discrete: 10 action types)
        self.action_space = spaces.Discrete(NUM_ACTIONS)

        # Define observation space
        if use_history:
            # Full state with history: [emotion, trust, conflict, calmness] + history
            obs_dim = 4 + history_length
            self.observation_space = spaces.Box(
                low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32
            )
        else:
            # Core state only (for shallow RL): [emotion, trust, conflict] or [emotion, trust, conflict, calmness]
            # We include calmness for better state representation
            obs_dim = 4
            self.observation_space = spaces.Box(
                low=np.array([-1.0, 0.0, 0.0, 0.0]),
                high=np.array([1.0, 1.0, 1.0, 1.0]),
                dtype=np.float32,
            )

    def reset(
        self, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """
        Reset the environment to initial state.

        Args:
            seed: Random seed
            options: Optional reset options

        Returns:
            Observation and info dictionary
        """
        super().reset(seed=seed)

        # Reset episode tracking
        self.current_step = 0
        self.current_agent = 0
        self.termination_reason = None

        # Compute initial conflict from emotion and trust
        initial_conflict = (
            max(0, -self.initial_emotion) * 0.5 + (1.0 - self.initial_trust) * 0.5
        )

        # Initialize relationship state with internal states
        self.state = RelationshipState(
            emotion_level=self.initial_emotion,
            trust_level=self.initial_trust,
            conflict_intensity=initial_conflict,
            calmness_a=self.initial_calmness_a,
            calmness_b=self.initial_calmness_b,
            irritability_a=self.irritability_a,
            irritability_b=self.irritability_b,
            history_length=self.history_length,
        )
        self.initial_state = self.state.copy()

        # Get initial observation
        obs = self._get_observation()
        info = self._get_info()

        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one step in the environment.

        Args:
            action: Action index (0-9) corresponding to ActionType

        Returns:
            observation: New observation after action
            reward: Reward for the action taken
            terminated: Whether episode has ended (SUCCESS or FAILURE)
            truncated: Whether episode was truncated (max steps reached - NEUTRAL)
            info: Additional information including termination reason
        """
        if self.state is None:
            raise RuntimeError("Environment not reset. Call reset() first.")

        # Get action type
        action_type = ActionType(action)

        # Store previous state for reward calculation
        prev_state = self.state.copy()

        # Update state based on action
        self.state = self.transition_model.update_state(
            self.state, action_type, self.current_agent, self.recovery_rate
        )

        # Compute reward
        reward = self._compute_reward(prev_state, self.state, action_type)

        # Update step and agent turn
        self.current_step += 1
        self.current_agent = 1 - self.current_agent  # Alternate between A and B

        # Check termination conditions
        terminated, termination_reason = self._check_termination()
        if terminated:
            self.termination_reason = termination_reason

        truncated = self.current_step >= self.max_episode_steps
        if truncated and not terminated:
            self.termination_reason = "NEUTRAL"

        # Get observation and info
        obs = self._get_observation()
        info = self._get_info()
        if self.termination_reason:
            info["termination_reason"] = self.termination_reason

        return obs, reward, terminated, truncated, info

    def _get_observation(self) -> np.ndarray:
        """
        Get current observation based on use_history flag.
        Observation includes calmness for the current agent.
        """
        if self.use_history:
            return self.state.get_full_state(self.current_agent)
        else:
            return self.state.get_core_state_with_calmness(self.current_agent)

    def _compute_reward(
        self,
        prev_state: RelationshipState,
        curr_state: RelationshipState,
        action: ActionType,
    ) -> float:
        """
        Compute reward based on state changes and action quality.

        Reward components:
        1. Emotion improvement: Δemotion_level
        2. Trust improvement: Δtrust_level
        3. Action quality bonus: based on action type (positive/negative)

        Args:
            prev_state: State before action
            curr_state: State after action
            action: Action taken

        Returns:
            Reward value
        """
        # Immediate reward from state changes
        delta_emotion = curr_state.emotion_level - prev_state.emotion_level
        delta_trust = curr_state.trust_level - prev_state.trust_level

        immediate_reward = (
            self.reward_weights["emotion"] * delta_emotion
            + self.reward_weights["trust"] * delta_trust
        )

        # Action quality bonus
        from .actions import POSITIVE_ACTIONS, NEGATIVE_ACTIONS

        action_bonus = 0.0
        if action in POSITIVE_ACTIONS:
            action_bonus = self.reward_weights["action_bonus"]
        elif action in NEGATIVE_ACTIONS:
            action_bonus = -self.reward_weights["action_bonus"]

        return immediate_reward + action_bonus

    def _compute_final_reward(self) -> float:
        """
        Compute final episode reward based on final state values.

        Final reward formula:
        R_final = α * emotion_final + β * trust_final - γ * conflict_final

        Returns:
            Final reward value
        """
        if self.state is None:
            return 0.0

        alpha = 1.0
        beta = 1.0
        gamma = 1.0

        final_reward = (
            alpha * self.state.emotion_level
            + beta * self.state.trust_level
            - gamma * self.state.conflict_intensity
        )

        return final_reward

    def _check_termination(self) -> Tuple[bool, Optional[str]]:
        """
        Check if episode should terminate early and return termination reason.

        Termination conditions:
        1. SUCCESS: emotion > 0.7 AND trust > 0.75 (relationship repaired)
        2. FAILURE: emotion < -0.8 OR trust < 0.2 (relationship broken)
        3. NEUTRAL: max steps reached (no clear resolution)

        Returns:
            Tuple of (terminated: bool, reason: Optional[str])
        """
        if self.state is None:
            return False, None

        # Positive termination: relationship repaired
        if self.state.emotion_level > 0.7 and self.state.trust_level > 0.75:
            return True, "SUCCESS"

        # Negative termination: relationship broken
        if self.state.emotion_level < -0.8 or self.state.trust_level < 0.2:
            return True, "FAILURE"

        return False, None

    def _is_terminal(self) -> bool:
        """Legacy method for compatibility. Use _check_termination instead."""
        terminated, _ = self._check_termination()
        return terminated

    def _get_info(self) -> Dict:
        """Get additional information about current state."""
        if self.state is None:
            return {}

        return {
            "emotion": self.state.emotion_level,
            "trust": self.state.trust_level,
            "conflict": self.state.conflict_intensity,
            "calmness_a": self.state.calmness_a,
            "calmness_b": self.state.calmness_b,
            "step": self.current_step,
            "agent": "A" if self.current_agent == 0 else "B",
        }

    def render(self, mode="human"):
        """Render the environment state."""
        if self.state is None:
            return

        if mode == "human":
            print(
                f"\nStep {self.current_step} | Agent: {'A' if self.current_agent == 0 else 'B'}"
            )
            print(f"  Emotion: {self.state.emotion_level:.2f}")
            print(f"  Trust: {self.state.trust_level:.2f}")
            print(f"  Conflict: {self.state.conflict_intensity:.2f}")
            print(f"  Calmness A: {self.state.calmness_a:.2f}")
            print(f"  Calmness B: {self.state.calmness_b:.2f}")
            if self.termination_reason:
                print(f"  Termination: {self.termination_reason}")

    def get_action_feasibility(self, agent_id: int) -> np.ndarray:
        """
        Get action feasibility weights for specified agent based on their calmness.

        Args:
            agent_id: Agent ID (0 for A, 1 for B)

        Returns:
            Array of feasibility weights for each action
        """
        if self.state is None:
            return np.ones(NUM_ACTIONS) / NUM_ACTIONS

        calmness = self.state.get_calmness(agent_id)
        return self.action_feasibility.compute_feasibility(calmness)

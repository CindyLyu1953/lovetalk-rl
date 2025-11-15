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
        initial_trust: float = 0.6,
        initial_conflict: float = 0.7,
        reward_weights: Dict[str, float] = None,
        personality_a: str = "neutral",
        personality_b: str = "neutral",
    ):
        """
        Initialize relationship communication environment.

        Args:
            max_episode_steps: Maximum number of steps per episode (default: 20)
            history_length: Length of action history for deep RL (default: 10)
            use_history: Whether to include action history in state (default: False)
            initial_emotion: Initial emotion level (default: -0.3, slightly negative)
            initial_trust: Initial trust level (default: 0.6, moderate)
            initial_conflict: Initial conflict intensity (default: 0.7, high)
            reward_weights: Weights for reward components (default: balanced)
            personality_a: Personality type for agent A (default: 'neutral')
            personality_b: Personality type for agent B (default: 'neutral')
        """
        super().__init__()

        self.max_episode_steps = max_episode_steps
        self.history_length = history_length
        self.use_history = use_history
        self.personality_a = personality_a
        self.personality_b = personality_b

        # Initialize transition model
        self.transition_model = TransitionModel()

        # Reward weights (default: balanced)
        self.reward_weights = reward_weights or {
            "emotion": 1.0,
            "trust": 1.0,
            "conflict": 1.0,
            "action_bonus": 0.1,
        }

        # Episode tracking
        self.current_step = 0
        self.current_agent = 0  # 0 for A, 1 for B
        self.state: Optional[RelationshipState] = None
        self.initial_state: Optional[RelationshipState] = None

        # Initial state parameters
        self.initial_emotion = initial_emotion
        self.initial_trust = initial_trust
        self.initial_conflict = initial_conflict

        # Define action space (discrete: 10 action types)
        self.action_space = spaces.Discrete(NUM_ACTIONS)

        # Define observation space
        if use_history:
            # Full state with history
            obs_dim = 3 + history_length
            self.observation_space = spaces.Box(
                low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32
            )
        else:
            # Core state only (for shallow RL)
            self.observation_space = spaces.Box(
                low=np.array([-1.0, 0.0, 0.0]),
                high=np.array([1.0, 1.0, 1.0]),
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

        # Initialize relationship state
        self.state = RelationshipState(
            emotion_level=self.initial_emotion,
            trust_level=self.initial_trust,
            conflict_intensity=self.initial_conflict,
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
            terminated: Whether episode has ended (goal reached or conflict resolved)
            truncated: Whether episode was truncated (max steps reached)
            info: Additional information
        """
        if self.state is None:
            raise RuntimeError("Environment not reset. Call reset() first.")

        # Get action type
        action_type = ActionType(action)
        current_personality = (
            self.personality_a if self.current_agent == 0 else self.personality_b
        )

        # Store previous state for reward calculation
        prev_state = self.state.copy()

        # Update state based on action
        self.state = self.transition_model.update_state(
            self.state, action_type, current_personality
        )

        # Compute reward
        reward = self._compute_reward(prev_state, self.state, action_type)

        # Update step and agent turn
        self.current_step += 1
        self.current_agent = 1 - self.current_agent  # Alternate between A and B

        # Check termination conditions
        terminated = self._is_terminal()
        truncated = self.current_step >= self.max_episode_steps

        # Get observation and info
        obs = self._get_observation()
        info = self._get_info()

        return obs, reward, terminated, truncated, info

    def _get_observation(self) -> np.ndarray:
        """Get current observation based on use_history flag."""
        if self.use_history:
            return self.state.get_full_state()
        else:
            return self.state.get_core_state()

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
        3. Conflict reduction: -Δconflict_intensity
        4. Action quality bonus: based on action type (positive/negative)
        5. Final reward (if episode ends): weighted combination of final state

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
        delta_conflict = curr_state.conflict_intensity - prev_state.conflict_intensity

        immediate_reward = (
            self.reward_weights["emotion"] * delta_emotion
            + self.reward_weights["trust"] * delta_trust
            + self.reward_weights["conflict"] * (-delta_conflict)
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

    def _is_terminal(self) -> bool:
        """
        Check if episode should terminate early.

        Termination conditions:
        - Conflict resolved (conflict_intensity < 0.2) AND emotion > 0.5
        - Relationship broken (trust < 0.2) AND emotion < -0.8

        Returns:
            True if episode should terminate
        """
        if self.state is None:
            return False

        # Positive termination: conflict resolved and emotions positive
        if self.state.conflict_intensity < 0.2 and self.state.emotion_level > 0.5:
            return True

        # Negative termination: relationship broken
        if self.state.trust_level < 0.2 and self.state.emotion_level < -0.8:
            return True

        return False

    def _get_info(self) -> Dict:
        """Get additional information about current state."""
        if self.state is None:
            return {}

        return {
            "emotion": self.state.emotion_level,
            "trust": self.state.trust_level,
            "conflict": self.state.conflict_intensity,
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

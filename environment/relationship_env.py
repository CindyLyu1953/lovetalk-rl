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
from personality.personality_policy import PersonalityType


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
        max_episode_steps: int = 50,
        history_length: int = 10,
        use_history: bool = False,
        initial_emotion: float = -0.3,
        initial_trust: float = 0.4,
        initial_calmness_a: float = 0.4,
        initial_calmness_b: float = 0.4,
        irritability_a: float = 0.4,
        irritability_b: float = 0.4,
        recovery_rate: float = 0.02,
        reward_weights: Dict[str, float] = None,
        feasibility_alpha: float = 1.0,
        feasibility_beta: float = 1.0,
        use_deep_rl_reward: bool = False,
        termination_thresholds: Dict[str, float] = None,
        cross_agent_calmness_factor: float = 0.6,
    ):
        """
        Initialize relationship communication environment.

        Args:
            max_episode_steps: Maximum number of steps per episode (default: 50)
            history_length: Length of action history for deep RL (default: 10)
            use_history: Whether to include action history in state (default: False)
            initial_emotion: Initial emotion level (default: -0.3, slightly negative)
            initial_trust: Initial trust level (default: 0.4, lower trust for conflict scenario)
            initial_calmness_a: Initial calmness for agent A (default: 0.4, moderate)
            initial_calmness_b: Initial calmness for agent B (default: 0.4, moderate)
            irritability_a: Irritability trait for agent A (default: 0.4, moderate)
            irritability_b: Irritability trait for agent B (default: 0.4, moderate)
            recovery_rate: Automatic calmness recovery rate per step (default: 0.02)
            reward_weights: Weights for reward components (default: balanced)
            feasibility_alpha: Weight for calmness in action feasibility (default: 1.0)
            feasibility_beta: Weight for action difficulty (default: 1.0)
            cross_agent_calmness_factor: Factor for cross-agent calmness influence (default: 0.6)
                                        When agent A takes an action, agent B's calmness is
                                        affected by delta_calmness * this_factor
        """
        super().__init__()

        self.max_episode_steps = max_episode_steps
        self.history_length = history_length
        self.use_history = use_history
        self.recovery_rate = recovery_rate

        # Personalities for agent A and B (can be set to PersonalityType values or strings)
        # Keep default as NEUTRAL for backward compatibility
        self.personality_a = PersonalityType.NEUTRAL
        self.personality_b = PersonalityType.NEUTRAL

        # Initialize transition model and action feasibility
        self.transition_model = TransitionModel()
        self.action_feasibility = ActionFeasibility(
            alpha=feasibility_alpha, beta=feasibility_beta
        )

        # Reward weights (default: balanced)
        # Note: For Deep RL, use custom reward function in _compute_reward
        self.reward_weights = reward_weights or {
            "emotion": 1.0,
            "trust": 1.0,
            "action_bonus": 0.1,
        }

        # Flag to use Deep RL reward function
        self.use_deep_rl_reward = use_deep_rl_reward

        # Cross-agent calmness influence factor
        # When one agent takes an action, the other agent's calmness is affected
        # by delta_calmness * cross_agent_calmness_factor
        # This allows positive actions to help both agents recover from low calmness
        self.cross_agent_calmness_factor = cross_agent_calmness_factor

        # Termination thresholds (can be customized)
        if termination_thresholds:
            self.success_emotion_threshold = termination_thresholds.get(
                "success_emotion", 0.4
            )
            self.success_trust_threshold = termination_thresholds.get(
                "success_trust", 0.6
            )
            self.failure_emotion_threshold = termination_thresholds.get(
                "failure_emotion", -0.5
            )
            self.failure_trust_threshold = termination_thresholds.get(
                "failure_trust", 0.1
            )
        else:
            self.success_emotion_threshold = 0.4
            self.success_trust_threshold = 0.6
            self.failure_emotion_threshold = -0.5
            self.failure_trust_threshold = 0.1

        # Episode tracking
        self.current_step = 0
        self.current_agent = 0  # 0 for A, 1 for B
        self.state: Optional[RelationshipState] = None
        self.initial_state: Optional[RelationshipState] = None
        self.termination_reason: Optional[str] = None  # SUCCESS, FAILURE, NEUTRAL
        # Random generator for sampling transitions (set in reset with seed)
        self._rng = np.random.default_rng()

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
        # NEW: Added repair stage (1 dimension, normalized to [0, 1])
        if use_history:
            # Full state with history: [emotion, trust, conflict, calmness, stage] + history
            obs_dim = 5 + history_length
            self.observation_space = spaces.Box(
                low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32
            )
        else:
            # Core state only: [emotion, trust, conflict, calmness, stage]
            obs_dim = 5
            self.observation_space = spaces.Box(
                low=np.array([-1.0, 0.0, 0.0, 0.0, 0.0]),
                high=np.array([1.0, 1.0, 1.0, 1.0, 1.0]),
                dtype=np.float32,
            )

    def _compute_stage_shaping_reward(
        self, stage: int, action: ActionType, prev_state: RelationshipState
    ) -> float:
        """
        Compute stage-based reward shaping to guide agents toward contextually appropriate actions.

        This implements soft guidance that doesn't break the agent's freedom to explore:
        - Rewards contextually appropriate actions (not forced, just encouraged)
        - Small penalties for suboptimal timing (not severe punishment)
        - Aggressive actions are ALWAYS penalized across all stages

        Args:
            stage: Current repair stage (1-4)
            action: Action taken
            prev_state: State before action (for additional context)

        Returns:
            Shaping reward value (typically in range [-1.5, 1.0])
        """
        from .actions import ActionType

        shaping = 0.0

        # STAGE 1: Tension/Eruption (emotion < -0.3)
        # Reality: Emotional explosion phase - explanations/solutions backfire
        # Need: Empathy and reassurance to de-escalate
        if stage == 1:
            if action in {ActionType.EMPATHIZE, ActionType.REASSURE}:
                shaping = 1.0  # Strong positive reward for de-escalation
            elif action in {ActionType.EXPLAIN, ActionType.SUGGEST_SOLUTION}:
                shaping = -0.5  # Small penalty (too soon for problem-solving)
            elif action in {ActionType.BLAME, ActionType.DEFENSIVE}:
                shaping = -1.5  # Strong penalty (always bad, especially now)
            elif action in {ActionType.WITHDRAW, ActionType.CHANGE_TOPIC}:
                # Context-dependent: sometimes strategic to cool down
                if prev_state.conflict_intensity >= 0.7:
                    shaping = 0.3  # OK to withdraw from extreme conflict
                else:
                    shaping = -0.3  # But not ideal if conflict is manageable

        # STAGE 2: Clarification (emotion >= -0.3 and < 0, trust < 0.4)
        # Reality: Need to clear the air, explain perspectives
        # Need: Explanation and understanding
        elif stage == 2:
            if action == ActionType.EXPLAIN:
                shaping = 1.0  # Perfect time to explain
            elif action in {
                ActionType.EMPATHIZE,
                ActionType.APOLOGIZE,
                ActionType.REASSURE,
            }:
                shaping = 0.2  # Still helpful, just not the main need
            elif action == ActionType.SUGGEST_SOLUTION:
                shaping = -0.2  # Too early for solutions
            elif action in {ActionType.BLAME, ActionType.DEFENSIVE}:
                shaping = -1.0  # Destructive to clarification
            elif action in {ActionType.WITHDRAW, ActionType.CHANGE_TOPIC}:
                shaping = -0.4  # Avoidance hinders clarification

        # STAGE 3: Problem-Solving (emotion >= 0, trust < 0.6)
        # Reality: Emotion stabilized, now address the actual issues
        # Need: Solutions and concrete plans
        elif stage == 3:
            if action == ActionType.SUGGEST_SOLUTION:
                shaping = 1.0  # Perfect time for problem-solving
            elif action in {
                ActionType.EXPLAIN,
                ActionType.REASSURE,
                ActionType.ASK_FOR_NEEDS,
            }:
                shaping = 0.3  # Supportive actions
            elif action in {ActionType.APOLOGIZE, ActionType.EMPATHIZE}:
                shaping = -0.1  # Not wrong, just past that phase
            elif action in {ActionType.BLAME, ActionType.DEFENSIVE}:
                shaping = -1.0  # Regresses progress
            elif action == ActionType.WITHDRAW:
                shaping = -0.3  # Avoidance when should be solving

        # STAGE 4: Closure (emotion >= 0, trust >= 0.6)
        # Reality: Relationship stable, time for final reassurance and commitment
        # Need: Apology (if not done), check-in on needs, affirm commitment
        elif stage == 4:
            if action in {ActionType.APOLOGIZE, ActionType.ASK_FOR_NEEDS}:
                shaping = 1.0  # Perfect for closure and commitment
            elif action == ActionType.REASSURE:
                shaping = 0.5  # Good supportive action
            elif action in {ActionType.SUGGEST_SOLUTION, ActionType.EXPLAIN}:
                shaping = -0.2  # Overthinking at this point
            elif action in {ActionType.BLAME, ActionType.DEFENSIVE}:
                shaping = -1.0  # Would undo all the repair work
            elif action == ActionType.CHANGE_TOPIC:
                # Context: might be OK to move on naturally now
                shaping = 0.1  # Slight positive (natural transition)

        return shaping

    @staticmethod
    def infer_repair_stage(emotion: float, trust: float, calmness: float = None) -> int:
        """
        Infer the current repair stage based on emotion and trust levels.

        Repair stages represent different phases of conflict resolution:
        - STAGE 1 (Tension/Eruption): High negative emotion
        - STAGE 2 (Clarification): Low emotion but trust still low
        - STAGE 3 (Problem-Solving): Emotion stabilized but trust not fully recovered
        - STAGE 4 (Closure): Both emotion and trust stabilized

        Args:
            emotion: Current emotion level [-1, 1]
            trust: Current trust level [0, 1]
            calmness: Current calmness level [0, 1] (optional, not used in current logic)

        Returns:
            stage: Integer 1-4 representing the current repair stage
        """
        # STAGE 1: Tension/Eruption phase (high negative emotion)
        if emotion < -0.3:
            return 1

        # STAGE 2: Clarification phase (emotion stabilizing but trust still low OR moderate trust)
        # This covers: -0.3 <= emotion < 0 (regardless of trust)
        elif emotion < 0:
            return 2

        # STAGE 3: Problem-Solving phase (emotion positive but trust not fully recovered)
        elif emotion >= 0 and trust < 0.6:
            return 3

        # STAGE 4: Closure phase (both emotion and trust stabilized)
        else:  # emotion >= 0 and trust >= 0.6
            return 4

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

        # Initialize RNG for reproducible sampling if seed provided
        # Use numpy.default_rng for modern reproducible RNGs
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        else:
            self._rng = np.random.default_rng()

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
            reward: TEAM REWARD for training (both agents receive the same reward)
            terminated: Whether episode has ended (SUCCESS or FAILURE)
            truncated: Whether episode was truncated (max steps reached - NEUTRAL)
            info: Additional information including:
                  - termination_reason
                  - individual_reward_a: Agent A's individual reward (for evaluation)
                  - individual_reward_b: Agent B's individual reward (for evaluation)
                  - team_reward: Team reward (same as returned reward)
        """
        if self.state is None:
            raise RuntimeError("Environment not reset. Call reset() first.")

        # Get action type and store the agent who took this action
        action_type = ActionType(action)
        acting_agent_id = self.current_agent

        # Store previous state for reward calculation
        prev_state = self.state.copy()

        # Update state based on action, passing personality and RNG for sampling
        agent_personality = (
            self.personality_a if self.current_agent == 0 else self.personality_b
        )
        self.state = self.transition_model.update_state(
            self.state,
            action_type,
            self.current_agent,
            self.recovery_rate,
            personality=agent_personality,
            rng=self._rng,
            cross_agent_calmness_factor=self.cross_agent_calmness_factor,
        )

        # Update step and agent turn
        self.current_step += 1
        self.current_agent = 1 - self.current_agent  # Alternate between A and B

        # Check termination conditions
        terminated, termination_reason = self._check_termination()
        truncated = self.current_step >= self.max_episode_steps

        # Set termination reason
        if terminated:
            self.termination_reason = termination_reason
        elif truncated:
            self.termination_reason = "NEUTRAL"
            # For stalemate (truncated), we should still mark as terminal for Deep RL reward
            terminated = True  # Mark as terminated for stalemate
            termination_reason = "NEUTRAL"

        # Compute TEAM REWARD for training (both agents receive this)
        team_reward = self._compute_reward(
            prev_state, self.state, action_type, terminated, termination_reason
        )

        # Compute INDIVIDUAL REWARDS for evaluation/analysis (not used for training)
        individual_reward_acting = self._compute_individual_reward(
            acting_agent_id,
            prev_state,
            self.state,
            action_type,
            terminated,
            termination_reason,
        )

        # For the non-acting agent, assign a small shared reward (they didn't act this step)
        # This is mainly for bookkeeping in evaluation
        non_acting_agent_id = 1 - acting_agent_id
        individual_reward_non_acting = (
            0.0  # Non-acting agent gets 0 individual reward this step
        )

        # Get observation and info
        obs = self._get_observation()
        info = self._get_info()

        # Add reward information to info
        if self.termination_reason:
            info["termination_reason"] = self.termination_reason

        # Store individual rewards in info (for evaluation)
        if acting_agent_id == 0:
            info["individual_reward_a"] = float(individual_reward_acting)
            info["individual_reward_b"] = float(individual_reward_non_acting)
        else:
            info["individual_reward_a"] = float(individual_reward_non_acting)
            info["individual_reward_b"] = float(individual_reward_acting)

        info["team_reward"] = float(team_reward)
        info["acting_agent"] = acting_agent_id

        # Return team_reward as the main reward (used for training)
        return obs, team_reward, terminated, truncated, info

    def _get_observation(self) -> np.ndarray:
        """
        Get current observation based on use_history flag.
        Observation includes calmness for the current agent and repair stage.

        NEW: Observation now includes repair stage as an additional dimension.
        Stage is normalized to [0, 1] by dividing by 4.0 (since stages are 1-4).
        """
        # Get base observation
        if self.use_history:
            base_obs = self.state.get_full_state(self.current_agent)
        else:
            base_obs = self.state.get_core_state_with_calmness(self.current_agent)

        # Infer current repair stage
        stage = self.infer_repair_stage(
            self.state.emotion_level,
            self.state.trust_level,
            self.state.get_calmness(self.current_agent),
        )

        # Normalize stage to [0, 1] (stage is 1-4, so divide by 4.0)
        stage_normalized = stage / 4.0

        # Append stage to observation
        obs_with_stage = np.append(base_obs, stage_normalized).astype(np.float32)

        return obs_with_stage

    def _compute_reward(
        self,
        prev_state: RelationshipState,
        curr_state: RelationshipState,
        action: ActionType,
        is_terminal: bool = False,
        termination_reason: Optional[str] = None,
    ) -> float:
        """
        Compute reward based on state changes and action quality.

        NOTE: This method now primarily computes TEAM REWARD for training.
        For individual rewards (evaluation only), use _compute_individual_reward().

        If use_deep_rl_reward is True, uses the 4-part Deep RL reward design:
        1. Continuous state change reward
        2. Action-level reward (cooperative/aggressive/withdraw)
        3. Termination reward (success/failure/stalemate)
        4. Reward clipping to [-3.0, 3.0]

        Otherwise, uses the original reward function.

        Args:
            prev_state: State before action
            curr_state: State after action
            action: Action taken
            is_terminal: Whether episode terminated
            termination_reason: Reason for termination (SUCCESS/FAILURE/NEUTRAL)

        Returns:
            Reward value (clipped to [-3.0, 3.0] for Deep RL)
        """
        if self.use_deep_rl_reward:
            return self._compute_team_reward(
                prev_state, curr_state, action, is_terminal, termination_reason
            )

        # Original reward function for Shallow RL
        # Immediate reward from state changes
        delta_emotion = curr_state.emotion_level - prev_state.emotion_level
        delta_trust = curr_state.trust_level - prev_state.trust_level

        # Increased weight for emotion to encourage improvement from negative values
        emotion_weight = self.reward_weights["emotion"] * 1.5  # Increase emotion weight
        immediate_reward = (
            emotion_weight * delta_emotion + self.reward_weights["trust"] * delta_trust
        )

        # Bonus for crossing emotion threshold from negative to positive
        emotion_crossing_bonus = 0.0
        if prev_state.emotion_level < 0 and curr_state.emotion_level >= 0:
            emotion_crossing_bonus = 0.2  # Bonus for crossing from negative to positive

        # Action quality bonus
        from .actions import POSITIVE_ACTIONS, NEGATIVE_ACTIONS

        action_bonus = 0.0
        if action in POSITIVE_ACTIONS:
            # Increased bonus for positive actions, especially when emotion is negative
            base_bonus = self.reward_weights["action_bonus"] * 2.0  # Double the bonus
            if prev_state.emotion_level < 0:
                # Extra bonus when emotion is negative to encourage improvement
                action_bonus = base_bonus + self.reward_weights["action_bonus"]
            else:
                action_bonus = base_bonus
        elif action in NEGATIVE_ACTIONS:
            # Increased penalty for negative actions (3x instead of 1x)
            action_bonus = -self.reward_weights["action_bonus"] * 3.0

        # Termination reward for Shallow RL (increased to strongly encourage success and discourage failure)
        termination_bonus = 0.0
        if is_terminal and termination_reason:
            if termination_reason == "SUCCESS":
                termination_bonus = 4.0  # Increased from 2.0 to 4.0 (doubled)
            elif termination_reason == "FAILURE":
                termination_bonus = -4.0  # Increased from -2.0 to -4.0 (doubled)
            elif termination_reason == "NEUTRAL":
                termination_bonus = -0.2  # Small penalty for stalemate

        return (
            immediate_reward + action_bonus + emotion_crossing_bonus + termination_bonus
        )

    def _compute_team_reward(
        self,
        prev_state: RelationshipState,
        curr_state: RelationshipState,
        action: ActionType,
        is_terminal: bool = False,
        termination_reason: Optional[str] = None,
    ) -> float:
        """
        Compute Team Reward for multi-agent cooperative training.

        Based on QMIX/VDN principles: shared team reward encourages cooperation.
        Both agents receive the same reward during training to optimize joint objective.

        Design Philosophy (inspired by MARL literature):
        - Relationship repair is a pure cooperative task
        - Team reward = f(shared state changes: emotion, trust, conflict)
        - Both agents aim to maximize the same team objective

        Reward components:
        1. Continuous state change:
           - When emotion < 0: 3.0 * Δemotion + 1.0 * Δtrust - 0.5 * Δconflict (higher weight for recovery)
           - When emotion >= 0: 2.5 * Δemotion + 1.0 * Δtrust - 0.5 * Δconflict
        2. Emotion crossing bonus: +0.6 when emotion crosses from negative to positive
        3. Emotion progress bonus: Progressive reward when emotion is negative but improving (up to +0.2)
        4. Action-level: cooperative +0.20 (or +0.45 if emotion < 0), aggressive -0.20, withdraw depends on conflict
        5. Termination: success +4.0, failure -4.0, stalemate -0.2
        6. Clipping: clip to [-5.0, 5.0]

        Args:
            prev_state: State before action
            curr_state: State after action
            action: Action taken
            is_terminal: Whether episode terminated
            termination_reason: Reason for termination

        Returns:
            Team reward value clipped to [-5.0, 5.0]
        """
        from .actions import ActionType

        # 1. Continuous state change reward (increased weight for emotion improvement)
        delta_emotion = curr_state.emotion_level - prev_state.emotion_level
        delta_trust = curr_state.trust_level - prev_state.trust_level
        delta_conflict = curr_state.conflict_intensity - prev_state.conflict_intensity

        # Increased weight for emotion to strongly encourage improvement from negative values
        # Use higher weight (3.0) when emotion is negative, normal weight (2.5) otherwise
        if prev_state.emotion_level < 0:
            emotion_weight = (
                3.0  # Higher weight when emotion is negative to encourage recovery
            )
        else:
            emotion_weight = (
                2.5  # Still higher than trust to prioritize emotion improvement
            )

        r_state = (
            emotion_weight * delta_emotion + 1.0 * delta_trust - 0.5 * delta_conflict
        )

        # Bonus for crossing emotion threshold from negative to positive
        emotion_crossing_bonus = 0.0
        if prev_state.emotion_level < 0 and curr_state.emotion_level >= 0:
            emotion_crossing_bonus = 0.6  # Increased from 0.3 to 0.6 (doubled)

        # Progressive bonus for emotion improvement when still negative but getting closer to 0
        # This encourages incremental progress even before crossing to positive
        emotion_progress_bonus = 0.0
        if prev_state.emotion_level < 0 and curr_state.emotion_level < 0:
            # Reward progress toward 0 when emotion is still negative
            if delta_emotion > 0:  # Only reward positive progress
                # Bonus scales with how much progress is made
                # Max bonus when improving from very negative (e.g., -0.5) toward 0
                progress_bonus_scale = (
                    abs(prev_state.emotion_level) * 0.3
                )  # Scale with how negative it was
                emotion_progress_bonus = progress_bonus_scale * (
                    delta_emotion / 0.3
                )  # Normalize by typical improvement
                emotion_progress_bonus = min(emotion_progress_bonus, 0.2)  # Cap at 0.2

        # 2. Action-level reward (increased to encourage cooperative actions)
        COOPERATIVE_ACTIONS = {
            ActionType.APOLOGIZE,
            ActionType.EMPATHIZE,
            ActionType.REASSURE,
            ActionType.SUGGEST_SOLUTION,
            ActionType.ASK_FOR_NEEDS,
        }
        AGGRESSIVE_ACTIONS = {
            ActionType.DEFENSIVE,
            ActionType.BLAME,
        }
        WITHDRAW_ACTIONS = {
            ActionType.WITHDRAW,
            ActionType.CHANGE_TOPIC,
        }

        r_action = 0.0
        if action in COOPERATIVE_ACTIONS:
            # Increased reward for cooperative actions, especially when emotion is negative
            base_reward = 0.20  # Increased from 0.15 to encourage more positive actions
            if prev_state.emotion_level < 0:
                # Extra bonus when emotion is negative to encourage improvement
                # Increased from 0.15 to 0.25 to make positive actions even more attractive
                r_action = base_reward + 0.25  # Increased from 0.15 to 0.25
            else:
                r_action = base_reward
        elif action in AGGRESSIVE_ACTIONS:
            r_action = (
                -0.20
            )  # Increased penalty from -0.15 to discourage negative actions
        elif action in WITHDRAW_ACTIONS:
            # High conflict (>= 0.6) -> +0.02, otherwise -0.02
            if prev_state.conflict_intensity >= 0.6:
                r_action = 0.02
            else:
                r_action = -0.05  # Increased penalty from -0.02

        # 2.5. NEW: Stage-based reward shaping
        # Infer repair stage from PREVIOUS state (before action)
        stage = self.infer_repair_stage(
            prev_state.emotion_level,
            prev_state.trust_level,
            prev_state.get_calmness(self.current_agent),
        )

        r_stage_shaping = self._compute_stage_shaping_reward(stage, action, prev_state)

        # Combine action reward with stage shaping
        r_action = r_action + r_stage_shaping

        # 3. Termination reward (UPGRADED: significantly increased to strongly drive success)
        r_terminal = 0.0
        if is_terminal and termination_reason:
            if termination_reason == "SUCCESS":
                r_terminal = (
                    30.0  # UPGRADED from 20.0 to 30.0 (stronger success signal)
                )
            elif termination_reason == "FAILURE":
                r_terminal = -20.0  # UPGRADED from -4.0 to -20.0 (5x increase)
            elif termination_reason == "NEUTRAL":
                r_terminal = (
                    -10.0
                )  # UPGRADED from -0.2 to -10.0 (strong stalemate penalty)

        # 4. Total reward with clipping (UPGRADED: expanded range for larger terminal rewards)
        total_reward = (
            r_state
            + r_action
            + r_terminal
            + emotion_crossing_bonus
            + emotion_progress_bonus
        )
        return np.clip(
            total_reward, -25.0, 25.0
        )  # UPGRADED from [-5.0, 5.0] to [-25.0, 25.0]

    def _compute_individual_reward(
        self,
        agent_id: int,
        prev_state: RelationshipState,
        curr_state: RelationshipState,
        action: ActionType,
        is_terminal: bool = False,
        termination_reason: Optional[str] = None,
    ) -> float:
        """
        Compute Individual Reward for evaluation and analysis (not used for training).

        Based on COMA (Counterfactual Multi-Agent Policy Gradients) principles:
        - Quantifies individual agent's contribution to team success
        - Used for evaluation, ablation studies, and behavioral analysis
        - NOT used for training (training uses team_reward)

        Purpose:
        - Identify which agent contributes more to relationship repair
        - Analyze personality-specific cooperative/antagonistic tendencies
        - Diagnose learning issues and behavioral patterns

        Individual reward components:
        1. Action contribution bonus:
           - Cooperative actions (apologize, empathize, etc.) → positive contribution
           - Aggressive actions (blame, defensive) → negative contribution
           - Context-dependent (more bonus when emotion is negative)

        2. State improvement attribution:
           - If this agent's action aligned with positive state change → bonus
           - If this agent's action aligned with negative state change → penalty

        3. Individual termination contribution:
           - SUCCESS: both agents contributed, small bonus
           - FAILURE: identify which agent's actions were more antagonistic

        Args:
            agent_id: Agent ID (0 for A, 1 for B)
            prev_state: State before action
            curr_state: State after action
            action: Action taken by this agent
            is_terminal: Whether episode terminated
            termination_reason: Reason for termination

        Returns:
            Individual reward value (for evaluation only, not training)
        """
        from .actions import ActionType

        # Calculate state changes (shared by both agents)
        delta_emotion = curr_state.emotion_level - prev_state.emotion_level
        delta_trust = curr_state.trust_level - prev_state.trust_level
        delta_conflict = curr_state.conflict_intensity - prev_state.conflict_intensity

        # 1. Action contribution score
        COOPERATIVE_ACTIONS = {
            ActionType.APOLOGIZE,
            ActionType.EMPATHIZE,
            ActionType.REASSURE,
            ActionType.SUGGEST_SOLUTION,
            ActionType.ASK_FOR_NEEDS,
        }
        AGGRESSIVE_ACTIONS = {
            ActionType.DEFENSIVE,
            ActionType.BLAME,
        }
        WITHDRAW_ACTIONS = {
            ActionType.WITHDRAW,
            ActionType.CHANGE_TOPIC,
        }

        action_contribution = 0.0
        if action in COOPERATIVE_ACTIONS:
            # Positive contribution for cooperative actions
            action_contribution = 0.15
            if prev_state.emotion_level < 0:
                # Extra credit when cooperating during negative emotion
                action_contribution += 0.10
        elif action in AGGRESSIVE_ACTIONS:
            # Negative contribution for aggressive actions
            action_contribution = -0.15
            if prev_state.emotion_level < 0:
                # Extra penalty for being aggressive when emotion is already negative
                action_contribution -= 0.10
        elif action in WITHDRAW_ACTIONS:
            # Context-dependent
            if prev_state.conflict_intensity >= 0.6:
                action_contribution = 0.05  # Strategic withdrawal in high conflict
            else:
                action_contribution = -0.05  # Avoidance in moderate conflict

        # 2. Alignment with state improvement
        # If agent took cooperative action and state improved → bonus
        # If agent took aggressive action and state worsened → penalty
        alignment_reward = 0.0

        # Simplified alignment: did the action align with positive changes?
        positive_state_change = (
            (delta_emotion > 0) or (delta_trust > 0) or (delta_conflict < 0)
        )
        negative_state_change = (
            (delta_emotion < 0) or (delta_trust < 0) or (delta_conflict > 0)
        )

        if action in COOPERATIVE_ACTIONS and positive_state_change:
            alignment_reward = 0.10  # Agent's cooperation helped
        elif action in AGGRESSIVE_ACTIONS and negative_state_change:
            alignment_reward = -0.10  # Agent's aggression hurt
        elif action in COOPERATIVE_ACTIONS and negative_state_change:
            alignment_reward = (
                0.05  # Tried to help but didn't work (still get some credit)
            )
        elif action in AGGRESSIVE_ACTIONS and positive_state_change:
            alignment_reward = (
                -0.05
            )  # State improved despite aggression (still penalize)

        # 3. Termination contribution
        termination_contribution = 0.0
        if is_terminal and termination_reason:
            if termination_reason == "SUCCESS":
                # Both agents contributed to success
                termination_contribution = 0.5
            elif termination_reason == "FAILURE":
                # Attribute failure based on recent actions
                if action in AGGRESSIVE_ACTIONS:
                    termination_contribution = (
                        -0.5
                    )  # This agent's aggression contributed to failure
                else:
                    termination_contribution = -0.2  # Shared responsibility
            elif termination_reason == "NEUTRAL":
                termination_contribution = -0.1  # Small penalty for stalemate

        # Total individual reward
        individual_reward = (
            action_contribution + alignment_reward + termination_contribution
        )

        return np.clip(individual_reward, -2.0, 2.0)  # Smaller range than team reward

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
        1. SUCCESS: emotion > 0.2 AND trust > 0.6 (balanced repair achieved)
        2. FAILURE: emotion < failure_emotion_threshold OR trust < failure_trust_threshold
        3. NEUTRAL: max steps reached (no clear resolution - stalemate)

        Success Condition:
        - Emotion must be moderately positive (> 0.2)
        - Trust must be high (> 0.6)
        - This represents a successfully repaired relationship with balanced requirements

        Returns:
            Tuple of (terminated: bool, reason: Optional[str])
        """
        if self.state is None:
            return False, None

        # Positive termination: relationship repaired (emotion > 0.2 AND trust > 0.6)
        if self.state.emotion_level > 0.2 and self.state.trust_level > 0.6:
            return True, "SUCCESS"

        # Negative termination: relationship broken (unchanged)
        if (
            self.state.emotion_level < self.failure_emotion_threshold
            or self.state.trust_level < self.failure_trust_threshold
        ):
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

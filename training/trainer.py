"""
Multi-Agent Trainer

Implements training loops for multi-agent RL scenarios:
- Self-play: Both agents learn simultaneously
- Fixed-opponent: One agent learns against fixed opponent
"""

from typing import Optional, Dict
import numpy as np
from collections import defaultdict

from environment import RelationshipEnv


class MultiAgentTrainer:
    """
    Trainer for multi-agent relationship dynamics simulator.

    Supports different training modes:
    - Self-play: Both agents learn simultaneously (symmetric multi-agent)
    - Fixed-opponent: One agent learns against fixed opponent (baseline)
    - Asymmetric multi-agent: Both learn with different personalities
    """

    def __init__(
        self,
        env: RelationshipEnv,
        agent_a,
        agent_b,
        train_mode: str = "self_play",
        log_interval: int = 100,
        save_interval: int = 1000,
        save_dir: str = "./checkpoints",
    ):
        """
        Initialize multi-agent trainer.

        Args:
            env: Relationship communication environment
            agent_a: Agent A (first agent)
            agent_b: Agent B (second agent, can be None for fixed-opponent)
            train_mode: Training mode ('self_play', 'fixed_opponent', 'asymmetric')
            log_interval: Interval for logging statistics (default: 100)
            save_interval: Interval for saving checkpoints (default: 1000)
            save_dir: Directory to save checkpoints (default: './checkpoints')
        """
        self.env = env
        self.agent_a = agent_a
        self.agent_b = agent_b
        self.train_mode = train_mode
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.save_dir = save_dir

        # Training statistics
        self.stats = defaultdict(list)
        self.episode_rewards_a = []
        self.episode_rewards_b = []
        self.episode_lengths = []

        # Episode tracking
        self.episode_count = 0

    def train(self, num_episodes: int):
        """
        Train agents for specified number of episodes.

        Args:
            num_episodes: Number of episodes to train
        """
        print(f"Starting training: {num_episodes} episodes, mode: {self.train_mode}")

        for episode in range(num_episodes):
            episode_reward_a = 0.0
            episode_reward_b = 0.0
            episode_length = 0

            # Reset environment
            obs, info = self.env.reset()
            current_agent = 0  # 0 for A, 1 for B

            # Episode data for on-policy agents (e.g., SARSA, PPO)
            episode_data_a = {
                "states": [],
                "actions": [],
                "rewards": [],
                "log_probs": [],
            }
            episode_data_b = {
                "states": [],
                "actions": [],
                "rewards": [],
                "log_probs": [],
            }

            done = False
            truncated = False

            while not (done or truncated):
                # Select agent
                if current_agent == 0:
                    agent = self.agent_a
                    agent_id = 0
                else:
                    # Agent B's turn
                    if self.train_mode == "fixed_opponent" and self.agent_b is None:
                        # In fixed_opponent mode, use random actions for agent B
                        action = self.env.action_space.sample()
                        log_prob = None
                        # Take step with random action
                        next_obs, reward, done, truncated, info = self.env.step(action)
                        # Store reward for agent B (we track it but don't train)
                        episode_reward_b += reward
                        obs = next_obs
                        current_agent = 1 - current_agent  # Alternate
                        episode_length += 1
                        continue
                    else:
                        agent = self.agent_b
                        agent_id = 1

                # Select action
                if hasattr(agent, "select_action"):
                    # For Q-learning, DQN
                    if "ppo" in agent.__class__.__name__.lower():
                        # For PPO
                        action, log_prob = agent.select_action(obs, training=True)
                    else:
                        action = agent.select_action(obs, training=True)
                        log_prob = None
                else:
                    raise ValueError(f"Unknown agent type: {type(agent)}")

                # Store for on-policy methods
                if agent_id == 0:
                    episode_data_a["states"].append(obs.copy())
                    episode_data_a["actions"].append(action)
                    if log_prob is not None:
                        episode_data_a["log_probs"].append(log_prob)
                else:
                    episode_data_b["states"].append(obs.copy())
                    episode_data_b["actions"].append(action)
                    if log_prob is not None:
                        episode_data_b["log_probs"].append(log_prob)

                # Take step
                next_obs, reward, done, truncated, info = self.env.step(action)

                # Store reward
                if agent_id == 0:
                    episode_reward_a += reward
                    episode_data_a["rewards"].append(reward)
                else:
                    episode_reward_b += reward
                    episode_data_b["rewards"].append(reward)

                # Update agent (off-policy methods: Q-learning, SARSA)
                # Skip incremental updates for PPO (handled at episode end)
                if (
                    hasattr(agent, "update")
                    and not hasattr(agent, "store_transition")
                    and "ppo" not in agent.__class__.__name__.lower()
                ):
                    # Q-learning, SARSA (single-step update)
                    if agent_id == 0:
                        if current_agent == 0:  # A just acted, B hasn't
                            # Wait for next state after B acts
                            pass
                    else:
                        # B just acted, update with previous state
                        prev_obs = (
                            episode_data_b["states"][-2]
                            if len(episode_data_b["states"]) > 1
                            else obs
                        )
                        prev_action = (
                            episode_data_b["actions"][-2]
                            if len(episode_data_b["actions"]) > 1
                            else action
                        )
                        prev_reward = (
                            episode_data_b["rewards"][-2]
                            if len(episode_data_b["rewards"]) > 1
                            else reward
                        )

                        # For SARSA, need next action
                        if "sarsa" in agent.__class__.__name__.lower():
                            next_action = self.agent_a.select_action(
                                next_obs, training=True
                            )
                            agent.update(
                                prev_obs,
                                prev_action,
                                prev_reward,
                                obs,
                                next_action,
                                done,
                            )
                        else:
                            agent.update(prev_obs, prev_action, prev_reward, obs, done)

                # Store transition for experience replay (DQN)
                # Skip for PPO (handled at episode end)
                if (
                    hasattr(agent, "store_transition")
                    and "ppo" not in agent.__class__.__name__.lower()
                ):
                    agent.store_transition(obs, action, reward, next_obs, done)
                    # Update DQN periodically
                    if len(agent.memory) >= agent.batch_size:
                        loss = agent.update()
                        if loss is not None and agent_id == 0:
                            self.stats["loss_a"].append(loss)
                        elif loss is not None:
                            self.stats["loss_b"].append(loss)

                obs = next_obs
                current_agent = 1 - current_agent  # Alternate
                episode_length += 1

            # Update on-policy agents (PPO, SARSA at episode end)
            if hasattr(self.agent_a, "update") and len(episode_data_a["states"]) > 0:
                if "ppo" in self.agent_a.__class__.__name__.lower():
                    # PPO update
                    states = np.array(episode_data_a["states"])
                    actions = np.array(episode_data_a["actions"])
                    log_probs = np.array(episode_data_a["log_probs"])
                    rewards = np.array(episode_data_a["rewards"])
                    dones = np.array([False] * (len(rewards) - 1) + [done])
                    self.agent_a.update(states, actions, log_probs, rewards, dones)

            if (
                self.agent_b is not None
                and hasattr(self.agent_b, "update")
                and len(episode_data_b["states"]) > 0
            ):
                if "ppo" in self.agent_b.__class__.__name__.lower():
                    states = np.array(episode_data_b["states"])
                    actions = np.array(episode_data_b["actions"])
                    log_probs = np.array(episode_data_b["log_probs"])
                    rewards = np.array(episode_data_b["rewards"])
                    dones = np.array([False] * (len(rewards) - 1) + [done])
                    self.agent_b.update(states, actions, log_probs, rewards, dones)

            # Decay epsilon for exploration
            if hasattr(self.agent_a, "decay_epsilon"):
                self.agent_a.decay_epsilon()
            if self.agent_b is not None and hasattr(self.agent_b, "decay_epsilon"):
                self.agent_b.decay_epsilon()

            # Record statistics
            self.episode_rewards_a.append(episode_reward_a)
            self.episode_rewards_b.append(episode_reward_b)
            self.episode_lengths.append(episode_length)

            final_emotion = info.get("emotion", 0.0)
            final_trust = info.get("trust", 0.0)
            final_conflict = info.get("conflict", 0.0)

            self.stats["final_emotion"].append(final_emotion)
            self.stats["final_trust"].append(final_trust)
            self.stats["final_conflict"].append(final_conflict)

            self.episode_count += 1

            # Logging
            if (episode + 1) % self.log_interval == 0:
                self._log_statistics(episode + 1)

            # Save checkpoints
            if (episode + 1) % self.save_interval == 0:
                self._save_checkpoints(episode + 1)

    def _log_statistics(self, episode: int):
        """Log training statistics."""
        recent_rewards_a = self.episode_rewards_a[-self.log_interval :]
        recent_rewards_b = (
            self.episode_rewards_b[-self.log_interval :] if self.agent_b else []
        )
        recent_lengths = self.episode_lengths[-self.log_interval :]
        recent_emotion = self.stats["final_emotion"][-self.log_interval :]
        recent_trust = self.stats["final_trust"][-self.log_interval :]
        recent_conflict = self.stats["final_conflict"][-self.log_interval :]

        print(f"\nEpisode {episode}")
        print(f"  Agent A - Avg Reward: {np.mean(recent_rewards_a):.3f}")
        if recent_rewards_b:
            print(f"  Agent B - Avg Reward: {np.mean(recent_rewards_b):.3f}")
        print(f"  Avg Episode Length: {np.mean(recent_lengths):.1f}")
        print(
            f"  Final State - Emotion: {np.mean(recent_emotion):.3f}, "
            f"Trust: {np.mean(recent_trust):.3f}, "
            f"Conflict: {np.mean(recent_conflict):.3f}"
        )

    def _save_checkpoints(self, episode: int):
        """Save agent checkpoints."""
        import os

        os.makedirs(self.save_dir, exist_ok=True)

        if hasattr(self.agent_a, "save"):
            self.agent_a.save(f"{self.save_dir}/agent_a_ep{episode}.pth")

        if self.agent_b is not None and hasattr(self.agent_b, "save"):
            self.agent_b.save(f"{self.save_dir}/agent_b_ep{episode}.pth")

        print(f"Checkpoints saved at episode {episode}")

    def get_statistics(self) -> Dict:
        """Get training statistics."""
        return {
            "episode_rewards_a": self.episode_rewards_a,
            "episode_rewards_b": self.episode_rewards_b,
            "episode_lengths": self.episode_lengths,
            "stats": dict(self.stats),
        }

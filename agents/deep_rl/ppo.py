"""
Proximal Policy Optimization (PPO) Agent

Implements PPO for the relationship dynamics environment.
PPO is well-suited for multi-agent scenarios due to its stability.
"""

from typing import Optional
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from personality.personality_policy import PersonalityPolicy, PersonalityType


class ActorCritic(nn.Module):
    """
    Actor-Critic network for PPO.

    Shared feature extractor with separate heads for policy (actor) and value (critic).
    """

    def __init__(
        self, state_dim: int, action_dim: int, hidden_dims: Optional[list] = None
    ):
        """
        Initialize Actor-Critic network.

        Args:
            state_dim: Dimension of state vector
            action_dim: Number of actions
            hidden_dims: List of hidden layer dimensions (default: [128, 128])
        """
        super(ActorCritic, self).__init__()

        if hidden_dims is None:
            hidden_dims = [128, 128]

        # Shared feature extractor
        shared_layers = []
        input_dim = state_dim

        for hidden_dim in hidden_dims:
            shared_layers.extend(
                [
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                ]
            )
            input_dim = hidden_dim

        self.shared = nn.Sequential(*shared_layers)

        # Actor head (policy)
        self.actor = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[-1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[-1], action_dim),
            nn.Softmax(dim=-1),
        )

        # Critic head (value function)
        self.critic = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[-1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[-1], 1),
        )

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through network.

        Returns:
            Action probabilities and state value
        """
        features = self.shared(state)
        action_probs = self.actor(features)
        value = self.critic(features)
        return action_probs, value


class PPOAgent:
    """
    Proximal Policy Optimization agent for relationship dynamics simulator.

    PPO is an on-policy algorithm known for stability in multi-agent settings.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int = 10,
        learning_rate: float = 3e-4,
        discount_factor: float = 0.95,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        ppo_epochs: int = 4,
        batch_size: int = 64,
        personality: PersonalityType = PersonalityType.NEUTRAL,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize PPO agent.

        Args:
            state_dim: Dimension of state vector
            action_dim: Number of discrete actions (default: 10)
            learning_rate: Learning rate for optimizer (default: 3e-4)
            discount_factor: Discount factor gamma (default: 0.95)
            gae_lambda: GAE lambda parameter (default: 0.95)
            clip_epsilon: PPO clip parameter (default: 0.2)
            value_coef: Value function loss coefficient (default: 0.5)
            entropy_coef: Entropy bonus coefficient (default: 0.01)
            max_grad_norm: Maximum gradient norm for clipping (default: 0.5)
            ppo_epochs: Number of PPO update epochs (default: 4)
            batch_size: Batch size for updates (default: 64)
            personality: Personality type for this agent
            device: PyTorch device (default: cuda if available, else cpu)
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size

        # Device
        self.device = (
            device
            if device
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        # Neural network
        self.network = ActorCritic(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)

        # Personality policy
        self.personality = PersonalityPolicy(personality)

        # Training statistics
        self.training_stats = {
            "episodes": 0,
            "total_reward": 0.0,
            "action_counts": np.zeros(action_dim),
            "policy_losses": [],
            "value_losses": [],
        }

    def select_action(self, state: np.ndarray, training: bool = True):
        """
        Select action from policy distribution.

        Args:
            state: Current state vector
            training: Whether in training mode

        Returns:
            Selected action index and log probability
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            action_probs, _ = self.network(state_tensor)
            action_probs = action_probs.cpu().numpy()[0]

        # Apply personality bias to action probabilities
        for action_idx in range(self.action_dim):
            bias = self.personality.get_action_bias(action_idx)
            action_probs[action_idx] += bias

        # Normalize probabilities
        action_probs = np.clip(action_probs, 0, None)
        action_probs = action_probs / (action_probs.sum() + 1e-8)

        # Sample action
        dist = Categorical(torch.FloatTensor(action_probs))
        action = dist.sample().item()
        log_prob = dist.log_prob(torch.LongTensor([action])).item()

        return action, log_prob

    def compute_gae(
        self,
        rewards: np.ndarray,
        values: np.ndarray,
        next_value: float,
        dones: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Generalized Advantage Estimation (GAE).

        Args:
            rewards: Array of rewards
            values: Array of value estimates
            next_value: Value estimate for next state
            dones: Array of done flags

        Returns:
            Advantages and returns
        """
        advantages = np.zeros_like(rewards)
        last_gae = 0

        for t in reversed(range(len(rewards))):
            if dones[t]:
                next_value = 0

            delta = rewards[t] + self.discount_factor * next_value - values[t]
            advantages[t] = last_gae = (
                delta + self.discount_factor * self.gae_lambda * last_gae
            )
            next_value = values[t]

        returns = advantages + values
        return advantages, returns

    def update(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        old_log_probs: np.ndarray,
        rewards: np.ndarray,
        dones: np.ndarray,
        next_value: float = 0.0,
    ):
        """
        Update policy using PPO algorithm.

        Args:
            states: Array of states
            actions: Array of actions taken
            old_log_probs: Array of log probabilities from old policy
            rewards: Array of rewards
            dones: Array of done flags
            next_value: Value estimate for terminal state
        """
        # Convert to tensors
        states_tensor = torch.FloatTensor(np.array(states)).to(self.device)
        actions_tensor = torch.LongTensor(np.array(actions)).to(self.device)
        old_log_probs_tensor = torch.FloatTensor(np.array(old_log_probs)).to(
            self.device
        )

        # Get current policy and values
        with torch.no_grad():
            _, values = self.network(states_tensor)
            values = values.squeeze().cpu().numpy()

        # Compute GAE
        advantages, returns = self.compute_gae(rewards, values, next_value, dones)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        returns_tensor = torch.FloatTensor(returns).to(self.device)
        advantages_tensor = torch.FloatTensor(advantages).to(self.device)

        # PPO update epochs
        for _ in range(self.ppo_epochs):
            # Shuffle data
            indices = np.random.permutation(len(states_tensor))

            for start in range(0, len(states_tensor), self.batch_size):
                end = start + self.batch_size
                batch_indices = indices[start:end]

                # Get batch
                batch_states = states_tensor[batch_indices]
                batch_actions = actions_tensor[batch_indices]
                batch_old_log_probs = old_log_probs_tensor[batch_indices]
                batch_advantages = advantages_tensor[batch_indices]
                batch_returns = returns_tensor[batch_indices]

                # Get current policy
                action_probs, values = self.network(batch_states)
                dist = Categorical(action_probs)

                # Compute losses
                new_log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()

                # PPO clip
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = (
                    torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
                    * batch_advantages
                )
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = nn.MSELoss()(values.squeeze(), batch_returns)

                # Total loss
                loss = (
                    policy_loss
                    + self.value_coef * value_loss
                    - self.entropy_coef * entropy
                )

                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.network.parameters(), self.max_grad_norm
                )
                self.optimizer.step()

                # Update statistics
                self.training_stats["policy_losses"].append(policy_loss.item())
                self.training_stats["value_losses"].append(value_loss.item())

    def save(self, filepath: str):
        """Save agent model and parameters."""
        torch.save(
            {
                "network": self.network.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "training_stats": self.training_stats,
            },
            filepath,
        )

    def load(self, filepath: str):
        """Load agent model and parameters."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.network.load_state_dict(checkpoint["network"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.training_stats = checkpoint["training_stats"]

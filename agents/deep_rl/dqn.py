"""
Deep Q-Network (DQN) Agent

Implements DQN for the relationship dynamics environment.
Uses neural networks to approximate Q-values in continuous state space.
"""

from typing import Optional
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

from personality.personality_policy import PersonalityPolicy, PersonalityType
from environment.action_feasibility import ActionFeasibility


class QNetwork(nn.Module):
    """
    Neural network for Q-value approximation.

    Input: State vector (core state + action history)
    Output: Q-values for each action
    """

    def __init__(
        self, state_dim: int, action_dim: int, hidden_dims: Optional[list] = None
    ):
        """
        Initialize Q-network.

        Args:
            state_dim: Dimension of state vector
            action_dim: Number of actions
            hidden_dims: List of hidden layer dimensions (default: [128, 128])
        """
        super(QNetwork, self).__init__()

        if hidden_dims is None:
            hidden_dims = [128, 128]

        layers = []
        input_dim = state_dim

        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                ]
            )
            input_dim = hidden_dim

        layers.append(nn.Linear(input_dim, action_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass through network."""
        return self.network(state)


class DQNAgent:
    """
    Deep Q-Network agent for relationship dynamics simulator.

    Uses neural networks to approximate Q-values for continuous state space.
    Implements experience replay and target network for stable learning.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int = 10,
        learning_rate: float = 1e-4,
        discount_factor: float = 0.95,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01,
        batch_size: int = 64,
        memory_size: int = 10000,
        target_update_freq: int = 100,
        personality: PersonalityType = PersonalityType.NEUTRAL,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize DQN agent.

        Args:
            state_dim: Dimension of state vector
            action_dim: Number of discrete actions (default: 10)
            learning_rate: Learning rate for optimizer (default: 1e-4)
            discount_factor: Discount factor gamma (default: 0.95)
            epsilon: Initial exploration rate (default: 1.0)
            epsilon_decay: Epsilon decay rate per update (default: 0.995)
            epsilon_min: Minimum epsilon value (default: 0.01)
            batch_size: Batch size for experience replay (default: 64)
            memory_size: Size of replay buffer (default: 10000)
            target_update_freq: Frequency of target network updates (default: 100)
            personality: Personality type for this agent
            device: PyTorch device (default: cuda if available, else cpu)
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        # Device
        self.device = (
            device
            if device
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        # Neural networks
        self.q_network = QNetwork(state_dim, action_dim).to(self.device)
        self.target_network = QNetwork(state_dim, action_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)

        # Experience replay buffer
        self.memory = deque(maxlen=memory_size)

        # Personality policy
        self.personality = PersonalityPolicy(personality)

        # Action feasibility (for calmness-based action selection)
        self.action_feasibility = ActionFeasibility()

        # Training statistics
        self.update_counter = 0
        self.training_stats = {
            "episodes": 0,
            "total_reward": 0.0,
            "action_counts": np.zeros(action_dim),
            "losses": [],
        }

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        Select action using epsilon-greedy policy with calmness feasibility.

        Args:
            state: Current state vector [emotion, trust, conflict, calmness, ...]
            training: Whether in training mode (affects exploration)

        Returns:
            Selected action index
        """
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.action_dim)

        # Use Q-network to get Q-values
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_network(state_tensor).cpu().numpy()[0]

        # Apply personality bias
        for action_idx in range(self.action_dim):
            q_values[action_idx] += self.personality.get_action_bias(action_idx)

        # Get calmness from state (4th element if state includes it)
        calmness = state[3] if len(state) > 3 else 0.5

        # Apply action feasibility based on calmness
        q_values = self.action_feasibility.modify_q_values(q_values, calmness)

        return np.argmax(q_values)

    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        """
        Store transition in replay buffer.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode terminated
        """
        self.memory.append((state, action, reward, next_state, done))

    def update(self) -> Optional[float]:
        """
        Update Q-network using experience replay.

        Returns:
            Loss value if update was performed, None otherwise
        """
        if len(self.memory) < self.batch_size:
            return None

        # Sample batch from replay buffer
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Convert to tensors
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)

        # Current Q-values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))

        # Next Q-values from target network
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            next_q_values[dones] = 0.0
            target_q_values = rewards + self.discount_factor * next_q_values

        # Compute loss
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()

        # Update target network
        self.update_counter += 1
        if self.update_counter % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        # Update statistics
        loss_value = loss.item()
        self.training_stats["losses"].append(loss_value)

        return loss_value

    def save(self, filepath: str):
        """Save agent model and parameters."""
        torch.save(
            {
                "q_network": self.q_network.state_dict(),
                "target_network": self.target_network.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "epsilon": self.epsilon,
                "training_stats": self.training_stats,
            },
            filepath,
        )

    def load(self, filepath: str):
        """Load agent model and parameters."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint["q_network"])
        self.target_network.load_state_dict(checkpoint["target_network"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.epsilon = checkpoint["epsilon"]
        self.training_stats = checkpoint["training_stats"]

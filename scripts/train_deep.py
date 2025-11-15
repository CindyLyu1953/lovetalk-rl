"""
Training Script for Deep RL Agents

Train deep RL agents (DQN, PPO) in the relationship dynamics simulator.
"""

import argparse
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from environment import RelationshipEnv
from agents.deep_rl import DQNAgent, PPOAgent
from personality import PersonalityType
from training import MultiAgentTrainer


def main():
    parser = argparse.ArgumentParser(description="Train deep RL agents")
    parser.add_argument(
        "--algorithm",
        type=str,
        default="dqn",
        choices=["dqn", "ppo"],
        help="RL algorithm to use",
    )
    parser.add_argument(
        "--episodes", type=int, default=5000, help="Number of episodes to train"
    )
    parser.add_argument(
        "--personality_a",
        type=str,
        default="neutral",
        choices=["neutral", "impulsive", "sensitive", "avoidant"],
        help="Personality type for agent A",
    )
    parser.add_argument(
        "--personality_b",
        type=str,
        default="neutral",
        choices=["neutral", "impulsive", "sensitive", "avoidant"],
        help="Personality type for agent B",
    )
    parser.add_argument(
        "--train_mode",
        type=str,
        default="self_play",
        choices=["self_play", "fixed_opponent"],
        help="Training mode",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./checkpoints/deep",
        help="Directory to save checkpoints",
    )
    parser.add_argument(
        "--log_interval", type=int, default=100, help="Logging interval"
    )
    parser.add_argument(
        "--save_interval", type=int, default=1000, help="Checkpoint save interval"
    )
    parser.add_argument(
        "--history_length", type=int, default=10, help="Length of action history"
    )

    args = parser.parse_args()

    # Create environment with new parameters
    # Fixed: Use moderate initial state to prevent immediate termination
    # Note: emotion is negative to reflect conflict scenario, but not too negative
    env = RelationshipEnv(
        max_episode_steps=20,
        use_history=True,  # Deep RL uses history
        history_length=args.history_length,
        initial_emotion=-0.2,  # Slightly negative (conflict scenario, but not too severe)
        initial_trust=0.6,  # Moderate trust (was 0.5, too low)
        initial_calmness_a=0.6,  # More calm (was 0.4, too low)
        initial_calmness_b=0.6,  # More calm (was 0.4, too low)
        irritability_a=0.7 if args.personality_a == "impulsive" else 0.4,
        irritability_b=0.7 if args.personality_b == "impulsive" else 0.4,
        recovery_rate=0.02,
    )

    # Debug: Print initial state
    test_obs, test_info = env.reset()
    print(f"\nInitial Environment State:")
    print(f"  Emotion: {test_info['emotion']:.3f}")
    print(f"  Trust: {test_info['trust']:.3f}")
    print(f"  Conflict: {test_info['conflict']:.3f}")
    print(f"  Calmness A: {test_info['calmness_a']:.3f}")
    print(f"  Calmness B: {test_info['calmness_b']:.3f}")
    terminated, reason = env._check_termination()
    print(f"  Initial Termination Check: {terminated}, Reason: {reason}")
    print()

    # Get state dimension
    obs, _ = env.reset()
    state_dim = len(obs)

    # Create agents
    personality_a = PersonalityType[args.personality_a.upper()]
    personality_b = PersonalityType[args.personality_b.upper()]

    if args.algorithm == "dqn":
        agent_a = DQNAgent(
            state_dim=state_dim,
            action_dim=10,
            personality=personality_a,
        )
        agent_b = (
            DQNAgent(
                state_dim=state_dim,
                action_dim=10,
                personality=personality_b,
            )
            if args.train_mode == "self_play"
            else None
        )
    elif args.algorithm == "ppo":
        agent_a = PPOAgent(
            state_dim=state_dim,
            action_dim=10,
            personality=personality_a,
        )
        agent_b = (
            PPOAgent(
                state_dim=state_dim,
                action_dim=10,
                personality=personality_b,
            )
            if args.train_mode == "self_play"
            else None
        )
    else:
        raise ValueError(f"Unknown algorithm: {args.algorithm}")

    # Create trainer
    trainer = MultiAgentTrainer(
        env=env,
        agent_a=agent_a,
        agent_b=agent_b,
        train_mode=args.train_mode,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        save_dir=args.save_dir,
    )

    # Train
    print(f"Training {args.algorithm} agents...")
    print(f"  Personality A: {args.personality_a}")
    print(f"  Personality B: {args.personality_b}")
    print(f"  Training mode: {args.train_mode}")
    print(f"  Episodes: {args.episodes}")
    print(f"  State dimension: {state_dim}")

    trainer.train(args.episodes)

    print("\nTraining completed!")


if __name__ == "__main__":
    main()

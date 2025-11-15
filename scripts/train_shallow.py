"""
Training Script for Shallow RL Agents

Train tabular RL agents (Q-learning, SARSA) in the relationship dynamics simulator.
"""

import argparse
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from environment import RelationshipEnv
from agents.shallow_rl import QLearningAgent, SarsaAgent
from personality import PersonalityType
from training import MultiAgentTrainer


def main():
    parser = argparse.ArgumentParser(description="Train shallow RL agents")
    parser.add_argument(
        "--algorithm",
        type=str,
        default="q_learning",
        choices=["q_learning", "sarsa"],
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
        default="./checkpoints/shallow",
        help="Directory to save checkpoints",
    )
    parser.add_argument(
        "--log_interval", type=int, default=100, help="Logging interval"
    )
    parser.add_argument(
        "--save_interval", type=int, default=1000, help="Checkpoint save interval"
    )

    args = parser.parse_args()

    # Create environment
    env = RelationshipEnv(
        max_episode_steps=20,
        use_history=False,  # Shallow RL doesn't use history
        personality_a=args.personality_a,
        personality_b=args.personality_b,
    )

    # Create agents
    personality_a = PersonalityType[args.personality_a.upper()]
    personality_b = PersonalityType[args.personality_b.upper()]

    if args.algorithm == "q_learning":
        agent_a = QLearningAgent(
            num_actions=10,
            personality=personality_a,
        )
        agent_b = (
            QLearningAgent(
                num_actions=10,
                personality=personality_b,
            )
            if args.train_mode == "self_play"
            else None
        )
    elif args.algorithm == "sarsa":
        agent_a = SarsaAgent(
            num_actions=10,
            personality=personality_a,
        )
        agent_b = (
            SarsaAgent(
                num_actions=10,
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

    trainer.train(args.episodes)

    print("\nTraining completed!")


if __name__ == "__main__":
    main()

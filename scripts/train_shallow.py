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
        choices=[
            "neutral",
            "impulsive",
            "sensitive",
            "avoidant",
            "neurotic",
            "agreeable",
            "conscientious",
        ],
        help="Personality type for agent A",
    )
    parser.add_argument(
        "--personality_b",
        type=str,
        default="neutral",
        choices=[
            "neutral",
            "impulsive",
            "sensitive",
            "avoidant",
            "neurotic",
            "agreeable",
            "conscientious",
        ],
        help="Personality type for agent B",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=10,
        help="Number of independent repeats (different seeds) for shallow RL (default: 10)",
    )
    parser.add_argument(
        "--base_seed",
        type=int,
        default=42,
        help="Base random seed for repeats (default: 42)",
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

    # Create environment template (we will recreate per repeat to ensure fresh training)
    # Fixed: Use moderate initial state to prevent immediate termination
    # Note: emotion is negative to reflect conflict scenario, but not too negative
    repeats = args.repeats
    base_seed = args.base_seed

    # Convert personalities
    personality_a = PersonalityType[args.personality_a.upper()]
    personality_b = PersonalityType[args.personality_b.upper()]

    for run_idx in range(repeats):
        run_seed = base_seed + run_idx
        run_save_dir = Path(args.save_dir) / f"run_{run_idx+1}"
        run_save_dir.mkdir(parents=True, exist_ok=True)

        env = RelationshipEnv(
            max_episode_steps=50,
            use_history=False,  # Shallow RL doesn't use history
            initial_emotion=-0.2,  # Slightly negative (conflict scenario, but not too severe)
            initial_trust=0.6,  # Moderate trust (was 0.5, too low)
            initial_calmness_a=0.6,  # More calm (was 0.4, too low)
            initial_calmness_b=0.6,  # More calm (was 0.4, too low)
            irritability_a=0.7 if args.personality_a == "impulsive" else 0.4,
            irritability_b=0.7 if args.personality_b == "impulsive" else 0.4,
            recovery_rate=0.02,
        )

        # Set personalities on environment for sampling
        env.personality_a = personality_a
        env.personality_b = personality_b

        # Debug: Print initial state for this run
        test_obs, test_info = env.reset(seed=run_seed)
        print(
            f"\n[Run {run_idx+1}/{repeats}] Initial Environment State (seed={run_seed}):"
        )
        print(f"  Emotion: {test_info['emotion']:.3f}")
        print(f"  Trust: {test_info['trust']:.3f}")
        print(f"  Conflict: {test_info['conflict']:.3f}")
        print(f"  Calmness A: {test_info['calmness_a']:.3f}")
        print(f"  Calmness B: {test_info['calmness_b']:.3f}")
        terminated, reason = env._check_termination()
        print(f"  Initial Termination Check: {terminated}, Reason: {reason}")
        print()

        # Create agents (fresh for each run)
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
            save_dir=str(run_save_dir),
        )

        # Train
        print(f"Training {args.algorithm} agents (run {run_idx+1}/{repeats})...")
        print(f"  Personality A: {args.personality_a}")
        print(f"  Personality B: {args.personality_b}")
        print(f"  Training mode: {args.train_mode}")
        print(f"  Episodes: {args.episodes}")

        trainer.train(args.episodes, initial_seed=run_seed)

        # Save per-run statistics
        import json

        stats = trainer.get_statistics()
        with open(Path(run_save_dir) / "train_stats.json", "w") as f:
            json.dump(
                {"seed": run_seed, "stats": stats},
                f,
                default=lambda o: list(o) if hasattr(o, "__iter__") else o,
            )

        print(f"[Run {run_idx+1}] Training completed and stats saved to {run_save_dir}")


if __name__ == "__main__":
    main()

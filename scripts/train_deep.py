"""
Deep RL Training Script for 5 Conflict Scenarios

Train DQN agents with optimized parameters for human-like conflict resolution:
- D1: neutral × neutral (baseline)
- D2: neurotic × agreeable (neurotic vs agreeable)
- D3: neurotic × neurotic (extreme neurotic conflict)
- D4: neutral × avoidant (cold war)
- D5: agreeable × conscientious (cooperative scenario)
"""

import argparse
import sys
import os
from pathlib import Path
from typing import Optional
import json
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from environment import RelationshipEnv
from agents.deep_rl import DQNAgent
from personality import PersonalityType
from training import MultiAgentTrainer


# Optimized DQN hyperparameters for conflict resolution
DQN_CONFIG = {
    "learning_rate": 3e-4,
    "discount_factor": 0.99,
    "epsilon": 1.0,
    "epsilon_decay": 0.998,  # UPDATED: Slower decay for more exploration (was 0.997)
    "epsilon_min": 0.1,  # UPDATED: Higher minimum for sustained exploration (was 0.05)
    "batch_size": 64,
    "memory_size": 20000,
    "target_update_freq": 200,
}

# Termination thresholds (UPDATED to match latest environment upgrade)
TERMINATION_THRESHOLDS = {
    "success_emotion": 0.2,  # Balanced repair (emotion > 0.2)
    "success_trust": 0.6,  # High trust recovery (trust > 0.6)
    "failure_emotion": -0.5,  # Extreme conflict (emotion < -0.5)
    "failure_trust": 0.1,  # Very low trust (trust < 0.1)
}

# Training configuration
TRAINING_CONFIG = {
    "num_episodes": 8000,
    "train_mode": "self_play",
    "log_interval": 200,
    "save_interval": 2000,
}


def get_experiment_config(exp_id: str):
    """
    Get configuration for a specific experiment.

    Args:
        exp_id: Experiment ID (D1, D2, D3, D4, D5)

    Returns:
        Dictionary with experiment configuration
    """
    configs = {
        "D1": {
            "personality_a": "neutral",
            "personality_b": "neutral",
            "description": "Baseline (neutral vs neutral)",
            "irritability_a": 0.4,
            "irritability_b": 0.4,
        },
        "D2": {
            "personality_a": "neurotic",
            "personality_b": "agreeable",
            "description": "Neurotic vs agreeable (conflict scenario)",
            "irritability_a": 0.5,  # Neurotic: moderate-high irritability
            "irritability_b": 0.3,  # Agreeable: lower irritability
        },
        "D3": {
            "personality_a": "neurotic",
            "personality_b": "neurotic",
            "description": "Extreme neurotic conflict (neurotic vs neurotic)",
            "irritability_a": 0.5,
            "irritability_b": 0.5,
        },
        "D4": {
            "personality_a": "neutral",
            "personality_b": "avoidant",
            "description": "Cold war (neutral vs avoidant)",
            "irritability_a": 0.4,
            "irritability_b": 0.3,  # Avoidant: lower irritability, but withdraws
        },
        "D5": {
            "personality_a": "agreeable",
            "personality_b": "conscientious",
            "description": "Cooperative scenario (agreeable vs conscientious)",
            "irritability_a": 0.3,
            "irritability_b": 0.3,
        },
    }

    if exp_id not in configs:
        raise ValueError(
            f"Unknown experiment ID: {exp_id}. Must be one of: {list(configs.keys())}"
        )

    return configs[exp_id]


def train_experiment(exp_id: str, save_dir: str, num_episodes: Optional[int] = None):
    """
    Train a single experiment.

    Args:
        exp_id: Experiment ID (D1-D5)
        save_dir: Directory to save checkpoints
        num_episodes: Number of episodes to train (if None, uses TRAINING_CONFIG default)
    """
    config = get_experiment_config(exp_id)

    # Use provided num_episodes or default from TRAINING_CONFIG
    episodes_to_train = (
        num_episodes if num_episodes is not None else TRAINING_CONFIG["num_episodes"]
    )
    print(f"\n{'='*80}")
    print(f"Training Experiment {exp_id}: {config['description']}")
    print(f"{'='*80}\n")

    # Training repeats and seeding (Deep RL: default 15 repeats)
    repeats = 15
    base_seed = 42

    # Run multiple independent training repeats (each with its own seed and fresh agents)
    for run_idx in range(repeats):
        run_seed = base_seed + run_idx
        run_save_dir = Path(save_dir) / f"run_{run_idx+1}"
        run_save_dir.mkdir(parents=True, exist_ok=True)

        # Create environment with Deep RL reward and LATEST UPGRADE configuration
        env = RelationshipEnv(
            max_episode_steps=50,
            use_history=True,  # Deep RL uses history
            history_length=10,
            initial_emotion=-0.3,  # UPDATED: Conflict scenario (matches environment default)
            initial_trust=0.4,  # UPDATED: Lower trust for challenging scenario
            initial_calmness_a=0.4,  # UPDATED: Moderate calmness (matches environment default)
            initial_calmness_b=0.4,
            irritability_a=config["irritability_a"],
            irritability_b=config["irritability_b"],
            recovery_rate=0.02,
            use_deep_rl_reward=True,  # Enable Deep RL reward function
            termination_thresholds=TERMINATION_THRESHOLDS,
        )

        # Debug: Print initial state for this run
        test_obs, test_info = env.reset(seed=run_seed)
        print(
            f"[Run {run_idx+1}/{repeats}] Initial Environment State (seed={run_seed}):"
        )
        print(f"  Emotion: {test_info['emotion']:.3f}")
        print(f"  Trust: {test_info['trust']:.3f}")
        print(f"  Conflict: {test_info['conflict']:.3f}")
        print(f"  Calmness A: {test_info['calmness_a']:.3f}")
        print(f"  Calmness B: {test_info['calmness_b']:.3f}")
        terminated, reason = env._check_termination()
        print(f"  Initial Termination Check: {terminated}, Reason: {reason}")
        print()

        # Get state dimension
        obs, _ = env.reset(seed=run_seed)
        state_dim = len(obs)

        # Create agents with optimized DQN parameters
        personality_a = PersonalityType[config["personality_a"].upper()]
        personality_b = PersonalityType[config["personality_b"].upper()]

        # Set personalities on environment so transition model can sample personality-specific ranges
        env.personality_a = personality_a
        env.personality_b = personality_b

        agent_a = DQNAgent(
            state_dim=state_dim,
            action_dim=10,
            personality=personality_a,
            learning_rate=DQN_CONFIG["learning_rate"],
            discount_factor=DQN_CONFIG["discount_factor"],
            epsilon=DQN_CONFIG["epsilon"],
            epsilon_decay=DQN_CONFIG["epsilon_decay"],
            epsilon_min=DQN_CONFIG["epsilon_min"],
            batch_size=DQN_CONFIG["batch_size"],
            memory_size=DQN_CONFIG["memory_size"],
            target_update_freq=DQN_CONFIG["target_update_freq"],
        )

        agent_b = DQNAgent(
            state_dim=state_dim,
            action_dim=10,
            personality=personality_b,
            learning_rate=DQN_CONFIG["learning_rate"],
            discount_factor=DQN_CONFIG["discount_factor"],
            epsilon=DQN_CONFIG["epsilon"],
            epsilon_decay=DQN_CONFIG["epsilon_decay"],
            epsilon_min=DQN_CONFIG["epsilon_min"],
            batch_size=DQN_CONFIG["batch_size"],
            memory_size=DQN_CONFIG["memory_size"],
            target_update_freq=DQN_CONFIG["target_update_freq"],
        )

        # Create trainer (per-run save_dir)
        trainer = MultiAgentTrainer(
            env=env,
            agent_a=agent_a,
            agent_b=agent_b,
            train_mode=TRAINING_CONFIG["train_mode"],
            log_interval=TRAINING_CONFIG["log_interval"],
            save_interval=TRAINING_CONFIG["save_interval"],
            save_dir=str(run_save_dir),
        )

        # Print run configuration
        print(f"[Run {run_idx+1}/{repeats}] Training Configuration:")
        print(f"  Algorithm: DQN (optimized for conflict resolution)")
        print(f"  Personality A: {config['personality_a']}")
        print(f"  Personality B: {config['personality_b']}")
        print(f"  Training mode: {TRAINING_CONFIG['train_mode']}")
        print(f"  Episodes: {episodes_to_train}")
        print(f"  State dimension: {state_dim}")
        print(f"  Seed: {run_seed}")

        trainer.train(episodes_to_train, initial_seed=run_seed)

        # Save per-run statistics for later aggregation
        import json

        stats = trainer.get_statistics()
        # Convert numpy arrays to lists where necessary
        stats_serializable = {}
        for k, v in stats.items():
            if k == "detailed_episodes":
                # Skip detailed_episodes in train_stats.json (already saved separately)
                continue
            elif isinstance(v, list):
                # Only convert numeric values, skip dicts and other complex types
                stats_serializable[k] = [
                    (
                        float(x)
                        if isinstance(x, (int, float, np.number))
                        and not isinstance(x, (dict, list))
                        else (list(x) if isinstance(x, np.ndarray) else x)
                    )
                    for x in v
                ]
            else:
                stats_serializable[k] = v

        with open(run_save_dir / "train_stats.json", "w") as f:
            json.dump({"seed": run_seed, "stats": stats_serializable}, f, indent=2)

        print(f"[Run {run_idx+1}] Training completed and stats saved to {run_save_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Train optimized Deep RL (DQN) agents for 5 conflict scenarios"
    )
    parser.add_argument(
        "--experiment",
        type=str,
        required=False,
        choices=["D1", "D2", "D3", "D4", "D5"],
        help="Experiment ID (D1-D5). Required if --all is not specified.",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./experiments",
        help="Base directory to save checkpoints (default: ./experiments)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Train all 5 experiments sequentially",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help="Number of episodes to train (overrides default 8000). Use this to quickly test with fewer episodes.",
    )

    args = parser.parse_args()

    # Validate arguments
    if not args.all and not args.experiment:
        parser.error("Either --experiment or --all must be specified")

    if args.all and args.experiment:
        parser.error("Cannot specify both --experiment and --all")

    if args.all:
        # Train all 5 experiments
        experiments = ["D1", "D2", "D3", "D4", "D5"]
        for exp_id in experiments:
            save_dir = Path(args.save_dir) / exp_id / "checkpoints"
            save_dir.mkdir(parents=True, exist_ok=True)
            train_experiment(exp_id, str(save_dir), args.episodes)
            print("\n" + "=" * 80 + "\n")
    else:
        # Train single experiment
        save_dir = Path(args.save_dir) / args.experiment / "checkpoints"
        save_dir.mkdir(parents=True, exist_ok=True)
        train_experiment(args.experiment, str(save_dir), args.episodes)

    print("\nAll training completed!")


if __name__ == "__main__":
    main()

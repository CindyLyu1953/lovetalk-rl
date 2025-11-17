"""
Deep RL Training Script for 5 Conflict Scenarios

Train DQN agents with optimized parameters for human-like conflict resolution:
- D1: neutral × neutral (baseline)
- D2: impulsive × sensitive (intense conflict)
- D3: impulsive × impulsive (extreme conflict)
- D4: neutral × avoidant (cold war)
- D5: sensitive × sensitive (mutual misunderstanding)
"""

import argparse
import sys
import os
from pathlib import Path

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
    "epsilon_decay": 0.997,
    "epsilon_min": 0.05,
    "batch_size": 64,
    "memory_size": 20000,
    "target_update_freq": 200,
}

# Termination thresholds for moderate repair
TERMINATION_THRESHOLDS = {
    "success_emotion": 0.2,  # Moderate repair (from negative to slightly positive)
    "success_trust": 0.6,  # Moderate trust recovery
    "failure_emotion": -0.9,  # Extreme conflict
    "failure_trust": 0.1,  # Very low trust
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
            "personality_a": "impulsive",
            "personality_b": "sensitive",
            "description": "Intense conflict (impulsive vs sensitive)",
            "irritability_a": 0.7,  # Impulsive: high irritability
            "irritability_b": 0.5,  # Sensitive: moderate-high irritability
        },
        "D3": {
            "personality_a": "impulsive",
            "personality_b": "impulsive",
            "description": "Extreme conflict (impulsive vs impulsive)",
            "irritability_a": 0.7,
            "irritability_b": 0.7,
        },
        "D4": {
            "personality_a": "neutral",
            "personality_b": "avoidant",
            "description": "Cold war (neutral vs avoidant)",
            "irritability_a": 0.4,
            "irritability_b": 0.3,  # Avoidant: lower irritability, but withdraws
        },
        "D5": {
            "personality_a": "sensitive",
            "personality_b": "sensitive",
            "description": "Mutual misunderstanding (sensitive vs sensitive)",
            "irritability_a": 0.5,
            "irritability_b": 0.5,
        },
    }

    if exp_id not in configs:
        raise ValueError(
            f"Unknown experiment ID: {exp_id}. Must be one of: {list(configs.keys())}"
        )

    return configs[exp_id]


def train_experiment(exp_id: str, save_dir: str):
    """
    Train a single experiment.

    Args:
        exp_id: Experiment ID (D1-D5)
        save_dir: Directory to save checkpoints
    """
    config = get_experiment_config(exp_id)
    print(f"\n{'='*80}")
    print(f"Training Experiment {exp_id}: {config['description']}")
    print(f"{'='*80}\n")

    # Create environment with Deep RL reward and optimized termination
    env = RelationshipEnv(
        max_episode_steps=20,
        use_history=True,  # Deep RL uses history
        history_length=10,
        initial_emotion=-0.2,  # Slightly negative (conflict scenario)
        initial_trust=0.6,  # Moderate trust
        initial_calmness_a=0.6,  # More calm (prevents immediate termination)
        initial_calmness_b=0.6,
        irritability_a=config["irritability_a"],
        irritability_b=config["irritability_b"],
        recovery_rate=0.02,
        use_deep_rl_reward=True,  # Enable Deep RL reward function
        termination_thresholds=TERMINATION_THRESHOLDS,
    )

    # Debug: Print initial state
    test_obs, test_info = env.reset()
    print(f"Initial Environment State:")
    print(f"  Emotion: {test_info['emotion']:.3f}")
    print(f"  Trust: {test_info['trust']:.3f}")
    print(f"  Conflict: {test_info['conflict']:.3f}")
    print(f"  Calmness A: {test_info['calmness_a']:.3f}")
    print(f"  Calmness B: {test_info['calmness_b']:.3f}")
    terminated, reason = env._check_termination()
    print(f"  Initial Termination Check: {terminated}, Reason: {reason}")
    print(
        f"  Success threshold: emotion > {env.success_emotion_threshold}, trust > {env.success_trust_threshold}"
    )
    print(
        f"  Failure threshold: emotion < {env.failure_emotion_threshold} OR trust < {env.failure_trust_threshold}"
    )
    print()

    # Get state dimension
    obs, _ = env.reset()
    state_dim = len(obs)

    # Create agents with optimized DQN parameters
    personality_a = PersonalityType[config["personality_a"].upper()]
    personality_b = PersonalityType[config["personality_b"].upper()]

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

    # Create trainer
    trainer = MultiAgentTrainer(
        env=env,
        agent_a=agent_a,
        agent_b=agent_b,
        train_mode=TRAINING_CONFIG["train_mode"],
        log_interval=TRAINING_CONFIG["log_interval"],
        save_interval=TRAINING_CONFIG["save_interval"],
        save_dir=save_dir,
    )

    # Train
    print(f"Training Configuration:")
    print(f"  Algorithm: DQN (optimized for conflict resolution)")
    print(f"  Personality A: {config['personality_a']}")
    print(f"  Personality B: {config['personality_b']}")
    print(f"  Training mode: {TRAINING_CONFIG['train_mode']}")
    print(f"  Episodes: {TRAINING_CONFIG['num_episodes']}")
    print(f"  State dimension: {state_dim}")
    print(f"\nDQN Hyperparameters:")
    for key, value in DQN_CONFIG.items():
        print(f"  {key}: {value}")
    print()

    trainer.train(TRAINING_CONFIG["num_episodes"])

    print(f"\n[OK] Training completed!")


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
            train_experiment(exp_id, str(save_dir))
            print("\n" + "=" * 80 + "\n")
    else:
        # Train single experiment
        save_dir = Path(args.save_dir) / args.experiment / "checkpoints"
        save_dir.mkdir(parents=True, exist_ok=True)
        train_experiment(args.experiment, str(save_dir))

    print("\nAll training completed!")


if __name__ == "__main__":
    main()

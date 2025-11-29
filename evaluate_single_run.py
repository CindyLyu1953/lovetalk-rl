"""
Quick evaluation script for a single training run.
"""

import sys
import os
import json
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from environment import RelationshipEnv
from agents.deep_rl import DQNAgent
from personality import PersonalityType
from training import Evaluator


def evaluate_run(checkpoint_dir, exp_config, num_episodes=100):
    """Evaluate a single training run."""
    checkpoint_dir = Path(checkpoint_dir)

    # Find checkpoint files
    agent_a_path = checkpoint_dir / "agent_a_ep8000.pth"
    agent_b_path = checkpoint_dir / "agent_b_ep8000.pth"

    if not agent_a_path.exists():
        print(f"Error: Agent A checkpoint not found: {agent_a_path}")
        return None

    if not agent_b_path.exists():
        print(f"Error: Agent B checkpoint not found: {agent_b_path}")
        return None

    print(f"Evaluating checkpoint: {checkpoint_dir}")
    print(f"  Personality A: {exp_config['personality_a']}")
    print(f"  Personality B: {exp_config['personality_b']}")
    print(f"  Episodes: {num_episodes}")
    print()

    # Termination thresholds (MUST match training configuration!)
    TERMINATION_THRESHOLDS = {
        "success_emotion": 0.2,  # Balanced repair (emotion > 0.2)
        "success_trust": 0.6,  # High trust recovery (trust > 0.6)
        "failure_emotion": -0.5,  # Extreme conflict (emotion < -0.5)
        "failure_trust": 0.1,  # Very low trust (trust < 0.1)
    }
    
    # Create environment (UPDATED to match latest training configuration!)
    env = RelationshipEnv(
        use_deep_rl_reward=True,
        max_episode_steps=50,
        use_history=True,  # Must match training
        history_length=10,  # Must match training
        initial_emotion=-0.3,  # UPDATED: Match latest training config
        initial_trust=0.4,  # UPDATED: Match latest training config
        initial_calmness_a=0.4,  # UPDATED: Match latest training config
        initial_calmness_b=0.4,  # UPDATED: Match latest training config
        irritability_a=exp_config["irritability_a"],
        irritability_b=exp_config["irritability_b"],
        termination_thresholds=TERMINATION_THRESHOLDS,  # CRITICAL: Pass thresholds!
    )

    # Get observation dimension
    obs, _ = env.reset(seed=42)
    state_dim = obs.shape[0]

    # Create agents
    personality_a = PersonalityType[exp_config["personality_a"].upper()]
    personality_b = PersonalityType[exp_config["personality_b"].upper()]

    agent_a = DQNAgent(
        state_dim=state_dim,
        action_dim=10,
        learning_rate=3e-4,
        discount_factor=0.99,
        epsilon=0.0,  # No exploration during evaluation
        personality=personality_a,
    )

    agent_b = DQNAgent(
        state_dim=state_dim,
        action_dim=10,
        learning_rate=3e-4,
        discount_factor=0.99,
        epsilon=0.0,
        personality=personality_b,
    )

    # Load checkpoints
    agent_a.load(str(agent_a_path))
    agent_b.load(str(agent_b_path))

    print("Agents loaded successfully!")
    print()

    # Create evaluator
    evaluator = Evaluator(env)

    # Run evaluation
    print(f"Running {num_episodes} evaluation episodes...")
    results = evaluator.evaluate_multiple_episodes(
        agent_a, agent_b, num_episodes=num_episodes
    )

    # Print results
    evaluator.print_evaluation_results(results)

    # Save results
    output_file = checkpoint_dir / "evaluation_results.json"

    # Convert numpy types for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {
                (
                    str(k) if isinstance(k, (np.integer, np.int64, np.int32)) else k
                ): convert_to_serializable(v)
                for k, v in obj.items()
            }
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        return obj

    results_serializable = convert_to_serializable(results)

    with open(output_file, "w") as f:
        json.dump(results_serializable, f, indent=2)

    print(f"\nResults saved to: {output_file}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate a single training run")
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        required=True,
        help="Directory containing agent checkpoints",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        required=True,
        choices=["D1", "D2", "D3", "D4", "D5"],
        help="Experiment ID",
    )
    parser.add_argument(
        "--num_episodes", type=int, default=100, help="Number of evaluation episodes"
    )

    args = parser.parse_args()

    # Experiment configurations
    EXPERIMENT_CONFIGS = {
        "D1": {
            "personality_a": "neutral",
            "personality_b": "neutral",
            "irritability_a": 0.4,
            "irritability_b": 0.4,
        },
        "D2": {
            "personality_a": "neurotic",
            "personality_b": "agreeable",
            "irritability_a": 0.5,
            "irritability_b": 0.3,
        },
        "D3": {
            "personality_a": "neurotic",
            "personality_b": "neurotic",
            "irritability_a": 0.5,
            "irritability_b": 0.5,
        },
        "D4": {
            "personality_a": "neutral",
            "personality_b": "avoidant",
            "irritability_a": 0.4,
            "irritability_b": 0.3,
        },
        "D5": {
            "personality_a": "agreeable",
            "personality_b": "conscientious",
            "irritability_a": 0.3,
            "irritability_b": 0.3,
        },
    }

    exp_config = EXPERIMENT_CONFIGS[args.experiment]
    evaluate_run(args.checkpoint_dir, exp_config, args.num_episodes)

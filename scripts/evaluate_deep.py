"""
Evaluation Script for Deep RL Agents

Evaluate trained Deep RL agents (DQN) in the relationship dynamics simulator.
Supports both single experiment and batch evaluation for all 5 Deep RL scenarios (D1-D5).
"""

import argparse
import sys
import os
import json
import numpy as np
from pathlib import Path
from collections import defaultdict

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from environment import RelationshipEnv
from agents.deep_rl import DQNAgent
from personality import PersonalityType
from training import Evaluator


# Experiment configurations (must match train_deep_optimized.py)
EXPERIMENT_CONFIGS = {
    "D1": {
        "personality_a": "neutral",
        "personality_b": "neutral",
        "irritability_a": 0.4,
        "irritability_b": 0.4,
    },
    "D2": {
        "personality_a": "impulsive",
        "personality_b": "sensitive",
        "irritability_a": 0.7,
        "irritability_b": 0.5,
    },
    "D3": {
        "personality_a": "impulsive",
        "personality_b": "impulsive",
        "irritability_a": 0.7,
        "irritability_b": 0.7,
    },
    "D4": {
        "personality_a": "neutral",
        "personality_b": "avoidant",
        "irritability_a": 0.4,
        "irritability_b": 0.3,
    },
    "D5": {
        "personality_a": "sensitive",
        "personality_b": "sensitive",
        "irritability_a": 0.5,
        "irritability_b": 0.5,
    },
}


def evaluate_experiment(
    exp_id: str,
    checkpoint_dir: str,
    num_episodes: int = 100,
    output_dir: str = "./experiments",
):
    """
    Evaluate a single experiment.

    Args:
        exp_id: Experiment ID (D1-D5)
        checkpoint_dir: Directory containing checkpoints
        num_episodes: Number of episodes to evaluate
        output_dir: Output directory for results

    Returns:
        Dictionary containing evaluation results
    """
    if exp_id not in EXPERIMENT_CONFIGS:
        raise ValueError(f"Unknown experiment ID: {exp_id}")

    config = EXPERIMENT_CONFIGS[exp_id]
    checkpoint_path = Path(checkpoint_dir) / exp_id / "checkpoints"

    # Check if checkpoints exist
    checkpoint_a = checkpoint_path / "agent_a_ep8000.pth"
    checkpoint_b = checkpoint_path / "agent_b_ep8000.pth"

    if not checkpoint_a.exists():
        print(f"[SKIP] Skipping {exp_id} - checkpoint A not found: {checkpoint_a}")
        return None

    if not checkpoint_b.exists():
        print(f"[SKIP] Skipping {exp_id} - checkpoint B not found: {checkpoint_b}")
        return None

    print(f"\nEvaluating Experiment {exp_id}...")
    print(f"  Personality A: {config['personality_a']}")
    print(f"  Personality B: {config['personality_b']}")
    print(f"  Episodes: {num_episodes}")

    # Create environment with Deep RL reward and optimized termination
    env = RelationshipEnv(
        max_episode_steps=20,
        use_history=True,  # Deep RL uses history
        history_length=10,
        initial_emotion=-0.2,  # Slightly negative (conflict scenario)
        initial_trust=0.6,  # Moderate trust
        initial_calmness_a=0.6,  # More calm
        initial_calmness_b=0.6,
        irritability_a=config["irritability_a"],
        irritability_b=config["irritability_b"],
        recovery_rate=0.02,
        use_deep_rl_reward=True,  # Enable Deep RL reward function
        termination_thresholds={
            "success_emotion": 0.2,
            "success_trust": 0.6,
            "failure_emotion": -0.9,
            "failure_trust": 0.1,
        },
    )

    # Get state dimension
    obs, _ = env.reset()
    state_dim = len(obs)

    # Load agents
    personality_a = PersonalityType[config["personality_a"].upper()]
    personality_b = PersonalityType[config["personality_b"].upper()]

    agent_a = DQNAgent(
        state_dim=state_dim,
        action_dim=10,
        personality=personality_a,
    )
    agent_a.load(str(checkpoint_a))

    agent_b = DQNAgent(
        state_dim=state_dim,
        action_dim=10,
        personality=personality_b,
    )
    agent_b.load(str(checkpoint_b))

    # Evaluate
    evaluator = Evaluator(env)

    # Evaluate multiple episodes and collect per-episode data
    episode_results = []
    for _ in range(num_episodes):
        metrics = evaluator.evaluate_episode(agent_a, agent_b, render=False)
        episode_results.append(metrics)

    # Aggregate results
    results = {
        "experiment_id": exp_id,
        "config": config,
        "num_episodes": num_episodes,
        # Average metrics
        "avg_reward_a": sum(r["total_reward_a"] for r in episode_results)
        / num_episodes,
        "avg_reward_b": sum(r["total_reward_b"] for r in episode_results)
        / num_episodes,
        "std_reward_a": (
            sum(
                (
                    r["total_reward_a"]
                    - sum(r["total_reward_a"] for r in episode_results) / num_episodes
                )
                ** 2
                for r in episode_results
            )
            / num_episodes
        )
        ** 0.5,
        "std_reward_b": (
            sum(
                (
                    r["total_reward_b"]
                    - sum(r["total_reward_b"] for r in episode_results) / num_episodes
                )
                ** 2
                for r in episode_results
            )
            / num_episodes
        )
        ** 0.5,
        "avg_episode_length": sum(r["episode_length"] for r in episode_results)
        / num_episodes,
        "avg_final_emotion": sum(r["final_emotion"] for r in episode_results)
        / num_episodes,
        "avg_final_trust": sum(r["final_trust"] for r in episode_results)
        / num_episodes,
        "avg_final_conflict": sum(r["final_conflict"] for r in episode_results)
        / num_episodes,
        "avg_final_calmness_a": sum(r["final_calmness_a"] for r in episode_results)
        / num_episodes,
        "avg_final_calmness_b": sum(r["final_calmness_b"] for r in episode_results)
        / num_episodes,
        # Termination rates
        "success_rate": sum(r["success"] for r in episode_results) / num_episodes,
        "failure_rate": sum(r["failure"] for r in episode_results) / num_episodes,
        "stalemate_rate": sum(r["neutral"] for r in episode_results) / num_episodes,
        # Per-episode data (for convergence curves)
        "episode_rewards_a": [r["total_reward_a"] for r in episode_results],
        "episode_rewards_b": [r["total_reward_b"] for r in episode_results],
        "episode_lengths": [r["episode_length"] for r in episode_results],
        "episode_final_emotions": [r["final_emotion"] for r in episode_results],
        "episode_final_trusts": [r["final_trust"] for r in episode_results],
        "episode_termination_reasons": [
            r["termination_reason"] for r in episode_results
        ],
        # Action distributions
        "action_distribution_a": defaultdict(int),
        "action_distribution_b": defaultdict(int),
    }

    # Aggregate action distributions
    for r in episode_results:
        for action, count in r["action_distribution_a"].items():
            results["action_distribution_a"][action] += count
        for action, count in r["action_distribution_b"].items():
            results["action_distribution_b"][action] += count

    # Convert to regular dict
    results["action_distribution_a"] = dict(results["action_distribution_a"])
    results["action_distribution_b"] = dict(results["action_distribution_b"])

    print(f"[OK] Evaluation for {exp_id} completed")
    print(f"  Success rate: {results['success_rate']:.2%}")
    print(f"  Failure rate: {results['failure_rate']:.2%}")
    print(f"  Stalemate rate: {results['stalemate_rate']:.2%}")
    print(
        f"  Avg reward A: {results['avg_reward_a']:.3f} ± {results['std_reward_a']:.3f}"
    )
    print(f"  Avg episode length: {results['avg_episode_length']:.1f}")

    # Save results
    output_path = Path(output_dir) / exp_id / f"evaluation_deep_{exp_id}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        # Convert numpy types to native Python types for JSON serialization
        def convert_to_serializable(obj):
            if isinstance(obj, (dict, defaultdict)):
                # Convert keys to strings if they are numpy types
                return {
                    (
                        str(k) if isinstance(k, (np.integer, np.int64, np.int32)) else k
                    ): convert_to_serializable(v)
                    for k, v in obj.items()
                }
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            elif hasattr(obj, "item"):  # numpy scalar
                return obj.item()
            elif isinstance(obj, (np.integer, np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, (float, int, str, bool)) or obj is None:
                return obj
            else:
                return str(obj)

        json.dump(convert_to_serializable(results), f, indent=2)

    print(f"  Results saved to: {output_path}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate optimized Deep RL experiments (D1-D5)"
    )
    parser.add_argument(
        "--experiment",
        type=str,
        choices=["D1", "D2", "D3", "D4", "D5"],
        help="Single experiment to evaluate (if not specified, evaluates all)",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="./experiments",
        help="Base directory containing experiment checkpoints (default: ./experiments)",
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=100,
        help="Number of episodes per experiment (default: 100)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./experiments",
        help="Output directory for evaluation results (default: ./experiments)",
    )

    args = parser.parse_args()

    if args.experiment:
        # Evaluate single experiment
        experiments = [args.experiment]
    else:
        # Evaluate all experiments
        experiments = ["D1", "D2", "D3", "D4", "D5"]

    print(f"Evaluating {len(experiments)} experiments...")

    all_results = {}
    for exp_id in experiments:
        try:
            results = evaluate_experiment(
                exp_id,
                args.checkpoint_dir,
                args.num_episodes,
                args.output_dir,
            )
            if results:
                all_results[exp_id] = results
        except Exception as e:
            print(f"[FAILED] Evaluation for {exp_id} failed: {e}")
            import traceback

            traceback.print_exc()

    # Save aggregated results
    if all_results:
        aggregated_path = Path(args.output_dir) / "deep_rl_evaluation_results.json"
        with open(aggregated_path, "w", encoding="utf-8") as f:
            # Use the same conversion function for aggregated results
            def convert_to_serializable(obj):
                if isinstance(obj, (dict, defaultdict)):
                    # Convert keys to strings if they are numpy types
                    return {
                        (
                            str(k)
                            if isinstance(k, (np.integer, np.int64, np.int32))
                            else k
                        ): convert_to_serializable(v)
                        for k, v in obj.items()
                    }
                elif isinstance(obj, list):
                    return [convert_to_serializable(item) for item in obj]
                elif hasattr(obj, "item"):  # numpy scalar
                    return obj.item()
                elif isinstance(obj, (np.integer, np.int64, np.int32)):
                    return int(obj)
                elif isinstance(obj, (np.floating, np.float64, np.float32)):
                    return float(obj)
                elif isinstance(obj, (float, int, str, bool)) or obj is None:
                    return obj
                else:
                    return str(obj)

            json.dump(
                convert_to_serializable(all_results), f, indent=2, ensure_ascii=False
            )
        print(f"\nAggregated results saved to: {aggregated_path}")

        # Print summary
        print("\n" + "=" * 80)
        print("Evaluation Summary")
        print("=" * 80)
        for exp_id, results in all_results.items():
            print(
                f"\n{exp_id}: {EXPERIMENT_CONFIGS[exp_id]['personality_a']} vs {EXPERIMENT_CONFIGS[exp_id]['personality_b']}"
            )
            print(f"  Success rate: {results['success_rate']:.2%}")
            print(f"  Failure rate: {results['failure_rate']:.2%}")
            print(f"  Stalemate rate: {results['stalemate_rate']:.2%}")
            print(f"  Avg reward A: {results['avg_reward_a']:.3f}")
            print(f"  Avg episode length: {results['avg_episode_length']:.1f}")

    print("\nEvaluation completed!")


if __name__ == "__main__":
    main()

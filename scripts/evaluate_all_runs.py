"""
Aggregate evaluation script for all runs of each experiment.
Evaluates all runs (run_1 to run_15) for each experiment (D1-D5)
and aggregates the results with mean and standard deviation.
"""

import sys
import os
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from environment import RelationshipEnv
from agents.deep_rl import DQNAgent
from personality import PersonalityType
from training import Evaluator


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

# Termination thresholds (MUST match training configuration!)
TERMINATION_THRESHOLDS = {
    "success_emotion": 0.2,
    "success_trust": 0.6,
    "failure_emotion": -0.5,
    "failure_trust": 0.1,
}


def evaluate_single_run(checkpoint_dir, exp_config, num_episodes=100, verbose=False):
    """Evaluate a single training run."""
    checkpoint_dir = Path(checkpoint_dir)

    # Find checkpoint files
    agent_a_path = checkpoint_dir / "agent_a_ep4000.pth"
    agent_b_path = checkpoint_dir / "agent_b_ep4000.pth"

    if not agent_a_path.exists() or not agent_b_path.exists():
        return None

    # Create environment
    env = RelationshipEnv(
        use_deep_rl_reward=True,
        max_episode_steps=50,
        use_history=True,
        history_length=10,
        initial_emotion=-0.3,
        initial_trust=0.4,
        initial_calmness_a=0.4,
        initial_calmness_b=0.4,
        irritability_a=exp_config["irritability_a"],
        irritability_b=exp_config["irritability_b"],
        termination_thresholds=TERMINATION_THRESHOLDS,
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
        epsilon=0.0,
        memory_size=100000,
        tau=0.005,
        personality=personality_a,
    )

    agent_b = DQNAgent(
        state_dim=state_dim,
        action_dim=10,
        learning_rate=3e-4,
        discount_factor=0.99,
        epsilon=0.0,
        memory_size=100000,
        tau=0.005,
        personality=personality_b,
    )

    # Load checkpoints
    try:
        agent_a.load(str(agent_a_path))
        agent_b.load(str(agent_b_path))
    except Exception as e:
        if verbose:
            print(f"Error loading checkpoint: {e}")
        return None

    # Create evaluator
    evaluator = Evaluator(env)

    # Run evaluation
    results = evaluator.evaluate_multiple_episodes(
        agent_a, agent_b, num_episodes=num_episodes
    )

    return results


def aggregate_results(all_results):
    """Aggregate results from multiple runs."""
    if not all_results:
        return None

    # Extract metrics from all runs
    success_rates = [r["success_rate"] for r in all_results]
    failure_rates = [r["failure_rate"] for r in all_results]
    neutral_rates = [r["neutral_rate"] for r in all_results]
    avg_lengths = [r["avg_episode_length"] for r in all_results]
    avg_emotions = [r["avg_final_emotion"] for r in all_results]
    avg_trusts = [r["avg_final_trust"] for r in all_results]
    avg_conflicts = [r["avg_final_conflict"] for r in all_results]
    avg_calmness_a = [r["avg_final_calmness_a"] for r in all_results]
    avg_calmness_b = [r["avg_final_calmness_b"] for r in all_results]
    
    # Team and Individual rewards
    team_reward_a = [r.get("avg_team_reward_a", r.get("avg_reward_a", 0)) for r in all_results]
    team_reward_b = [r.get("avg_team_reward_b", r.get("avg_reward_b", 0)) for r in all_results]
    individual_reward_a = [r.get("avg_individual_reward_a", 0) for r in all_results]
    individual_reward_b = [r.get("avg_individual_reward_b", 0) for r in all_results]

    aggregated = {
        "num_runs": len(all_results),
        "team_reward_a": {
            "mean": float(np.mean(team_reward_a)),
            "std": float(np.std(team_reward_a)),
        },
        "team_reward_b": {
            "mean": float(np.mean(team_reward_b)),
            "std": float(np.std(team_reward_b)),
        },
        "individual_reward_a": {
            "mean": float(np.mean(individual_reward_a)),
            "std": float(np.std(individual_reward_a)),
        },
        "individual_reward_b": {
            "mean": float(np.mean(individual_reward_b)),
            "std": float(np.std(individual_reward_b)),
        },
        "success_rate": {
            "mean": float(np.mean(success_rates)),
            "std": float(np.std(success_rates)),
            "min": float(np.min(success_rates)),
            "max": float(np.max(success_rates)),
        },
        "failure_rate": {
            "mean": float(np.mean(failure_rates)),
            "std": float(np.std(failure_rates)),
            "min": float(np.min(failure_rates)),
            "max": float(np.max(failure_rates)),
        },
        "neutral_rate": {
            "mean": float(np.mean(neutral_rates)),
            "std": float(np.std(neutral_rates)),
            "min": float(np.min(neutral_rates)),
            "max": float(np.max(neutral_rates)),
        },
        "episode_length": {
            "mean": float(np.mean(avg_lengths)),
            "std": float(np.std(avg_lengths)),
            "min": float(np.min(avg_lengths)),
            "max": float(np.max(avg_lengths)),
        },
        "final_emotion": {
            "mean": float(np.mean(avg_emotions)),
            "std": float(np.std(avg_emotions)),
            "min": float(np.min(avg_emotions)),
            "max": float(np.max(avg_emotions)),
        },
        "final_trust": {
            "mean": float(np.mean(avg_trusts)),
            "std": float(np.std(avg_trusts)),
            "min": float(np.min(avg_trusts)),
            "max": float(np.max(avg_trusts)),
        },
        "final_conflict": {
            "mean": float(np.mean(avg_conflicts)),
            "std": float(np.std(avg_conflicts)),
        },
        "final_calmness_a": {
            "mean": float(np.mean(avg_calmness_a)),
            "std": float(np.std(avg_calmness_a)),
        },
        "final_calmness_b": {
            "mean": float(np.mean(avg_calmness_b)),
            "std": float(np.std(avg_calmness_b)),
        },
    }

    return aggregated


def evaluate_experiment(exp_id, max_runs=15, num_episodes=100, save_dir="./experiments"):
    """Evaluate all runs of a single experiment."""
    print(f"\n{'='*70}")
    print(f"Evaluating Experiment: {exp_id}")
    print(f"{'='*70}")

    exp_config = EXPERIMENT_CONFIGS[exp_id]
    print(f"Personality A: {exp_config['personality_a']}")
    print(f"Personality B: {exp_config['personality_b']}")
    print()

    exp_dir = Path(save_dir) / exp_id / "checkpoints"
    if not exp_dir.exists():
        print(f"❌ Experiment directory not found: {exp_dir}")
        return None

    # Find all run directories
    run_dirs = sorted(
        [d for d in exp_dir.iterdir() if d.is_dir() and d.name.startswith("run_")]
    )

    if not run_dirs:
        print(f"❌ No run directories found in {exp_dir}")
        return None

    print(f"Found {len(run_dirs)} run(s)")
    print()

    # Evaluate each run
    all_results = []
    for run_dir in tqdm(run_dirs[:max_runs], desc=f"Evaluating {exp_id}"):
        results = evaluate_single_run(run_dir, exp_config, num_episodes, verbose=False)
        if results is not None:
            all_results.append(results)

    if not all_results:
        print(f"❌ No valid results for {exp_id}")
        return None

    print(f"\n✅ Successfully evaluated {len(all_results)}/{len(run_dirs[:max_runs])} runs")

    # Aggregate results
    aggregated = aggregate_results(all_results)

    # Print summary in detailed format (matching single run evaluation style)
    print(f"\n{'='*70}")
    print(f"AGGREGATED EVALUATION RESULTS - {exp_id}")
    print(f"{'='*70}")
    print(f"Number of Runs Evaluated: {aggregated['num_runs']}")
    
    print(f"\nTeam Rewards (used for training):")
    print(f"  Agent A - Mean: {aggregated['team_reward_a']['mean']:.3f} ± {aggregated['team_reward_a']['std']:.3f}")
    print(f"  Agent B - Mean: {aggregated['team_reward_b']['mean']:.3f} ± {aggregated['team_reward_b']['std']:.3f}")
    
    print(f"\nIndividual Rewards (for analysis):")
    print(f"  Agent A - Mean: {aggregated['individual_reward_a']['mean']:.3f} ± {aggregated['individual_reward_a']['std']:.3f}")
    print(f"  Agent B - Mean: {aggregated['individual_reward_b']['mean']:.3f} ± {aggregated['individual_reward_b']['std']:.3f}")
    
    print(f"\nCooperation Scores:")
    print(f"  Agent A: {aggregated['individual_reward_a']['mean']:.3f} (higher = more cooperative)")
    print(f"  Agent B: {aggregated['individual_reward_b']['mean']:.3f} (higher = more cooperative)")
    
    print(f"\nEpisode Statistics:")
    print(f"  Average Length: {aggregated['episode_length']['mean']:.1f} steps (std: {aggregated['episode_length']['std']:.1f})")
    
    print(f"\nFinal State Metrics:")
    print(f"  Emotion:  {aggregated['final_emotion']['mean']:.3f} (std: {aggregated['final_emotion']['std']:.3f})")
    print(f"  Trust:    {aggregated['final_trust']['mean']:.3f} (std: {aggregated['final_trust']['std']:.3f})")
    print(f"  Conflict: {aggregated['final_conflict']['mean']:.3f} (std: {aggregated['final_conflict']['std']:.3f})")
    
    print(f"\nTermination Rates:")
    print(f"  Success (Repaired):  {aggregated['success_rate']['mean']*100:5.1f}% (std: {aggregated['success_rate']['std']*100:4.1f}%, range: {aggregated['success_rate']['min']*100:.1f}%-{aggregated['success_rate']['max']*100:.1f}%)")
    print(f"  Failure (Broken):    {aggregated['failure_rate']['mean']*100:5.1f}% (std: {aggregated['failure_rate']['std']*100:4.1f}%, range: {aggregated['failure_rate']['min']*100:.1f}%-{aggregated['failure_rate']['max']*100:.1f}%)")
    print(f"  Neutral (Stalemate): {aggregated['neutral_rate']['mean']*100:5.1f}% (std: {aggregated['neutral_rate']['std']*100:4.1f}%, range: {aggregated['neutral_rate']['min']*100:.1f}%-{aggregated['neutral_rate']['max']*100:.1f}%)")
    
    print(f"\nFinal Calmness:")
    print(f"  Agent A: {aggregated['final_calmness_a']['mean']:.3f} (std: {aggregated['final_calmness_a']['std']:.3f})")
    print(f"  Agent B: {aggregated['final_calmness_b']['mean']:.3f} (std: {aggregated['final_calmness_b']['std']:.3f})")
    
    print(f"{'='*70}")
    print(f"\n💡 Note: 'std' = standard deviation (variability across runs)")
    print(f"    'range' = [min, max] values observed across all runs")

    # Save aggregated results
    output_file = Path(save_dir) / exp_id / f"aggregated_evaluation_{exp_id}.json"
    with open(output_file, "w") as f:
        json.dump(
            {
                "experiment": exp_id,
                "personality_a": exp_config["personality_a"],
                "personality_b": exp_config["personality_b"],
                "aggregated_metrics": aggregated,
            },
            f,
            indent=2,
        )

    print(f"\n📊 Aggregated results saved to: {output_file}")

    return aggregated


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate all runs for each experiment and aggregate results"
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=["D1", "D2", "D3", "D4", "D5"],
        choices=["D1", "D2", "D3", "D4", "D5"],
        help="Experiments to evaluate (default: all)",
    )
    parser.add_argument(
        "--max_runs",
        type=int,
        default=15,
        help="Maximum number of runs to evaluate per experiment",
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=100,
        help="Number of episodes per run evaluation",
    )
    parser.add_argument(
        "--save_dir", type=str, default="./experiments", help="Experiments directory"
    )

    args = parser.parse_args()

    print("="*70)
    print("AGGREGATE EVALUATION OF ALL RUNS")
    print("="*70)
    print(f"Experiments: {', '.join(args.experiments)}")
    print(f"Max runs per experiment: {args.max_runs}")
    print(f"Episodes per run: {args.num_episodes}")
    print(f"Save directory: {args.save_dir}")

    # Evaluate all experiments
    all_aggregated = {}
    for exp_id in args.experiments:
        aggregated = evaluate_experiment(
            exp_id,
            max_runs=args.max_runs,
            num_episodes=args.num_episodes,
            save_dir=args.save_dir,
        )
        if aggregated is not None:
            all_aggregated[exp_id] = aggregated

    # Print comparison table
    if len(all_aggregated) > 1:
        print(f"\n\n{'='*90}")
        print("COMPARISON ACROSS ALL EXPERIMENTS")
        print(f"{'='*90}\n")
        print(f"{'Exp':<5} {'Personality':^25} {'Success%':>12} {'Failure%':>12} {'Stalemate%':>12} {'Ep.Len':>10}")
        print(f"{'─'*90}")

        for exp_id, agg in all_aggregated.items():
            config = EXPERIMENT_CONFIGS[exp_id]
            pers = f"{config['personality_a']} × {config['personality_b']}"
            succ = f"{agg['success_rate']['mean']*100:5.1f}±{agg['success_rate']['std']*100:4.1f}"
            fail = f"{agg['failure_rate']['mean']*100:5.1f}±{agg['failure_rate']['std']*100:4.1f}"
            neut = f"{agg['neutral_rate']['mean']*100:5.1f}±{agg['neutral_rate']['std']*100:4.1f}"
            eplen = f"{agg['episode_length']['mean']:4.1f}±{agg['episode_length']['std']:3.1f}"
            print(f"{exp_id:<5} {pers:^25} {succ:>12} {fail:>12} {neut:>12} {eplen:>10}")
        
        print(f"{'─'*90}")
        print("\nNote: Values shown as Mean±Std across all runs")

    print("\n✅ All evaluations completed!\n")


if __name__ == "__main__":
    main()


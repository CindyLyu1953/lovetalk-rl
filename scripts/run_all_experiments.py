"""
Batch Training Script for All 13 Core Experiments

Runs all training combinations as specified in the experiment design:
- Table A: Baseline + Personality Comparison (Shallow RL) - 6 experiments
- Table B: Deep RL Comparison - 4 experiments
- Table C: Algorithm Comparison - 3 experiments (requires S2 + D2 + additional)

Total: 13 core experiments
"""

import argparse
import subprocess
import os
import json
import shutil
from datetime import datetime
from pathlib import Path


# Experiment configurations
EXPERIMENTS = {
    # Table A: Shallow RL - Baseline + Personality Comparison
    "S1": {
        "algorithm": "q_learning",
        "mode": "self_play",
        "personality_a": "neutral",
        "personality_b": "neutral",
        "description": "Baseline (neutral vs neutral)",
        "category": "shallow",
    },
    "S2": {
        "algorithm": "q_learning",
        "mode": "self_play",
        "personality_a": "impulsive",
        "personality_b": "sensitive",
        "description": "Most intense conflict combination (impulsive vs sensitive)",
        "category": "shallow",
    },
    "S3": {
        "algorithm": "q_learning",
        "mode": "self_play",
        "personality_a": "impulsive",
        "personality_b": "impulsive",
        "description": "Extreme conflict (impulsive vs impulsive)",
        "category": "shallow",
    },
    "S4": {
        "algorithm": "q_learning",
        "mode": "self_play",
        "personality_a": "neutral",
        "personality_b": "avoidant",
        "description": "Cold war mode (neutral vs avoidant)",
        "category": "shallow",
    },
    "S5": {
        "algorithm": "q_learning",
        "mode": "self_play",
        "personality_a": "sensitive",
        "personality_b": "sensitive",
        "description": "Both sensitive, easily misunderstood (sensitive vs sensitive)",
        "category": "shallow",
    },
    "S6": {
        "algorithm": "q_learning",
        "mode": "fixed_opponent",
        "personality_a": "impulsive",
        "personality_b": "sensitive",
        "description": "A learns to understand sensitive partner (fixed opponent)",
        "category": "shallow",
    },
    # Table B: Deep RL Comparison
    "D1": {
        "algorithm": "dqn",
        "mode": "self_play",
        "personality_a": "neutral",
        "personality_b": "neutral",
        "description": "Deep baseline (DQN, neutral vs neutral)",
        "category": "deep",
    },
    "D2": {
        "algorithm": "dqn",
        "mode": "self_play",
        "personality_a": "impulsive",
        "personality_b": "sensitive",
        "description": "Intense conflict (DQN, impulsive vs sensitive)",
        "category": "deep",
    },
    "D3": {
        "algorithm": "dqn",
        "mode": "self_play",
        "personality_a": "impulsive",
        "personality_b": "impulsive",
        "description": "Extreme conflict (DQN, impulsive vs impulsive)",
        "category": "deep",
    },
    "D4": {
        "algorithm": "dqn",
        "mode": "self_play",
        "personality_a": "neutral",
        "personality_b": "avoidant",
        "description": "Cold war (DQN, neutral vs avoidant)",
        "category": "deep",
    },
    "D5": {
        "algorithm": "dqn",
        "mode": "self_play",
        "personality_a": "sensitive",
        "personality_b": "sensitive",
        "description": "Mutual misunderstanding (DQN, sensitive vs sensitive)",
        "category": "deep",
    },
    # Table C: Algorithm Comparison (additional experiments)
    "S2_SARSA": {
        "algorithm": "sarsa",
        "mode": "self_play",
        "personality_a": "impulsive",
        "personality_b": "sensitive",
        "description": "SARSA for comparison with S2 (Q-learning vs SARSA)",
        "category": "shallow",
    },
    # Note: C3 (Shallow vs Deep) can be compared using S2 vs D2
}


def run_experiment(
    exp_id, config, num_episodes, log_interval, save_interval, base_output_dir
):
    """Run a single experiment and save metadata."""
    print(f"\n{'='*80}")
    print(f"Running Experiment {exp_id}: {config['description']}")
    print(f"{'='*80}")

    # Create output directory for this experiment
    output_dir = Path(base_output_dir) / exp_id
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Determine which script to use
    if config["category"] == "shallow":
        script = "scripts/train_shallow.py"
        cmd = [
            "python",
            script,
            "--algorithm",
            config["algorithm"],
            "--episodes",
            str(num_episodes),
            "--personality_a",
            config["personality_a"],
            "--personality_b",
            config["personality_b"],
            "--train_mode",
            config["mode"],
            "--save_dir",
            str(checkpoint_dir),
            "--log_interval",
            str(log_interval),
            "--save_interval",
            str(save_interval),
        ]
    else:  # deep
        script = "scripts/train_deep.py"
        # Deep RL training script uses --experiment parameter
        cmd = [
            "python",
            script,
            "--experiment",
            exp_id,
            "--save_dir",
            str(checkpoint_dir.parent),  # train_deep.py expects base dir
        ]

    # Create log file
    log_file = output_dir / f"training_log_{exp_id}.txt"

    print(f"Command: {' '.join(cmd)}")
    print(f"Log file: {log_file}")
    print(f"Checkpoint dir: {checkpoint_dir}")

    # Save experiment metadata
    metadata = {
        "exp_id": exp_id,
        "config": config,
        "num_episodes": num_episodes,
        "start_time": datetime.now().isoformat(),
        "command": " ".join(cmd),
    }

    metadata_file = output_dir / "metadata.json"
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)

    # Run training
    try:
        with open(log_file, "w") as f:
            result = subprocess.run(
                cmd,
                stdout=f,
                stderr=subprocess.STDOUT,
                text=True,
                check=True,
            )

        # Update metadata with completion
        metadata["end_time"] = datetime.now().isoformat()
        metadata["status"] = "completed"
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"[OK] Experiment {exp_id} completed successfully")
        return True

    except subprocess.CalledProcessError as e:
        print(f"[FAILED] Experiment {exp_id} failed with error: {e}")
        metadata["end_time"] = datetime.now().isoformat()
        metadata["status"] = "failed"
        metadata["error"] = str(e)
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Run all 12 core experiments for Relationship Dynamics Simulator (7 Shallow RL + 5 Deep RL)"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=5000,
        help="Number of episodes per experiment (default: 5000)",
    )
    parser.add_argument(
        "--log_interval",
        type=int,
        default=100,
        help="Logging interval (default: 100)",
    )
    parser.add_argument(
        "--save_interval",
        type=int,
        default=1000,
        help="Checkpoint save interval (default: 1000)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./experiments",
        help="Base output directory for all experiments (default: ./experiments)",
    )
    parser.add_argument(
        "--experiments",
        type=str,
        nargs="+",
        default=None,
        help="Specific experiments to run (e.g., S1 S2 D1). If not specified, runs all.",
    )
    parser.add_argument(
        "--skip_completed",
        action="store_true",
        help="Skip experiments that already have completed metadata",
    )
    parser.add_argument(
        "--clear_first",
        action="store_true",
        help="Clear experiments directory before running (overwrites old results)",
    )
    parser.add_argument(
        "--clear_force",
        action="store_true",
        help="Clear without confirmation (use with --clear_first)",
    )

    args = parser.parse_args()

    # Determine which experiments to run
    if args.experiments:
        exp_ids = args.experiments
    else:
        # Run all 12 core experiments (Shallow RL: S1-S6 + S2_SARSA, Deep RL: D1-D5)
        exp_ids = [
            "S1",
            "S2",
            "S3",
            "S4",
            "S5",
            "S6",
            "S2_SARSA",
            "D1",
            "D2",
            "D3",
            "D4",
            "D5",
        ]

    # Filter to only include defined experiments
    exp_ids = [eid for eid in exp_ids if eid in EXPERIMENTS]

    print(f"\n{'='*80}")
    print(f"Starting Batch Training: {len(exp_ids)} Experiments")
    print(f"Total episodes per experiment: {args.episodes}")
    print(f"Output directory: {args.output_dir}")
    print(f"{'='*80}\n")

    # Clear experiments directory if requested
    output_dir = Path(args.output_dir)
    if args.clear_first:
        if output_dir.exists():
            if args.clear_force:
                print(f"Clearing experiments directory: {output_dir}")
                shutil.rmtree(output_dir)
                print(f"✓ Cleared {output_dir}")
            else:
                response = input(
                    f"Clear all experiments in '{output_dir}'? This will delete all existing results. (yes/no): "
                )
                if response.lower() in ["yes", "y"]:
                    print(f"Clearing experiments directory: {output_dir}")
                    shutil.rmtree(output_dir)
                    print(f"✓ Cleared {output_dir}")
                else:
                    print("Skipping clear. Continuing with existing experiments.")
        else:
            print(f"Experiments directory does not exist, will create: {output_dir}")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save overall experiment plan
    plan = {
        "total_experiments": len(exp_ids),
        "episodes_per_experiment": args.episodes,
        "experiments": {eid: EXPERIMENTS[eid] for eid in exp_ids},
        "start_time": datetime.now().isoformat(),
    }
    plan_file = output_dir / "experiment_plan.json"
    with open(plan_file, "w") as f:
        json.dump(plan, f, indent=2)

    # Run experiments
    results = {}
    for exp_id in exp_ids:
        # Skip if already completed
        if args.skip_completed:
            metadata_file = output_dir / exp_id / "metadata.json"
            if metadata_file.exists():
                with open(metadata_file, "r") as f:
                    metadata = json.load(f)
                    if metadata.get("status") == "completed":
                        print(f"[SKIP] Skipping {exp_id} (already completed)")
                        results[exp_id] = "skipped"
                        continue

        config = EXPERIMENTS[exp_id]
        success = run_experiment(
            exp_id,
            config,
            args.episodes,
            args.log_interval,
            args.save_interval,
            output_dir,
        )
        results[exp_id] = "success" if success else "failed"

    # Save summary
    summary = {
        "total": len(exp_ids),
        "success": sum(1 for r in results.values() if r == "success"),
        "failed": sum(1 for r in results.values() if r == "failed"),
        "skipped": sum(1 for r in results.values() if r == "skipped"),
        "results": results,
        "end_time": datetime.now().isoformat(),
    }

    summary_file = output_dir / "experiment_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*80}")
    print("Experiment Summary:")
    print(f"  Total: {summary['total']}")
    print(f"  Success: {summary['success']}")
    print(f"  Failed: {summary['failed']}")
    print(f"  Skipped: {summary['skipped']}")
    print(f"{'='*80}\n")
    print(f"Summary saved to: {summary_file}")


if __name__ == "__main__":
    main()

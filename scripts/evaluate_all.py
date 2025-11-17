"""
Batch Evaluation Script

Evaluates all trained models and generates comparison statistics.
"""

import argparse
import subprocess
import json
from pathlib import Path
from collections import defaultdict


EXPERIMENT_CHECKPOINTS = {
    "S1": {
        "type": "q_learning",
        "checkpoint_a": "./experiments/S1/checkpoints/agent_a_ep5000.pth",
        "checkpoint_b": "./experiments/S1/checkpoints/agent_b_ep5000.pth",
        "personality_a": "neutral",
        "personality_b": "neutral",
    },
    "S2": {
        "type": "q_learning",
        "checkpoint_a": "./experiments/S2/checkpoints/agent_a_ep5000.pth",
        "checkpoint_b": "./experiments/S2/checkpoints/agent_b_ep5000.pth",
        "personality_a": "impulsive",
        "personality_b": "sensitive",
    },
    "S3": {
        "type": "q_learning",
        "checkpoint_a": "./experiments/S3/checkpoints/agent_a_ep5000.pth",
        "checkpoint_b": "./experiments/S3/checkpoints/agent_b_ep5000.pth",
        "personality_a": "impulsive",
        "personality_b": "impulsive",
    },
    "S4": {
        "type": "q_learning",
        "checkpoint_a": "./experiments/S4/checkpoints/agent_a_ep5000.pth",
        "checkpoint_b": "./experiments/S4/checkpoints/agent_b_ep5000.pth",
        "personality_a": "neutral",
        "personality_b": "avoidant",
    },
    "S5": {
        "type": "q_learning",
        "checkpoint_a": "./experiments/S5/checkpoints/agent_a_ep5000.pth",
        "checkpoint_b": "./experiments/S5/checkpoints/agent_b_ep5000.pth",
        "personality_a": "sensitive",
        "personality_b": "sensitive",
    },
    "S6": {
        "type": "q_learning",
        "checkpoint_a": "./experiments/S6/checkpoints/agent_a_ep5000.pth",
        "checkpoint_b": None,  # Fixed opponent
        "personality_a": "impulsive",
        "personality_b": "sensitive",
    },
    "S2_SARSA": {
        "type": "sarsa",
        "checkpoint_a": "./experiments/S2_SARSA/checkpoints/agent_a_ep5000.pth",
        "checkpoint_b": "./experiments/S2_SARSA/checkpoints/agent_b_ep5000.pth",
        "personality_a": "impulsive",
        "personality_b": "sensitive",
    },
    "D1": {
        "type": "dqn",
        "checkpoint_a": "./experiments/D1/checkpoints/agent_a_ep5000.pth",
        "checkpoint_b": "./experiments/D1/checkpoints/agent_b_ep5000.pth",
        "personality_a": "neutral",
        "personality_b": "neutral",
    },
    "D2": {
        "type": "dqn",
        "checkpoint_a": "./experiments/D2/checkpoints/agent_a_ep8000.pth",
        "checkpoint_b": "./experiments/D2/checkpoints/agent_b_ep8000.pth",
        "personality_a": "impulsive",
        "personality_b": "sensitive",
    },
    "D3": {
        "type": "dqn",
        "checkpoint_a": "./experiments/D3/checkpoints/agent_a_ep8000.pth",
        "checkpoint_b": "./experiments/D3/checkpoints/agent_b_ep8000.pth",
        "personality_a": "impulsive",
        "personality_b": "impulsive",
    },
    "D4": {
        "type": "dqn",
        "checkpoint_a": "./experiments/D4/checkpoints/agent_a_ep8000.pth",
        "checkpoint_b": "./experiments/D4/checkpoints/agent_b_ep8000.pth",
        "personality_a": "neutral",
        "personality_b": "avoidant",
    },
    "D5": {
        "type": "dqn",
        "checkpoint_a": "./experiments/D5/checkpoints/agent_a_ep8000.pth",
        "checkpoint_b": "./experiments/D5/checkpoints/agent_b_ep8000.pth",
        "personality_a": "sensitive",
        "personality_b": "sensitive",
    },
}


def evaluate_experiment(exp_id, config, num_episodes, output_dir):
    """Evaluate a single experiment."""
    print(f"\nEvaluating {exp_id}...")

    # Check if checkpoints exist
    checkpoint_a_path = Path(config["checkpoint_a"])
    if not checkpoint_a_path.exists():
        print(f"[SKIP] Skipping {exp_id} - checkpoint A not found: {checkpoint_a_path}")
        return False

    if config["checkpoint_b"]:
        checkpoint_b_path = Path(config["checkpoint_b"])
        if not checkpoint_b_path.exists():
            print(
                f"[SKIP] Skipping {exp_id} - checkpoint B not found: {checkpoint_b_path}"
            )
            return False

    # Determine which evaluation script to use
    if config["type"] in ["q_learning", "sarsa"]:
        script = "scripts/evaluate_shallow.py"
        cmd = [
            "python",
            script,
            "--agent_type",
            config["type"],
            "--checkpoint_a",
            config["checkpoint_a"],
            "--personality_a",
            config["personality_a"],
            "--personality_b",
            config["personality_b"],
            "--num_episodes",
            str(num_episodes),
        ]
        if config["checkpoint_b"]:
            cmd.extend(["--checkpoint_b", config["checkpoint_b"]])
        log_file = Path(output_dir) / f"evaluation_{exp_id}.txt"
    elif config["type"] == "dqn":
        script = "scripts/evaluate_deep.py"
        # Deep RL evaluation script uses --experiment parameter
        cmd = [
            "python",
            script,
            "--experiment",
            exp_id,
            "--checkpoint_dir",
            "./experiments",
            "--num_episodes",
            str(num_episodes),
            "--output_dir",
            output_dir,
        ]
        log_file = Path(output_dir) / exp_id / f"evaluation_deep_{exp_id}.txt"
        log_file.parent.mkdir(parents=True, exist_ok=True)
    else:
        print(f"[SKIP] Unknown agent type for {exp_id}: {config['type']}")
        return False

    try:
        with open(log_file, "w") as f:
            subprocess.run(
                cmd,
                stdout=f,
                stderr=subprocess.STDOUT,
                text=True,
                check=True,
                cwd=Path.cwd(),
            )
        print(f"[OK] Evaluation for {exp_id} completed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[FAILED] Evaluation for {exp_id} failed: {e}")
        # Print last few lines of log for debugging
        if log_file.exists():
            with open(log_file, "r") as f:
                lines = f.readlines()
                if lines:
                    print(f"  Last lines of log:")
                    for line in lines[-5:]:
                        print(f"    {line.strip()}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Evaluate all trained models")
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=100,
        help="Number of episodes for evaluation (default: 100)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./experiments",
        help="Base output directory (default: ./experiments)",
    )
    parser.add_argument(
        "--experiments",
        type=str,
        nargs="+",
        default=None,
        help="Specific experiments to evaluate (default: all)",
    )

    args = parser.parse_args()

    exp_ids = (
        args.experiments if args.experiments else list(EXPERIMENT_CHECKPOINTS.keys())
    )

    print(f"Evaluating {len(exp_ids)} experiments...")

    results = {}
    for exp_id in exp_ids:
        if exp_id not in EXPERIMENT_CHECKPOINTS:
            print(f"Warning: {exp_id} not found in checkpoint list, skipping")
            continue

        config = EXPERIMENT_CHECKPOINTS[exp_id]
        success = evaluate_experiment(
            exp_id, config, args.num_episodes, args.output_dir
        )
        results[exp_id] = "success" if success else "failed"

    print(
        f"\nEvaluation Summary: {sum(1 for r in results.values() if r == 'success')}/{len(results)} successful"
    )


if __name__ == "__main__":
    main()

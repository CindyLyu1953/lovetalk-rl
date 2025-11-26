"""
Result Collection Script

Collects training statistics and evaluation results from all experiments
and generates comparison tables and summary statistics.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
import re
from scipy import stats


def parse_training_log(log_file):
    """Parse training log to extract statistics."""
    stats = {
        "episodes": [],
        "rewards_a": [],
        "rewards_b": [],
        "lengths": [],
        "final_emotion": [],
        "final_trust": [],
        "final_conflict": [],
    }

    if not log_file.exists():
        return None

    with open(log_file, "r") as f:
        content = f.read()

    # Parse episode logs
    episode_pattern = r"Episode (\d+)"
    reward_pattern = r"Agent A - Avg Reward: ([-\d.]+)"
    reward_b_pattern = r"Agent B - Avg Reward: ([-\d.]+)"
    length_pattern = r"Avg Episode Length: ([\d.]+)"
    emotion_pattern = r"Emotion: ([-\d.]+)"
    trust_pattern = r"Trust: ([-\d.]+)"
    conflict_pattern = r"Conflict: ([-\d.]+)"

    episodes = re.findall(episode_pattern, content)
    rewards_a = re.findall(reward_pattern, content)
    rewards_b = re.findall(reward_b_pattern, content)
    lengths = re.findall(length_pattern, content)
    emotions = re.findall(emotion_pattern, content)
    trusts = re.findall(trust_pattern, content)
    conflicts = re.findall(conflict_pattern, content)

    # Extract final state metrics (from last log entry)
    if emotions:
        stats["final_emotion"] = float(emotions[-1]) if emotions else 0.0
    if trusts:
        stats["final_trust"] = float(trusts[-1]) if trusts else 0.0
    if conflicts:
        stats["final_conflict"] = float(conflicts[-1]) if conflicts else 0.0

    # Get training curve data
    stats["rewards_a"] = [float(r) for r in rewards_a]
    stats["rewards_b"] = [float(r) for r in rewards_b] if rewards_b else []
    stats["lengths"] = [float(l) for l in lengths]
    stats["episodes"] = [int(e) for e in episodes]

    return stats


def parse_evaluation_log(log_file):
    """Parse evaluation log to extract metrics."""
    if not log_file.exists():
        return None

    with open(log_file, "r") as f:
        content = f.read()

    metrics = {}

    # Extract success/failure rates
    success_pattern = r"Success \(Repaired\): ([\d.]+)%"
    failure_pattern = r"Failure \(Broken\): ([\d.]+)%"
    neutral_pattern = r"Neutral \(Stalemate\): ([\d.]+)%"

    success_match = re.search(success_pattern, content)
    failure_match = re.search(failure_pattern, content)
    neutral_match = re.search(neutral_pattern, content)

    if success_match:
        metrics["success_rate"] = float(success_match.group(1)) / 100
    if failure_match:
        metrics["failure_rate"] = float(failure_match.group(1)) / 100
    if neutral_match:
        metrics["neutral_rate"] = float(neutral_match.group(1)) / 100

    # Extract average metrics
    reward_pattern = r"Agent A - Mean: ([-\d.]+) ±"
    emotion_pattern = r"Emotion: ([\d.]+)"
    trust_pattern = r"Trust: ([\d.]+)"
    conflict_pattern = r"Conflict: ([\d.]+)"
    calmness_a_pattern = r"Agent A: ([\d.]+)"
    calmness_b_pattern = r"Agent B: ([\d.]+)"

    reward_match = re.search(reward_pattern, content)
    emotion_match = re.search(emotion_pattern, content)
    trust_match = re.search(trust_pattern, content)
    conflict_match = re.search(conflict_pattern, content)
    calmness_a_match = re.search(calmness_a_pattern, content)
    calmness_b_match = re.search(calmness_b_pattern, content)

    if reward_match:
        metrics["avg_reward_a"] = float(reward_match.group(1))
    if emotion_match:
        metrics["avg_final_emotion"] = float(emotion_match.group(1))
    if trust_match:
        metrics["avg_final_trust"] = float(trust_match.group(1))
    if conflict_match:
        metrics["avg_final_conflict"] = float(conflict_match.group(1))
    if calmness_a_match:
        metrics["avg_final_calmness_a"] = float(calmness_a_match.group(1))
    if calmness_b_match:
        metrics["avg_final_calmness_b"] = float(calmness_b_match.group(1))

    return metrics


def collect_all_results(experiment_dir="./experiments"):
    """Collect results from all experiments."""
    exp_dir = Path(experiment_dir)

    experiments = {
        "S1": "Q-learning, neutral vs neutral (Baseline)",
        "S2": "Q-learning, impulsive vs sensitive",
        "S3": "Q-learning, impulsive vs impulsive",
        "S4": "Q-learning, neutral vs avoidant",
        "S5": "Q-learning, sensitive vs sensitive",
        "S6": "Q-learning, fixed_opponent, impulsive vs sensitive",
        "S2_SARSA": "SARSA, impulsive vs sensitive",
        "D1": "DQN, neutral vs neutral (Deep baseline)",
        "D2": "DQN, impulsive vs sensitive",
        "D3": "PPO, impulsive vs sensitive",
        "D4": "PPO, sensitive vs sensitive",
    }

    results = {}

    for exp_id, description in experiments.items():
        exp_path = exp_dir / exp_id
        # Get metadata
        metadata_file = exp_path / "metadata.json"
        metadata = {}
        if metadata_file.exists():
            with open(metadata_file, "r") as f:
                metadata = json.load(f)

        # Get training statistics
        # Prefer per-run JSON stats saved by new training scripts
        run_dirs = sorted([p for p in exp_path.iterdir() if p.is_dir() and p.name.startswith("run_")])
        per_run_stats = []
        for run_dir in run_dirs:
            tfile = run_dir / "train_stats.json"
            if tfile.exists():
                try:
                    with open(tfile, "r") as f:
                        per_run_stats.append(json.load(f)["stats"])
                except Exception:
                    continue

        # If per-run stats not found, fallback to legacy training log parser
        if per_run_stats:
            training_stats = None
        else:
            training_log = exp_path / f"training_log_{exp_id}.txt"
            training_stats = parse_training_log(training_log)

        # Get evaluation metrics
        eval_log = exp_path / f"evaluation_{exp_id}.txt"
        eval_metrics = parse_evaluation_log(eval_log)

        # Aggregate per-run stats (mean ± 95% CI) when available
        aggregated = {}
        if per_run_stats:
            # Collect arrays for metrics
            final_emotions = []
            final_trusts = []
            avg_rewards = []
            avg_lengths = []
            for s in per_run_stats:
                # s format depends on trainer.get_statistics; try multiple keys
                # Final emotion/trust may be under s["stats"]["final_emotion"] (list)
                try:
                    final_emotions.append(float(s.get("stats", {}).get("final_emotion", [])[-1]))
                except Exception:
                    final_emotions.append(float(np.nan))
                try:
                    final_trusts.append(float(s.get("stats", {}).get("final_trust", [])[-1]))
                except Exception:
                    final_trusts.append(float(np.nan))
                # avg reward: mean of episode_rewards_a
                try:
                    er = s.get("episode_rewards_a", [])
                    avg_rewards.append(float(np.mean(er)) if er else float(np.nan))
                except Exception:
                    avg_rewards.append(float(np.nan))
                try:
                    lengths = s.get("episode_lengths", [])
                    avg_lengths.append(float(np.mean(lengths)) if lengths else float(np.nan))
                except Exception:
                    avg_lengths.append(float(np.nan))

            def mean_ci95(x):
                arr = np.array([v for v in x if not np.isnan(v)])
                if len(arr) == 0:
                    return {"mean": None, "ci95": None, "n": 0}
                mean = float(np.mean(arr))
                ci95 = float(1.96 * np.std(arr, ddof=0) / np.sqrt(len(arr)))
                return {"mean": mean, "ci95": ci95, "n": len(arr)}

            aggregated["final_emotion"] = mean_ci95(final_emotions)
            aggregated["final_trust"] = mean_ci95(final_trusts)
            aggregated["avg_reward_a"] = mean_ci95(avg_rewards)
            aggregated["avg_episode_length"] = mean_ci95(avg_lengths)

        results[exp_id] = {
            "description": description,
            "metadata": metadata,
            "training": training_stats,
            "per_run_stats": per_run_stats,
            "aggregated": aggregated if aggregated else None,
            "evaluation": eval_metrics,
        }

    return results


def generate_comparison_table(
    results, output_file="./experiments/comparison_table.csv"
):
    """Generate comparison table for all experiments."""
    rows = []

    for exp_id, data in results.items():
        row = {
            "Experiment": exp_id,
            "Description": data["description"],
        }

        # Training statistics
        if data["training"]:
            train = data["training"]
            row["Final_Reward_A"] = (
                train["rewards_a"][-1] if train["rewards_a"] else None
            )
            row["Final_Reward_B"] = (
                train["rewards_b"][-1] if train["rewards_b"] else None
            )
            row["Avg_Episode_Length"] = (
                np.mean(train["lengths"]) if train["lengths"] else None
            )
            row["Final_Emotion"] = train.get("final_emotion")
            row["Final_Trust"] = train.get("final_trust")
            row["Final_Conflict"] = train.get("final_conflict")

        # Evaluation metrics
        if data["evaluation"]:
            eval_data = data["evaluation"]
            row["Success_Rate"] = eval_data.get("success_rate")
            row["Failure_Rate"] = eval_data.get("failure_rate")
            row["Neutral_Rate"] = eval_data.get("neutral_rate")
            row["Eval_Avg_Emotion"] = eval_data.get("avg_final_emotion")
            row["Eval_Avg_Trust"] = eval_data.get("avg_final_trust")
            row["Eval_Avg_Calmness_A"] = eval_data.get("avg_final_calmness_a")
            row["Eval_Avg_Calmness_B"] = eval_data.get("avg_final_calmness_b")

        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(output_file, index=False)
    print(f"Comparison table saved to: {output_file}")

    return df


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Collect and summarize experiment results"
    )
    parser.add_argument(
        "--experiment_dir",
        type=str,
        default="./experiments",
        help="Experiment directory (default: ./experiments)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./experiments/comparison_table.csv",
        help="Output CSV file (default: ./experiments/comparison_table.csv)",
    )

    args = parser.parse_args()

    print("Collecting results from all experiments...")
    results = collect_all_results(args.experiment_dir)

    print(f"Generating comparison table...")
    df = generate_comparison_table(results, args.output)

    print("\nSummary Statistics:")
    print(df.describe())

    # Save full results as JSON
    json_file = Path(args.experiment_dir) / "all_results.json"
    with open(json_file, "w") as f:
        # Convert numpy types to native Python types for JSON serialization
        json_results = {}
        for exp_id, data in results.items():
            json_results[exp_id] = {
                "description": data["description"],
                "metadata": data["metadata"],
            }
            if data["training"]:
                train = data["training"]
                json_results[exp_id]["training"] = {
                    k: float(v) if isinstance(v, (int, float, np.number)) else v
                    for k, v in train.items()
                }
            if data["evaluation"]:
                json_results[exp_id]["evaluation"] = {
                    k: float(v) if isinstance(v, (int, float, np.number)) else v
                    for k, v in data["evaluation"].items()
                }

        json.dump(json_results, f, indent=2)

    print(f"\nFull results saved to: {json_file}")

    # Run basic statistical tests across experiments when per-run stats are available
    try:
        run_statistical_tests(Path(args.experiment_dir))
    except Exception as e:
        print(f"Statistical tests skipped due to error: {e}")


def run_statistical_tests(experiment_dir: Path):
    """
    Run basic statistical comparisons across experiments that have per-run stats.
    - Performs one-way ANOVA on average rewards across experiments (if >=3)
    - Performs pairwise t-tests (independent) for average rewards
    Outputs results to STDOUT and saves a JSON summary.
    """
    exp_dir = Path(experiment_dir)
    experiments = sorted([p for p in exp_dir.iterdir() if p.is_dir()])
    group_rewards = {}
    for exp_path in experiments:
        # collect per-run train_stats.json files
        run_dirs = sorted([p for p in exp_path.iterdir() if p.is_dir() and p.name.startswith("run_")])
        rewards = []
        for run in run_dirs:
            tfile = run / "train_stats.json"
            if not tfile.exists():
                continue
            try:
                with open(tfile, "r") as f:
                    s = json.load(f)["stats"]
                    er = s.get("episode_rewards_a", [])
                    if er:
                        rewards.append(float(np.mean(er)))
            except Exception:
                continue
        if rewards:
            group_rewards[exp_path.name] = rewards

    if not group_rewards:
        print("No per-run stats found for statistical testing.")
        return

    # One-way ANOVA if >=3 groups
    summary = {}
    groups = list(group_rewards.keys())
    if len(groups) >= 3:
        arrays = [group_rewards[g] for g in groups]
        fval, pval = stats.f_oneway(*arrays)
        summary["anova"] = {"f": float(fval), "p": float(pval), "groups": groups}
        print(f"\nANOVA across groups (avg reward): F={fval:.4f}, p={pval:.4e}")

    # Pairwise t-tests
    pairwise = {}
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            a = group_rewards[groups[i]]
            b = group_rewards[groups[j]]
            t, p = stats.ttest_ind(a, b, equal_var=False, nan_policy="omit")
            pairwise[f"{groups[i]}_vs_{groups[j]}"] = {"t": float(t), "p": float(p)}
            print(f"t-test {groups[i]} vs {groups[j]}: t={t:.4f}, p={p:.4e}")

    summary["pairwise_ttests"] = pairwise

    # Save summary
    out_file = exp_dir / "statistical_tests_summary.json"
    with open(out_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nStatistical test summary saved to: {out_file}")


if __name__ == "__main__":
    main()

"""
Visualization Script for Experiment Results

Generates learning curves and comparison charts for the paper.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd


def load_results(results_file="./experiments/all_results.json"):
    """Load collected results."""
    with open(results_file, "r") as f:
        return json.load(f)


def plot_learning_curves(results, output_dir="./experiments/figures"):
    """Plot learning curves for all experiments."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Plot 1: Episode Rewards (Shallow RL)
    ax = axes[0, 0]
    shallow_exps = ["S1", "S2", "S3", "S4", "S5", "S6"]
    for exp_id in shallow_exps:
        if exp_id in results and results[exp_id].get("training"):
            rewards = results[exp_id]["training"].get("rewards_a", [])
            episodes = results[exp_id]["training"].get("episodes", [])
            if rewards and episodes:
                ax.plot(episodes, rewards, label=exp_id, alpha=0.7)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Average Reward (Agent A)")
    ax.set_title("Learning Curves - Shallow RL (Q-learning)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Episode Rewards (Deep RL)
    ax = axes[0, 1]
    deep_exps = ["D1", "D2", "D3", "D4"]
    for exp_id in deep_exps:
        if exp_id in results and results[exp_id].get("training"):
            rewards = results[exp_id]["training"].get("rewards_a", [])
            episodes = results[exp_id]["training"].get("episodes", [])
            if rewards and episodes:
                ax.plot(episodes, rewards, label=exp_id, alpha=0.7)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Average Reward (Agent A)")
    ax.set_title("Learning Curves - Deep RL")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Episode Lengths
    ax = axes[1, 0]
    for exp_id, data in results.items():
        if data.get("training"):
            lengths = data["training"].get("lengths", [])
            episodes = data["training"].get("episodes", [])
            if lengths and episodes:
                ax.plot(episodes, lengths, label=exp_id, alpha=0.7)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Average Episode Length")
    ax.set_title("Episode Length Over Training")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.grid(True, alpha=0.3)

    # Plot 4: Final State Metrics
    ax = axes[1, 1]
    exp_ids = list(results.keys())

    # Filter to only experiments with training data
    valid_exp_ids = [eid for eid in exp_ids if results[eid].get("training")]

    # Extract metrics, handling None or list values
    emotions = []
    trusts = []
    conflicts = []
    valid_ids = []

    for eid in valid_exp_ids:
        train_data = results[eid]["training"]
        emotion = train_data.get("final_emotion", 0)
        trust = train_data.get("final_trust", 0)
        conflict = train_data.get("final_conflict", 0)

        # Handle None or list values
        if emotion is None:
            emotion = 0
        elif isinstance(emotion, list):
            emotion = emotion[-1] if emotion else 0

        if trust is None:
            trust = 0
        elif isinstance(trust, list):
            trust = trust[-1] if trust else 0

        if conflict is None:
            conflict = 0
        elif isinstance(conflict, list):
            conflict = conflict[-1] if conflict else 0

        emotions.append(float(emotion))
        trusts.append(float(trust))
        conflicts.append(float(conflict))
        valid_ids.append(eid)

    if valid_ids:
        x = np.arange(len(valid_ids))
        width = 0.25
        ax.bar(x - width, emotions, width, label="Emotion", alpha=0.7)
        ax.bar(x, trusts, width, label="Trust", alpha=0.7)
        ax.bar(x + width, conflicts, width, label="Conflict", alpha=0.7)
        ax.set_xlabel("Experiment")
        ax.set_ylabel("Final Value")
        ax.set_title("Final State Metrics Comparison")
        ax.set_xticks(x)
        ax.set_xticklabels(valid_ids, rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")
    else:
        ax.text(
            0.5,
            0.5,
            "No training data available",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_title("Final State Metrics Comparison")

    plt.tight_layout()
    plt.savefig(f"{output_dir}/learning_curves.pdf", dpi=300, bbox_inches="tight")
    plt.savefig(f"{output_dir}/learning_curves.png", dpi=300, bbox_inches="tight")
    print(f"Learning curves saved to {output_dir}/learning_curves.pdf")
    plt.close()


def plot_termination_rates(results, output_dir="./experiments/figures"):
    """Plot termination rate comparison."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 6))

    exp_ids = []
    success_rates = []
    failure_rates = []
    neutral_rates = []

    for exp_id, data in sorted(results.items()):
        if data.get("evaluation"):
            eval_data = data["evaluation"]
            exp_ids.append(exp_id)
            success_rates.append(eval_data.get("success_rate", 0) * 100)
            failure_rates.append(eval_data.get("failure_rate", 0) * 100)
            neutral_rates.append(eval_data.get("neutral_rate", 0) * 100)

    x = np.arange(len(exp_ids))
    width = 0.6

    ax.bar(
        x, success_rates, width, label="Success (Repaired)", color="green", alpha=0.7
    )
    ax.bar(
        x,
        failure_rates,
        width,
        bottom=success_rates,
        label="Failure (Broken)",
        color="red",
        alpha=0.7,
    )
    ax.bar(
        x,
        neutral_rates,
        width,
        bottom=np.array(success_rates) + np.array(failure_rates),
        label="Neutral (Stalemate)",
        color="gray",
        alpha=0.7,
    )

    ax.set_xlabel("Experiment")
    ax.set_ylabel("Rate (%)")
    ax.set_title("Termination Rate Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(exp_ids, rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim([0, 100])

    plt.tight_layout()
    plt.savefig(f"{output_dir}/termination_rates.pdf", dpi=300, bbox_inches="tight")
    plt.savefig(f"{output_dir}/termination_rates.png", dpi=300, bbox_inches="tight")
    print(f"Termination rates saved to {output_dir}/termination_rates.pdf")
    plt.close()


def plot_algorithm_comparison(results, output_dir="./experiments/figures"):
    """Plot algorithm comparison (C1, C2, C3)."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # C1: Q-learning vs SARSA
    ax = axes[0]
    if "S2" in results and "S2_SARSA" in results:
        s2_reward = results["S2"]["training"].get("rewards_a", [])
        s2_sarsa_reward = results["S2_SARSA"]["training"].get("rewards_a", [])
        s2_episodes = results["S2"]["training"].get("episodes", [])
        if s2_reward and s2_sarsa_reward:
            ax.plot(s2_episodes, s2_reward, label="Q-learning", alpha=0.7)
            ax.plot(s2_episodes, s2_sarsa_reward, label="SARSA", alpha=0.7)
            ax.set_xlabel("Episode")
            ax.set_ylabel("Average Reward")
            ax.set_title("C1: Q-learning vs SARSA\n(impulsive vs sensitive)")
            ax.legend()
            ax.grid(True, alpha=0.3)

    # C2: DQN vs PPO
    ax = axes[1]
    if "D2" in results and "D3" in results:
        d2_reward = results["D2"]["training"].get("rewards_a", [])
        d3_reward = results["D3"]["training"].get("rewards_a", [])
        d2_episodes = results["D2"]["training"].get("episodes", [])
        if d2_reward and d3_reward:
            ax.plot(d2_episodes, d2_reward, label="DQN", alpha=0.7)
            ax.plot(d2_episodes, d3_reward, label="PPO", alpha=0.7)
            ax.set_xlabel("Episode")
            ax.set_ylabel("Average Reward")
            ax.set_title("C2: DQN vs PPO\n(impulsive vs sensitive)")
            ax.legend()
            ax.grid(True, alpha=0.3)

    # C3: Shallow vs Deep
    ax = axes[2]
    if "S2" in results and "D2" in results:
        s2_reward = results["S2"]["training"].get("rewards_a", [])
        d2_reward = results["D2"]["training"].get("rewards_a", [])
        s2_episodes = results["S2"]["training"].get("episodes", [])
        if s2_reward and d2_reward:
            ax.plot(s2_episodes, s2_reward, label="Q-learning (Shallow)", alpha=0.7)
            ax.plot(s2_episodes, d2_reward, label="DQN (Deep)", alpha=0.7)
            ax.set_xlabel("Episode")
            ax.set_ylabel("Average Reward")
            ax.set_title("C3: Shallow vs Deep RL\n(impulsive vs sensitive)")
            ax.legend()
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/algorithm_comparison.pdf", dpi=300, bbox_inches="tight")
    plt.savefig(f"{output_dir}/algorithm_comparison.png", dpi=300, bbox_inches="tight")
    print(f"Algorithm comparison saved to {output_dir}/algorithm_comparison.pdf")
    plt.close()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Visualize experiment results")
    parser.add_argument(
        "--results_file",
        type=str,
        default="./experiments/all_results.json",
        help="Path to results JSON file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./experiments/figures",
        help="Output directory for figures",
    )

    args = parser.parse_args()

    print("Loading results...")
    results = load_results(args.results_file)

    print("Generating learning curves...")
    plot_learning_curves(results, args.output_dir)

    print("Generating termination rate comparison...")
    plot_termination_rates(results, args.output_dir)

    print("Generating algorithm comparison...")
    plot_algorithm_comparison(results, args.output_dir)

    print(f"\nAll figures saved to {args.output_dir}/")


if __name__ == "__main__":
    main()

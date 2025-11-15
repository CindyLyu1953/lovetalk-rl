"""
Visualization Utilities

Provides visualization tools for training statistics, action distributions,
and relationship dynamics.
"""

from typing import Dict, List, Optional
import numpy as np
import matplotlib.pyplot as plt


class Visualizer:
    """
    Visualization utilities for training statistics and evaluation results.
    """

    @staticmethod
    def plot_training_curves(stats: Dict, save_path: Optional[str] = None):
        """
        Plot training curves (rewards, losses, etc.).

        Args:
            stats: Dictionary containing training statistics
            save_path: Path to save figure (optional)
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        # Episode rewards
        if "episode_rewards_a" in stats:
            axes[0, 0].plot(stats["episode_rewards_a"], label="Agent A", alpha=0.7)
            if "episode_rewards_b" in stats and stats["episode_rewards_b"]:
                axes[0, 0].plot(stats["episode_rewards_b"], label="Agent B", alpha=0.7)
            axes[0, 0].set_title("Episode Rewards")
            axes[0, 0].set_xlabel("Episode")
            axes[0, 0].set_ylabel("Reward")
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)

        # Episode lengths
        if "episode_lengths" in stats:
            axes[0, 1].plot(stats["episode_lengths"], alpha=0.7)
            axes[0, 1].set_title("Episode Lengths")
            axes[0, 1].set_xlabel("Episode")
            axes[0, 1].set_ylabel("Steps")
            axes[0, 1].grid(True, alpha=0.3)

        # Final state metrics
        if "stats" in stats and "final_emotion" in stats["stats"]:
            final_emotion = stats["stats"]["final_emotion"]
            final_trust = stats["stats"]["final_trust"]
            final_conflict = stats["stats"]["final_conflict"]

            axes[1, 0].plot(final_emotion, label="Emotion", alpha=0.7)
            axes[1, 0].plot(final_trust, label="Trust", alpha=0.7)
            axes[1, 0].plot(final_conflict, label="Conflict", alpha=0.7)
            axes[1, 0].set_title("Final State Metrics")
            axes[1, 0].set_xlabel("Episode")
            axes[1, 0].set_ylabel("Value")
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)

        # Losses (if available)
        if "stats" in stats and "loss_a" in stats["stats"]:
            axes[1, 1].plot(stats["stats"]["loss_a"], label="Agent A", alpha=0.7)
            if "loss_b" in stats["stats"]:
                axes[1, 1].plot(stats["stats"]["loss_b"], label="Agent B", alpha=0.7)
            axes[1, 1].set_title("Training Loss")
            axes[1, 1].set_xlabel("Update")
            axes[1, 1].set_ylabel("Loss")
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        else:
            plt.show()

        plt.close()

    @staticmethod
    def plot_action_distribution(action_dist: Dict, save_path: Optional[str] = None):
        """
        Plot action distribution.

        Args:
            action_dist: Dictionary mapping action indices to counts
            save_path: Path to save figure (optional)
        """
        from environment.actions import ActionType

        action_names = [ActionType(i).name for i in range(10)]
        counts = [action_dist.get(i, 0) for i in range(10)]

        fig, ax = plt.subplots(figsize=(12, 6))
        bars = ax.bar(action_names, counts)
        ax.set_title("Action Distribution")
        ax.set_xlabel("Action Type")
        ax.set_ylabel("Count")
        ax.tick_params(axis="x", rotation=45)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        else:
            plt.show()

        plt.close()

    @staticmethod
    def plot_relationship_trajectory(
        states: List[np.ndarray], save_path: Optional[str] = None
    ):
        """
        Plot relationship state trajectory over an episode.

        Args:
            states: List of state vectors over an episode
            save_path: Path to save figure (optional)
        """
        if not states:
            return

        states_array = np.array(states)

        fig, ax = plt.subplots(figsize=(10, 6))

        steps = np.arange(len(states))
        ax.plot(steps, states_array[:, 0], label="Emotion", marker="o")
        ax.plot(steps, states_array[:, 1], label="Trust", marker="s")
        ax.plot(steps, states_array[:, 2], label="Conflict", marker="^")

        ax.set_title("Relationship State Trajectory")
        ax.set_xlabel("Step")
        ax.set_ylabel("Value")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        else:
            plt.show()

        plt.close()

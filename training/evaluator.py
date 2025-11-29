"""
Evaluator

Provides evaluation utilities for trained agents, including:
- Episode evaluation
- Policy visualization
- Action distribution analysis
- Conflict resolution success rate
"""

from typing import Dict, Optional
import numpy as np
from collections import Counter

from environment import RelationshipEnv, ActionType


class Evaluator:
    """
    Evaluator for relationship dynamics simulator.

    Provides comprehensive evaluation metrics including:
    - Conflict resolution success rate
    - Final relationship state metrics
    - Action distribution analysis
    - Episode statistics
    """

    def __init__(self, env: RelationshipEnv):
        """
        Initialize evaluator.

        Args:
            env: Relationship communication environment
        """
        self.env = env

    def evaluate_episode(
        self,
        agent_a,
        agent_b: Optional = None,
        render: bool = False,
        max_steps: Optional[int] = None,
        swap_agents: bool = False,
    ) -> Dict:
        """
        Evaluate a single episode with given agents.

        Args:
            agent_a: Agent A (required)
            agent_b: Agent B (if None, uses agent_a as B)
            render: Whether to render episode (default: False)
            max_steps: Maximum steps for evaluation (default: None, uses env default)
            swap_agents: If True, swap agent_a and agent_b roles to eliminate first-mover advantage

        Returns:
            Dictionary containing evaluation metrics
        """
        if agent_b is None:
            agent_b = agent_a

        # Swap agents if requested (to eliminate first-mover advantage)
        if swap_agents:
            actual_agent_a = agent_b
            actual_agent_b = agent_a
        else:
            actual_agent_a = agent_a
            actual_agent_b = agent_b

        obs, info = self.env.reset()
        current_agent = 0

        episode_data = {
            "states": [obs.copy()],
            "actions_a": [],
            "actions_b": [],
            "rewards_a": [],
            "rewards_b": [],
            "infos": [info],
        }

        done = False
        truncated = False
        step_count = 0
        max_steps = max_steps or self.env.max_episode_steps

        while not (done or truncated) and step_count < max_steps:
            agent = actual_agent_a if current_agent == 0 else actual_agent_b

            # Select action (no exploration during evaluation)
            if hasattr(agent, "select_action"):
                action = agent.select_action(obs, training=False)
            else:
                raise ValueError(f"Unknown agent type: {type(agent)}")

            # Store action
            if current_agent == 0:
                episode_data["actions_a"].append(action)
            else:
                episode_data["actions_b"].append(action)

            # Take step
            next_obs, reward, done, truncated, info = self.env.step(action)

            # Store reward (team reward) and individual rewards
            if current_agent == 0:
                episode_data["rewards_a"].append(reward)  # Team reward
                # Store individual reward for analysis
                if "individual_reward_a" not in episode_data:
                    episode_data["individual_reward_a"] = []
                episode_data["individual_reward_a"].append(info.get("individual_reward_a", 0.0))
            else:
                episode_data["rewards_b"].append(reward)  # Team reward
                # Store individual reward for analysis
                if "individual_reward_b" not in episode_data:
                    episode_data["individual_reward_b"] = []
                episode_data["individual_reward_b"].append(info.get("individual_reward_b", 0.0))

            obs = next_obs
            episode_data["states"].append(obs.copy())
            episode_data["infos"].append(info)

            current_agent = 1 - current_agent
            step_count += 1

            if render:
                self.env.render()

        # Compute evaluation metrics
        final_state = info
        termination_reason = final_state.get("termination_reason", "NEUTRAL")

        # If agents were swapped, we need to swap the rewards and actions back
        # to maintain consistency with the original agent_a/agent_b naming
        if swap_agents:
            # Swap rewards and actions
            episode_data["rewards_a"], episode_data["rewards_b"] = (
                episode_data["rewards_b"],
                episode_data["rewards_a"],
            )
            episode_data["actions_a"], episode_data["actions_b"] = (
                episode_data["actions_b"],
                episode_data["actions_a"],
            )
            # Swap calmness values
            final_calmness_a = final_state.get("calmness_b", 0.5)
            final_calmness_b = final_state.get("calmness_a", 0.5)
        else:
            final_calmness_a = final_state.get("calmness_a", 0.5)
            final_calmness_b = final_state.get("calmness_b", 0.5)

        metrics = {
            "total_team_reward_a": sum(episode_data["rewards_a"]),  # Team reward for A
            "total_team_reward_b": sum(episode_data["rewards_b"]),  # Team reward for B
            "total_individual_reward_a": sum(episode_data.get("individual_reward_a", [])),  # Individual reward for A
            "total_individual_reward_b": sum(episode_data.get("individual_reward_b", [])),  # Individual reward for B
            # Legacy fields
            "total_reward_a": sum(episode_data["rewards_a"]),
            "total_reward_b": sum(episode_data["rewards_b"]),
            "episode_length": step_count,
            "final_emotion": final_state["emotion"],
            "final_trust": final_state["trust"],
            "final_conflict": final_state["conflict"],
            "final_calmness_a": final_calmness_a,
            "final_calmness_b": final_calmness_b,
            "termination_reason": termination_reason,
            "success": termination_reason == "SUCCESS",
            "failure": termination_reason == "FAILURE",
            "neutral": termination_reason == "NEUTRAL",
            "action_distribution_a": Counter(episode_data["actions_a"]),
            "action_distribution_b": Counter(episode_data["actions_b"]),
        }

        return metrics

    def evaluate_multiple_episodes(
        self,
        agent_a,
        agent_b: Optional = None,
        num_episodes: int = 100,
        render: bool = False,
        alternate_first_move: bool = True,
    ) -> Dict:
        """
        Evaluate agents over multiple episodes.

        Args:
            agent_a: Agent A (required)
            agent_b: Agent B (if None, uses agent_a as B)
            num_episodes: Number of episodes to evaluate (default: 100)
            render: Whether to render episodes (default: False)
            alternate_first_move: If True, alternate which agent goes first to eliminate
                                  first-mover advantage (default: True)

        Returns:
            Dictionary containing aggregated evaluation metrics
        """
        all_metrics = []

        for episode_idx in range(num_episodes):
            # Alternate first move to eliminate first-mover advantage
            swap_agents = alternate_first_move and (episode_idx % 2 == 1)
            metrics = self.evaluate_episode(
                agent_a, agent_b, render=render, swap_agents=swap_agents
            )
            all_metrics.append(metrics)

        # Aggregate metrics
        aggregated = {
            "num_episodes": num_episodes,
            # Team rewards (used for training)
            "avg_team_reward_a": np.mean([m["total_team_reward_a"] for m in all_metrics]),
            "avg_team_reward_b": np.mean([m["total_team_reward_b"] for m in all_metrics]),
            "std_team_reward_a": np.std([m["total_team_reward_a"] for m in all_metrics]),
            "std_team_reward_b": np.std([m["total_team_reward_b"] for m in all_metrics]),
            # Individual rewards (for evaluation/analysis)
            "avg_individual_reward_a": np.mean([m["total_individual_reward_a"] for m in all_metrics]),
            "avg_individual_reward_b": np.mean([m["total_individual_reward_b"] for m in all_metrics]),
            "std_individual_reward_a": np.std([m["total_individual_reward_a"] for m in all_metrics]),
            "std_individual_reward_b": np.std([m["total_individual_reward_b"] for m in all_metrics]),
            # Legacy fields (backward compatibility)
            "avg_reward_a": np.mean([m["total_reward_a"] for m in all_metrics]),
            "avg_reward_b": np.mean([m["total_reward_b"] for m in all_metrics]),
            "std_reward_a": np.std([m["total_reward_a"] for m in all_metrics]),
            "std_reward_b": np.std([m["total_reward_b"] for m in all_metrics]),
            # Other metrics
            "avg_episode_length": np.mean([m["episode_length"] for m in all_metrics]),
            "avg_final_emotion": np.mean([m["final_emotion"] for m in all_metrics]),
            "avg_final_trust": np.mean([m["final_trust"] for m in all_metrics]),
            "avg_final_conflict": np.mean([m["final_conflict"] for m in all_metrics]),
            "avg_final_calmness_a": np.mean(
                [m["final_calmness_a"] for m in all_metrics]
            ),
            "avg_final_calmness_b": np.mean(
                [m["final_calmness_b"] for m in all_metrics]
            ),
            "success_rate": np.mean([m["success"] for m in all_metrics]),
            "failure_rate": np.mean([m["failure"] for m in all_metrics]),
            "neutral_rate": np.mean([m["neutral"] for m in all_metrics]),
        }

        # Aggregate action distributions
        all_actions_a = []
        all_actions_b = []
        for m in all_metrics:
            all_actions_a.extend(list(m["action_distribution_a"].elements()))
            all_actions_b.extend(list(m["action_distribution_b"].elements()))

        aggregated["action_distribution_a"] = Counter(all_actions_a)
        aggregated["action_distribution_b"] = Counter(all_actions_b)

        return aggregated

    def print_evaluation_results(self, results: Dict):
        """Print evaluation results in a formatted way."""
        print("\n" + "=" * 60)
        print("Evaluation Results")
        print("=" * 60)
        print(f"Number of Episodes: {results['num_episodes']}")
        print(f"\nTeam Rewards (used for training):")
        print(
            f"  Agent A - Mean: {results.get('avg_team_reward_a', results['avg_reward_a']):.3f} ± "
            f"{results.get('std_team_reward_a', results['std_reward_a']):.3f}"
        )
        if "avg_team_reward_b" in results or "avg_reward_b" in results:
            print(
                f"  Agent B - Mean: {results.get('avg_team_reward_b', results['avg_reward_b']):.3f} ± "
                f"{results.get('std_team_reward_b', results['std_reward_b']):.3f}"
            )
        if "avg_individual_reward_a" in results:
            print(f"\nIndividual Rewards (for analysis):")
            print(
                f"  Agent A - Mean: {results['avg_individual_reward_a']:.3f} ± {results['std_individual_reward_a']:.3f}"
            )
            print(
                f"  Agent B - Mean: {results['avg_individual_reward_b']:.3f} ± {results['std_individual_reward_b']:.3f}"
            )
            # Calculate cooperation score
            coop_score_a = results['avg_individual_reward_a'] if results['avg_individual_reward_a'] > 0 else 0
            coop_score_b = results['avg_individual_reward_b'] if results['avg_individual_reward_b'] > 0 else 0
            print(f"\nCooperation Scores:")
            print(f"  Agent A: {coop_score_a:.3f} (higher = more cooperative)")
            print(f"  Agent B: {coop_score_b:.3f} (higher = more cooperative)")
        print(f"\nEpisode Statistics:")
        print(f"  Average Length: {results['avg_episode_length']:.1f} steps")
        print(f"\nFinal State Metrics:")
        print(f"  Emotion: {results['avg_final_emotion']:.3f}")
        print(f"  Trust: {results['avg_final_trust']:.3f}")
        print(f"  Conflict: {results['avg_final_conflict']:.3f}")
        print(f"\nTermination Rates:")
        print(f"  Success (Repaired): {results['success_rate']*100:.1f}%")
        print(f"  Failure (Broken): {results['failure_rate']*100:.1f}%")
        print(f"  Neutral (Stalemate): {results['neutral_rate']*100:.1f}%")
        print(f"\nFinal Calmness:")
        print(f"  Agent A: {results['avg_final_calmness_a']:.3f}")
        print(f"  Agent B: {results['avg_final_calmness_b']:.3f}")

        if "action_distribution_a" in results:
            print("\nAction Distribution (Agent A):")
            action_names = [ActionType(i).name for i in range(10)]
            total_a = sum(results["action_distribution_a"].values())
            for action_idx in range(10):
                count = results["action_distribution_a"][action_idx]
                percentage = (count / total_a * 100) if total_a > 0 else 0
                print(
                    f"  {action_names[action_idx]:20s}: {count:4d} ({percentage:5.1f}%)"
                )

            if "action_distribution_b" in results:
                print("\nAction Distribution (Agent B):")
                total_b = sum(results["action_distribution_b"].values())
                for action_idx in range(10):
                    count = results["action_distribution_b"][action_idx]
                    percentage = (count / total_b * 100) if total_b > 0 else 0
                    print(
                        f"  {action_names[action_idx]:20s}: {count:4d} ({percentage:5.1f}%)"
                    )

        print("=" * 60)

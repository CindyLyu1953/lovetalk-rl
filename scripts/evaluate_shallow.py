"""
Evaluation Script for Shallow RL Agents

Evaluate trained Shallow RL agents (Q-learning, SARSA) in the relationship dynamics simulator.
"""

import argparse
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from environment import RelationshipEnv
from agents.shallow_rl import QLearningAgent, SarsaAgent
from personality import PersonalityType
from training import Evaluator


def load_agent(
    agent_type: str,
    checkpoint_path: str,
    personality: PersonalityType = PersonalityType.NEUTRAL,
):
    """Load a trained Shallow RL agent from checkpoint."""
    if agent_type == "q_learning":
        agent = QLearningAgent(num_actions=10, personality=personality)
        agent.load(checkpoint_path)
        return agent
    elif agent_type == "sarsa":
        agent = SarsaAgent(num_actions=10, personality=personality)
        agent.load(checkpoint_path)
        return agent
    else:
        raise ValueError(
            f"Unknown agent type: {agent_type}. Must be 'q_learning' or 'sarsa'"
        )


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained Shallow RL agents")
    parser.add_argument(
        "--agent_type",
        type=str,
        required=True,
        choices=["q_learning", "sarsa"],
        help="Type of agent to evaluate",
    )
    parser.add_argument(
        "--checkpoint_a", type=str, required=True, help="Checkpoint path for agent A"
    )
    parser.add_argument(
        "--checkpoint_b",
        type=str,
        default=None,
        help="Checkpoint path for agent B (optional)",
    )
    parser.add_argument(
        "--personality_a",
        type=str,
        default="neutral",
        choices=["neutral", "impulsive", "sensitive", "avoidant"],
        help="Personality type for agent A",
    )
    parser.add_argument(
        "--personality_b",
        type=str,
        default="neutral",
        choices=["neutral", "impulsive", "sensitive", "avoidant"],
        help="Personality type for agent B",
    )
    parser.add_argument(
        "--num_episodes", type=int, default=100, help="Number of episodes to evaluate"
    )
    parser.add_argument(
        "--render", action="store_true", help="Render episodes during evaluation"
    )

    args = parser.parse_args()

    # Create environment (Shallow RL doesn't use history)
    env = RelationshipEnv(
        max_episode_steps=50,
        use_history=False,  # Shallow RL doesn't use history
        initial_emotion=-0.2,  # Slightly negative (matching training - conflict scenario)
        initial_trust=0.6,  # Moderate trust (matching training)
        initial_calmness_a=0.6,  # More calm (matching training)
        initial_calmness_b=0.6,  # More calm (matching training)
        irritability_a=0.7 if args.personality_a == "impulsive" else 0.4,
        irritability_b=0.7 if args.personality_b == "impulsive" else 0.4,
        recovery_rate=0.02,
    )

    # Load agents
    personality_a = PersonalityType[args.personality_a.upper()]
    personality_b = PersonalityType[args.personality_b.upper()]

    agent_a = load_agent(args.agent_type, args.checkpoint_a, personality_a)

    if args.checkpoint_b:
        agent_b = load_agent(args.agent_type, args.checkpoint_b, personality_b)
    else:
        agent_b = agent_a  # Use same agent for both

    # Evaluate
    evaluator = Evaluator(env)

    print(f"Evaluating {args.agent_type} agents...")
    print(f"  Personality A: {args.personality_a}")
    print(f"  Personality B: {args.personality_b}")
    print(f"  Episodes: {args.num_episodes}")

    results = evaluator.evaluate_multiple_episodes(
        agent_a=agent_a,
        agent_b=agent_b,
        num_episodes=args.num_episodes,
        render=args.render,
    )

    # Print results
    evaluator.print_evaluation_results(results)

    print("\nEvaluation completed!")


if __name__ == "__main__":
    main()

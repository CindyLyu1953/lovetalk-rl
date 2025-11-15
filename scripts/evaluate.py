"""
Evaluation Script

Evaluate trained agents in the relationship dynamics simulator.
"""

import argparse
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from environment import RelationshipEnv
from agents.shallow_rl import QLearningAgent, SarsaAgent
from agents.deep_rl import DQNAgent, PPOAgent
from personality import PersonalityType
from training import Evaluator


def load_agent(
    agent_type: str,
    checkpoint_path: str,
    state_dim: int = None,
    personality: PersonalityType = PersonalityType.NEUTRAL,
):
    """Load a trained agent from checkpoint."""
    if agent_type == "q_learning":
        agent = QLearningAgent(num_actions=10, personality=personality)
        agent.load(checkpoint_path)
        return agent
    elif agent_type == "sarsa":
        agent = SarsaAgent(num_actions=10, personality=personality)
        agent.load(checkpoint_path)
        return agent
    elif agent_type == "dqn":
        if state_dim is None:
            raise ValueError("state_dim required for DQN agent")
        agent = DQNAgent(state_dim=state_dim, action_dim=10, personality=personality)
        agent.load(checkpoint_path)
        return agent
    elif agent_type == "ppo":
        if state_dim is None:
            raise ValueError("state_dim required for PPO agent")
        agent = PPOAgent(state_dim=state_dim, action_dim=10, personality=personality)
        agent.load(checkpoint_path)
        return agent
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained agents")
    parser.add_argument(
        "--agent_type",
        type=str,
        required=True,
        choices=["q_learning", "sarsa", "dqn", "ppo"],
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
    parser.add_argument(
        "--history_length",
        type=int,
        default=10,
        help="Length of action history (for deep RL)",
    )

    args = parser.parse_args()

    # Determine if deep RL (needs history)
    use_history = args.agent_type in ["dqn", "ppo"]

    # Create environment
    env = RelationshipEnv(
        max_episode_steps=20,
        use_history=use_history,
        history_length=args.history_length,
        personality_a=args.personality_a,
        personality_b=args.personality_b,
    )

    # Get state dimension
    obs, _ = env.reset()
    state_dim = len(obs) if use_history else None

    # Load agents
    personality_a = PersonalityType[args.personality_a.upper()]
    personality_b = PersonalityType[args.personality_b.upper()]

    agent_a = load_agent(args.agent_type, args.checkpoint_a, state_dim, personality_a)

    if args.checkpoint_b:
        agent_b = load_agent(
            args.agent_type, args.checkpoint_b, state_dim, personality_b
        )
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

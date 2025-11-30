"""
Render Episode with LLM Dialogue

This script reads a trained RL episode trajectory and generates natural language
dialogue for visualization purposes. It does NOT affect RL training in any way.

Usage:
    python llm_extension/render_episode.py --episode_file path/to/detailed_episodes.json --episode_idx 0
"""

import argparse
import json
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from llm_extension import DialogueRenderer
from environment.actions import ActionType


# Mapping from RL scenario IDs to dialogue renderer scenario IDs
SCENARIO_MAPPING = {
    "D1": "forgot_event",  # neutral × neutral
    "D2": "busy_schedule",  # neurotic × agreeable
    "D3": "past_connection",  # neurotic × neurotic
    "D4": "time_distribution",  # neutral × avoidant
    "D5": "future_direction",  # agreeable × conscientious
}

# Mapping from ActionType enum to dialogue action labels
ACTION_MAPPING = {
    ActionType.APOLOGIZE: "apologize",
    ActionType.EMPATHIZE: "empathize",
    ActionType.EXPLAIN: "explain",
    ActionType.REASSURE: "reassure",
    ActionType.SUGGEST_SOLUTION: "suggest_solution",
    ActionType.ASK_FOR_NEEDS: "ask_for_needs",
    ActionType.CHANGE_TOPIC: "change_topic",
    ActionType.DEFENSIVE: "defensive",
    ActionType.BLAME: "blame",
    ActionType.WITHDRAW: "withdraw",
}


def render_episode_with_dialogue(
    episode_data: dict,
    renderer: DialogueRenderer,
    scenario_id: str = "forgot_event",
    verbose: bool = True,
):
    """
    Render an RL episode with natural language dialogue.

    Args:
        episode_data: Episode trajectory data from RL training
        renderer: DialogueRenderer instance
        scenario_id: Dialogue scenario ID
        verbose: Whether to print detailed information

    Returns:
        List of dialogue turns with metadata
    """
    # Extract episode information from "steps" array
    steps = episode_data.get("steps", [])

    if not steps:
        print("Warning: No steps found in episode data")
        return []

    # Extract episode metadata
    episode_num = episode_data.get("episode", 0)
    episode_length = episode_data.get("episode_length", len(steps))
    initial_state = episode_data.get("initial_state", {})
    final_state = episode_data.get("final_state", {})

    # Print episode header
    if verbose:
        print("=" * 80)
        print(f"Episode {episode_num} Visualization with Natural Language Dialogue")
        print(f"Scenario: {scenario_id}")
        print(f"Total turns: {episode_length}")
        termination = final_state.get("termination_reason", "UNKNOWN")
        print(f"Termination: {termination}")
        print("=" * 80)
        print()

    dialogue_turns = []
    conversation_history = []  # Track full conversation history

    # Generate dialogue for each turn
    for turn_idx, step_data in enumerate(steps):
        # Extract action information from step
        agent = step_data.get("agent", "A")
        action_value = step_data.get("action", 0)
        action_name = step_data.get("action_name", "")
        reward = step_data.get("reward", 0)
        state_after = step_data.get("state_after", {})

        # Convert action to dialogue label
        try:
            action_enum = ActionType(action_value)
            action_label = ACTION_MAPPING.get(action_enum, "explain")
        except (ValueError, KeyError):
            # Fallback: use action_name if available
            if action_name:
                action_label = action_name.lower()
            else:
                action_label = "explain"

        # Generate dialogue with full conversation history
        try:
            utterance = renderer.generate_reply(
                scenario_id=scenario_id,
                agent_role=agent,
                action_label=action_label,
                conversation_history=conversation_history,  # Pass full history
            )
        except Exception as e:
            print(f"Warning: Failed to generate dialogue for turn {turn_idx + 1}: {e}")
            utterance = f"[Failed to generate dialogue for action: {action_label}]"

        # Print turn
        if verbose:
            print(f"Turn {turn_idx + 1}: Character {agent} [Action={action_name}]")
            print(f"Character {agent}: {utterance}")
            print(f"  → Reward: {reward:.3f}")

            # Print state after action
            emotion = state_after.get("emotion", 0)
            trust = state_after.get("trust", 0)
            calmness_a = state_after.get("calmness_a", 0)
            calmness_b = state_after.get("calmness_b", 0)
            print(
                f"  → State: emotion={emotion:.3f}, trust={trust:.3f}, "
                f"calmness_A={calmness_a:.3f}, calmness_B={calmness_b:.3f}"
            )
            print()

        # Save turn data
        turn_data = {
            "turn": turn_idx + 1,
            "agent": agent,
            "action": action_label,
            "action_name": action_name,
            "utterance": utterance,
            "reward": reward,
            "state": state_after,
        }
        dialogue_turns.append(turn_data)

        # Add this turn to conversation history for next turn
        conversation_history.append({"agent": agent, "text": utterance})

    if verbose:
        print("=" * 80)
        print("Episode Complete")
        print("=" * 80)

    return dialogue_turns


def main():
    parser = argparse.ArgumentParser(
        description="Render RL episode with natural language dialogue"
    )
    parser.add_argument(
        "--episode_file",
        type=str,
        required=True,
        help="Path to detailed_episodes.json file",
    )
    parser.add_argument(
        "--episode_idx",
        type=int,
        default=0,
        help="Index of episode to render (default: 0)",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default=None,
        help="Dialogue scenario ID (default: infer from filename)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file to save rendered dialogue (optional)",
    )

    args = parser.parse_args()

    # Load episode data
    episode_file = Path(args.episode_file)
    if not episode_file.exists():
        print(f"Error: Episode file not found: {episode_file}")
        return

    with open(episode_file, "r") as f:
        episodes_data = json.load(f)

    # Get specific episode
    if isinstance(episodes_data, list):
        if args.episode_idx >= len(episodes_data):
            print(
                f"Error: Episode index {args.episode_idx} out of range (max: {len(episodes_data) - 1})"
            )
            return
        episode_data = episodes_data[args.episode_idx]
    else:
        episode_data = episodes_data

    # Infer scenario from filename if not provided
    scenario_id = args.scenario
    if scenario_id is None:
        # Try to extract from path (e.g., experiments/D1/...)
        for exp_id in SCENARIO_MAPPING.keys():
            if exp_id in str(episode_file):
                scenario_id = SCENARIO_MAPPING[exp_id]
                break
        if scenario_id is None:
            scenario_id = "forgot_event"  # Default

    print(f"Loading episode from: {episode_file}")
    print(f"Episode index: {args.episode_idx}")
    print(f"Dialogue scenario: {scenario_id}")
    print()

    # Initialize renderer
    try:
        renderer = DialogueRenderer()
    except ValueError as e:
        print(f"Error: {e}")
        print("Please set GEMINI_API_KEY environment variable.")
        return

    # Render episode
    dialogue_turns = render_episode_with_dialogue(
        episode_data=episode_data,
        renderer=renderer,
        scenario_id=scenario_id,
        verbose=True,
    )

    # Save output if requested
    if args.output:
        output_data = {
            "scenario": scenario_id,
            "episode_file": str(episode_file),
            "episode_idx": args.episode_idx,
            "turns": dialogue_turns,
        }
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        print(f"\nDialogue saved to: {args.output}")


if __name__ == "__main__":
    main()

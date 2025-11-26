"""
Debug Environment Script

Diagnose why episodes terminate immediately by checking initial state and first action effects.
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from environment import RelationshipEnv
from environment.actions import ActionType


def debug_environment():
    """Debug environment initialization and first action effects."""
    print("=" * 80)
    print("Environment Debug - Initial State & Termination Analysis")
    print("=" * 80)
    print()

    # Create environment with default settings
    env = RelationshipEnv(
        max_episode_steps=50,
        use_history=False,
        initial_emotion=-0.3,
        initial_trust=0.5,
        initial_calmness_a=0.4,
        initial_calmness_b=0.4,
        irritability_a=0.4,
        irritability_b=0.4,
        recovery_rate=0.02,
    )

    # Reset environment
    obs, info = env.reset()

    print("Initial State:")
    print(f"  Emotion: {info['emotion']:.3f}")
    print(f"  Trust: {info['trust']:.3f}")
    print(f"  Conflict: {info['conflict']:.3f}")
    print(f"  Calmness A: {info['calmness_a']:.3f}")
    print(f"  Calmness B: {info['calmness_b']:.3f}")
    print()

    # Check termination conditions
    terminated, reason = env._check_termination()
    print(f"Termination Check: {terminated}, Reason: {reason}")
    print()

    if terminated:
        print("WARNING: Environment starts in terminated state!")
        print()

    # Test each action's effect
    print("Testing Action Effects (from initial state):")
    print("=" * 80)

    test_actions = list(ActionType)

    for action in test_actions:
        # Reset to initial state
        env.reset()
        prev_state = env.state.copy()

        # Take action
        obs, reward, terminated, truncated, info = env.step(action.value)

        # Calculate changes
        delta_emotion = info["emotion"] - prev_state.emotion_level
        delta_trust = info["trust"] - prev_state.trust_level
        delta_conflict = info["conflict"] - prev_state.conflict_intensity
        delta_calmness_a = info["calmness_a"] - prev_state.calmness_a

        # Check if terminated
        term_status = "[TERMINATED]" if terminated else "[OK]"

        print(
            f"{action.name:20s} | "
            f"ΔEmotion: {delta_emotion:6.3f} → {info['emotion']:6.3f} | "
            f"ΔTrust: {delta_trust:6.3f} → {info['trust']:6.3f} | "
            f"ΔConflict: {delta_conflict:6.3f} → {info['conflict']:6.3f} | "
            f"Reward: {reward:6.3f} | {term_status}"
        )

    print()
    print("=" * 80)
    print("Termination Thresholds:")
    print(f"  SUCCESS: emotion > 0.7 AND trust > 0.75")
    print(f"  FAILURE: emotion < -0.9 OR trust < 0.1 (FIXED: relaxed from -0.8/0.2)")
    print()

    # Count how many actions trigger termination
    terminating_actions = []
    for action in test_actions:
        env.reset()
        _, _, terminated, _, _ = env.step(action.value)
        if terminated:
            terminating_actions.append(action.name)

    print(
        f"Actions that trigger immediate termination: {len(terminating_actions)}/{len(test_actions)}"
    )
    if terminating_actions:
        print(f"  {', '.join(terminating_actions)}")
    print()

    # Suggest fixes
    print("=" * 80)
    print("Suggested Fixes:")
    print("=" * 80)
    print("1. Make initial state more moderate:")
    print("   - initial_emotion = 0.0 (neutral start)")
    print("   - initial_trust = 0.6 (moderate trust)")
    print("   - initial_calmness = 0.6 (more calm)")
    print()
    print("2. Relax termination thresholds:")
    print("   - FAILURE: emotion < -0.9 OR trust < 0.1")
    print()
    print("3. Reduce negative action impacts:")
    print("   - Scale down negative action deltas by 0.5-0.7")
    print()
    print("4. Increase reward for positive actions:")
    print("   - Add larger bonus for positive actions")
    print("   - Penalize immediate termination more")
    print()


if __name__ == "__main__":
    debug_environment()

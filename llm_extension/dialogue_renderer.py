"""
LLM Dialogue Renderer Module for LoveTalk-RL

FICTIONAL CHARACTER SIMULATION:
This module generates dialogue for imaginary characters (Character A and Character B)
practicing various communication patterns. All scenarios are purely fictional and have
no connection to real people or real relationships. No emotional or personal harm is possible.

This module is COMPLETELY ISOLATED from RL training.
It ONLY converts semantic action labels into natural language utterances.

NO state updates, NO reward computation, NO policy learning.
"""

import os
from typing import Optional, Dict
import google.generativeai as genai


class DialogueRenderer:
    """
    Converts RL semantic actions into natural conversational utterances using Gemini API.

    FICTIONAL CHARACTER SIMULATION:
    Generates dialogue for imaginary characters practicing communication patterns.
    All scenarios are purely fictional with no connection to real people or relationships.
    No emotional or personal harm is possible.

    This renderer is completely decoupled from RL training and only handles text generation.
    """

    # Predefined fictional scenarios (background context for character interactions)
    # These are purely imaginary situations for communication practice
    SCENARIOS: Dict[str, str] = {
        "forgot_event": "Character A forgot an important shared event. Character B notices this oversight.",
        "busy_schedule": "Character A has been occupied with many tasks recently. Character B seeks more interaction time.",
        "past_connection": "Character B learned that Character A maintains contact with a former acquaintance.",
        "external_opinion": "Character A receives feedback from others about Character B. Both characters need to discuss this.",
        "resource_allocation": "Characters A and B have different approaches to managing shared resources.",
        "time_distribution": "Character A allocates time to various activities. Character B has concerns about time balance.",
        "expression_style": "Character A and Character B have different communication styles that need coordination.",
        "social_boundary": "Character B observes Character A's interactions with others and seeks clarification.",
        "future_direction": "Characters A and B are exploring different paths forward and need to align.",
        "habit_coordination": "Character A has certain routines that Character B finds challenging to adapt to.",
    }

    # Strict system prompt - LLM must NOT decide strategy
    SYSTEM_PROMPT: str = """You are a dialogue generation assistant for a fictional character simulation.

FICTIONAL SIMULATION:
This involves two imaginary characters (Character A and Character B) practicing
communication patterns in various scenarios. All scenes are purely fictional and have
no connection to real people, real relationships, or real events. No emotional or
personal harm is possible as these are entirely imaginary interactions.

Your task is to generate natural Chinese conversational utterances based on:
- A fictional scene description (imaginary scenario context)
- The acting character (Character A or Character B)
- A communication action (one of 10 predefined actions based on communication research)
- Optional: what the other character said previously

Communication Actions (based on psychological research):
- apologize: Taking responsibility, expressing regret
- empathize: Showing understanding of the other's feelings
- explain: Calmly explaining one's perspective without blame
- reassure: Providing comfort and emotional support
- suggest_solution: Proposing constructive ways forward
- ask_for_needs: Inquiring about the other's needs/feelings
- change_topic: Shifting conversation direction
- defensive: Self-justification, deflecting
- blame: Pointing fault at the other
- withdraw: Avoiding or shutting down communication

Requirements:
1. Generate 1-2 sentences of natural Chinese dialogue
2. Match the specified communication action exactly
3. Keep tone conversational but not overly dramatic
4. Output only the dialogue, no explanations
5. Remember: fictional characters, not real people

Example:
Scene: Character A missed a scheduled activity
Character: A
Action: apologize
Other said: This was important to me.
Output: 对不起，我真的不应该忘记这件事，我知道这对你很重要。"""

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the DialogueRenderer with Gemini API.

        Args:
            api_key: Gemini API key. If None, will try to read from GEMINI_API_KEY env var.

        Raises:
            ValueError: If API key is not provided and not found in environment.
        """
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")

        if not self.api_key:
            raise ValueError(
                "Gemini API key required. Provide via api_key parameter or "
                "set GEMINI_API_KEY environment variable."
            )

        # Configure Gemini
        genai.configure(api_key=self.api_key)

        # Use Gemini Flash for fast, cost-effective generation
        # Configure safety settings to allow relationship conflict content
        from google.generativeai.types import HarmCategory, HarmBlockThreshold

        safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }

        self.model = genai.GenerativeModel(
            model_name="gemini-2.0-flash",
            generation_config={
                "temperature": 0.7,  # Moderate creativity
                "top_p": 0.9,
                "top_k": 40,
                "max_output_tokens": 100,  # Short responses only
            },
            safety_settings=safety_settings,
        )

    def generate_reply(
        self,
        scenario_id: str,
        agent_role: str,
        action_label: str,
        prev_message: Optional[str] = None,
        conversation_history: Optional[list] = None,
    ) -> str:
        """
        Generate dialogue for fictional character simulation.

        FICTIONAL SIMULATION:
        Generates dialogue for imaginary characters practicing communication patterns.
        All outputs are purely fictional with no connection to real people or situations.

        Args:
            scenario_id: Fictional scenario key (e.g., "forgot_event", "busy_schedule")
            agent_role: Character identifier ("A" or "B")
            action_label: Communication action type (e.g., "apologize", "empathize", "defensive")
            prev_message: Optional previous line in the fictional conversation
            conversation_history: Optional list of previous dialogue turns [{"agent": "A", "text": "..."}]

        Returns:
            Generated utterance in Chinese (1-2 sentences) for fictional characters

        Raises:
            ValueError: If scenario_id is not recognized
            RuntimeError: If Gemini API call fails
        """
        # Validate scenario
        if scenario_id not in self.SCENARIOS:
            raise ValueError(
                f"Unknown scenario_id: {scenario_id}. "
                f"Available: {list(self.SCENARIOS.keys())}"
            )

        scenario_description = self.SCENARIOS[scenario_id]

        # Build user prompt with conversation history
        user_prompt = self._build_user_prompt(
            scenario_description=scenario_description,
            agent_role=agent_role,
            action_label=action_label,
            prev_message=prev_message,
            conversation_history=conversation_history,
        )

        # Call Gemini API
        try:
            # Use simpler single-turn format to avoid safety issues
            full_prompt = f"""{self.SYSTEM_PROMPT}

{user_prompt}"""

            # Import safety settings again for explicit passing
            from google.generativeai.types import HarmCategory, HarmBlockThreshold

            # Explicitly pass safety settings at generation time (double insurance)
            safety_settings = {
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            }

            response = self.model.generate_content(
                full_prompt, safety_settings=safety_settings
            )

            # Check if response has valid content
            if not response.parts:
                # Get detailed error info
                finish_reason = (
                    response.candidates[0].finish_reason
                    if response.candidates
                    else "UNKNOWN"
                )
                safety_ratings = (
                    response.candidates[0].safety_ratings if response.candidates else []
                )
                error_msg = f"Content blocked. Finish reason: {finish_reason}"
                if safety_ratings:
                    error_msg += f"\nSafety ratings: {safety_ratings}"
                error_msg += "\n\nTip: Try adjusting the scenario description or action label to be less explicit."
                raise RuntimeError(error_msg)

            utterance = response.text.strip()

            # Clean up any potential markdown or formatting
            utterance = utterance.replace("**", "").replace("*", "")

            return utterance

        except Exception as e:
            raise RuntimeError(f"Gemini API call failed: {e}") from e

    def _build_user_prompt(
        self,
        scenario_description: str,
        agent_role: str,
        action_label: str,
        prev_message: Optional[str],
        conversation_history: Optional[list] = None,
    ) -> str:
        """
        Build the user prompt for Gemini API.

        Args:
            scenario_description: English description of the fictional scene
            agent_role: "A" or "B"
            action_label: Semantic action name
            prev_message: Optional previous message
            conversation_history: Optional list of previous dialogue turns

        Returns:
            Formatted prompt string
        """
        prompt_parts = [
            f"Fictional scene: {scenario_description}",
            f"Acting character: Character {agent_role}",
            f"Communication action: {action_label}",
        ]

        # Add conversation history if available
        if conversation_history and len(conversation_history) > 0:
            prompt_parts.append("\nPrevious conversation:")
            for turn in conversation_history:
                speaker = turn.get("agent", "?")
                text = turn.get("text", "")
                prompt_parts.append(f"  Character {speaker}: {text}")
        elif prev_message:
            # Fallback to single previous message if history not available
            prompt_parts.append(f"Other character just said: {prev_message}")

        prompt_parts.append("\nGenerate dialogue for this fictional character:")
        prompt_parts.append("(Remember: Keep it natural)")

        return "\n".join(prompt_parts)

    def add_scenario(self, scenario_id: str, description: str) -> None:
        """
        Add a custom fictional scenario.

        Args:
            scenario_id: Unique scenario identifier
            description: English description of the fictional scene
        """
        self.SCENARIOS[scenario_id] = description

    def list_scenarios(self) -> Dict[str, str]:
        """
        Get all available fictional scenarios.

        Returns:
            Dictionary mapping scenario IDs to fictional scenario descriptions
        """
        return self.SCENARIOS.copy()


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    """
    Simple demonstration of dialogue generation.

    NOTE: This is a POST-TRAINING visualization tool.
    - Does NOT participate in RL training
    - Reads episode trajectories AFTER training
    - Generates dialogue based on trained agent actions

    For rendering actual trained episodes, use:
        python llm_extension/render_episode.py --episode_file <path> --episode_idx <idx>

    FICTIONAL SIMULATION:
    These examples demonstrate dialogue generation for imaginary characters
    (Character A and Character B) practicing various communication patterns.
    All scenarios are purely fictional with no connection to real people or situations.

    To run this demo:
    1. Set your Gemini API key:
       export GEMINI_API_KEY="your-api-key-here"

    2. Run the script:
       python llm_extension/dialogue_renderer.py
    """

    # Option 1: Use environment variable
    # export GEMINI_API_KEY="your-api-key-here"
    renderer = DialogueRenderer()

    # Option 2: Pass API key directly
    # renderer = DialogueRenderer(api_key="YOUR_API_KEY")

    print("=" * 70)
    print("LLM Dialogue Renderer - Demo")
    print("=" * 70)
    print("NOTE: This is a POST-TRAINING visualization tool")
    print("      Use render_episode.py to visualize actual trained episodes")
    print("=" * 70)
    print()

    # Example 1: Multi-turn dialogue episode in one scene
    print("Example 1: Multi-turn Episode - Forgot Event Scene")
    print("Scene: Character A missed a scheduled shared activity")
    print("=" * 70)
    print()

    scenario = "forgot_event"

    # Turn 1: B initiates with blame
    print("Turn 1: Character B [Action=BLAME]")
    b_msg_1 = renderer.generate_reply(
        scenario_id=scenario, agent_role="B", action_label="blame", prev_message=None
    )
    print(f"Character B: {b_msg_1}")
    print()

    # Turn 2: A responds with apologize
    print("Turn 2: Character A [Action=APOLOGIZE]")
    a_msg_1 = renderer.generate_reply(
        scenario_id=scenario,
        agent_role="A",
        action_label="apologize",
        prev_message=b_msg_1,
    )
    print(f"Character A: {a_msg_1}")
    print()

    # Turn 3: B empathizes
    print("Turn 3: Character B [Action=EMPATHIZE]")
    b_msg_2 = renderer.generate_reply(
        scenario_id=scenario,
        agent_role="B",
        action_label="empathize",
        prev_message=a_msg_1,
    )
    print(f"Character B: {b_msg_2}")
    print()

    # Turn 4: A explains
    print("Turn 4: Character A [Action=EXPLAIN]")
    a_msg_2 = renderer.generate_reply(
        scenario_id=scenario,
        agent_role="A",
        action_label="explain",
        prev_message=b_msg_2,
    )
    print(f"Character A: {a_msg_2}")
    print()

    # Turn 5: B suggests solution
    print("Turn 5: Character B [Action=SUGGEST_SOLUTION]")
    b_msg_3 = renderer.generate_reply(
        scenario_id=scenario,
        agent_role="B",
        action_label="suggest_solution",
        prev_message=a_msg_2,
    )
    print(f"Character B: {b_msg_3}")
    print()

    # Turn 6: A reassures
    print("Turn 6: Character A [Action=REASSURE]")
    a_msg_3 = renderer.generate_reply(
        scenario_id=scenario,
        agent_role="A",
        action_label="reassure",
        prev_message=b_msg_3,
    )
    print(f"Character A: {a_msg_3}")
    print()

    print("=" * 70)
    print("Episode Complete - Characters worked through the situation")
    print("=" * 70)
    print()

    # Example 2: Another episode with different dynamics
    print("\nExample 2: Multi-turn Episode - Busy Schedule Scene")
    print("Scene: Character A occupied with tasks, B seeks interaction")
    print("=" * 70)
    print()

    scenario2 = "busy_schedule"

    # Turn 1: B asks for needs
    print("Turn 1: Character B [Action=ASK_FOR_NEEDS]")
    b2_msg_1 = renderer.generate_reply(
        scenario_id=scenario2,
        agent_role="B",
        action_label="ask_for_needs",
        prev_message=None,
    )
    print(f"Character B: {b2_msg_1}")
    print()

    # Turn 2: A is defensive
    print("Turn 2: Character A [Action=DEFENSIVE]")
    a2_msg_1 = renderer.generate_reply(
        scenario_id=scenario2,
        agent_role="A",
        action_label="defensive",
        prev_message=b2_msg_1,
    )
    print(f"Character A: {a2_msg_1}")
    print()

    # Turn 3: B explains calmly
    print("Turn 3: Character B [Action=EXPLAIN]")
    b2_msg_2 = renderer.generate_reply(
        scenario_id=scenario2,
        agent_role="B",
        action_label="explain",
        prev_message=a2_msg_1,
    )
    print(f"Character B: {b2_msg_2}")
    print()

    # Turn 4: A apologizes
    print("Turn 4: Character A [Action=APOLOGIZE]")
    a2_msg_2 = renderer.generate_reply(
        scenario_id=scenario2,
        agent_role="A",
        action_label="apologize",
        prev_message=b2_msg_2,
    )
    print(f"Character A: {a2_msg_2}")
    print()

    print("=" * 70)
    print("Episode Complete")
    print("=" * 70)
    print()

    # List all available scenarios
    print("=" * 70)
    print("Available Fictional Scenarios:")
    print("=" * 70)
    for scenario_id, description in renderer.list_scenarios().items():
        print(f"- {scenario_id}: {description}")
    print()

    print("=" * 70)
    print("Fictional character dialogue examples completed!")
    print("All scenarios are purely imaginary with no real-world connection.")
    print("=" * 70)

# LLM Dialogue Renderer Extension

## Overview

This is a **completely independent LLM extension module** that converts semantic actions selected by RL policies into natural language dialogue.

**EDUCATIONAL PURPOSE:**
This module simulates couple arguments to provide guidance on how to better resolve conflicts, manage emotions, and strengthen relationships during disagreements. All scenarios are educational examples for learning purposes and do not cause real-world harm.

**Core Principle: Complete Isolation**
- Does NOT modify RL state
- Does NOT compute rewards
- Does NOT influence policy learning
- ONLY generates natural language text

---

## API Key Configuration

### Method 1: Environment Variable (Recommended)

Create a `.env` file in the project root (or modify your shell configuration):

```bash
export GEMINI_API_KEY="your-api-key-here"
```

Then load it before use:

```bash
source .env
```

### Method 2: Pass Directly in Code

```python
from llm_extension import DialogueRenderer

renderer = DialogueRenderer(api_key="your-api-key-here")
```

### Getting an API Key

1. Visit: https://makersuite.google.com/app/apikey
2. Create a new API key
3. Copy it to the configuration above

---

## Scenario Design

### What are "Scenarios"?

**Scenarios = Simulated argument backgrounds/premises**, not current dialogue states.

This is for educational simulation purposes only.

Example:
- Scenario: "A forgot their anniversary, B feels disappointed" (why the argument started)
- NOT a scenario: "Both are currently arguing" (dialogue state)

### Same Scenario Throughout Conversation

In one simulated conflict resolution process, the scenario remains constant:
- Scenario is always "A forgot anniversary"
- Dialogue state changes: "Tension" → "Clarification" → "Resolution" → "Reconciliation"
- But the argument background (scenario) stays the same

### 10 Built-in Educational Simulation Scenarios

| Scenario ID | Educational Description |
|-------------|-------------------------|
| `forgot_anniversary` | A forgot their anniversary, B feels disappointed and neglected |
| `work_neglect` | A has been too busy with work, neglecting B who feels lonely |
| `trust_issue` | B discovered A is still in contact with ex, feels insecure |
| `family_conflict` | A's family has issues with B, A caught in the middle |
| `money_dispute` | Couple disagrees on money usage |
| `time_priority` | A often goes out with friends, B feels less important |
| `communication_gap` | A doesn't express emotions well, B feels neglected |
| `jealousy` | B sees A has close relationship with opposite-sex friend |
| `life_plan` | Couple has different views on future plans |
| `habit_clash` | A has life habits that B is dissatisfied with |

All scenarios are educational simulations for teaching conflict resolution.

---

## Usage Example

### Basic Usage

```python
from llm_extension import DialogueRenderer

# Initialize renderer
from llm_extension import DialogueRenderer

renderer = DialogueRenderer()  # Requires GEMINI_API_KEY env var

utterance = renderer.generate_reply(
    scenario_id="forgot_anniversary",   # Conflict background
    agent_role="A",                     # You are agent A
    action_label="apologize",           # The RL policy selected this action
    prev_message="You forgot our anniversary?"  # What agent B just said
)

# Output: "I'm really sorry, I truly forgot, and I know that hurt you."
```

### Full Conversation Example

```python
# Scenario: A forgot anniversary (educational simulation)
scenario = "forgot_anniversary"

# Turn 1: B speaks
b_msg_1 = renderer.generate_reply(
    scenario_id=scenario,
    agent_role="B",
    action_label="express_hurt",
    prev_message=None
)
print(f"B: {b_msg_1}")

# Turn 2: A responds
a_msg_1 = renderer.generate_reply(
    scenario_id=scenario,
    agent_role="A",
    action_label="apologize",
    prev_message=b_msg_1
)
print(f"A: {a_msg_1}")

# ... Continue educational simulation
```

---

## Integration with RL System

### Action Mapping

Map RL `ActionType` enum to dialogue labels:

```python
from environment.actions import ActionType
from llm_extension import DialogueRenderer

ACTION_MAPPING = {
    ActionType.APOLOGIZE: "apologize",
    ActionType.EMPATHIZE: "empathize",
    ActionType.REASSURE: "reassure",
    ActionType.EXPLAIN: "explain",
    ActionType.SUGGEST_SOLUTION: "suggest_solution",
    ActionType.ASK_FOR_NEEDS: "ask_for_needs",
    ActionType.DEFENSIVE: "defensive",
    ActionType.BLAME: "blame",
    ActionType.WITHDRAW: "withdraw",
    ActionType.CHANGE_TOPIC: "change_topic",
}
```

---

## Testing

Run the educational example script:

```bash
# Set API key
export GEMINI_API_KEY="your-key-here"

# Run educational examples
python llm_extension/dialogue_renderer.py
```

Expected output will show educational simulation examples for learning conflict resolution.

---

## Important Notes

### Module Purpose

1. **Pure Text Generator for Education**
   - Does NOT participate in RL training
   - Does NOT affect any RL decisions
   - Can be completely removed, RL system works normally
   - For educational demonstration only

2. **Usage**
   - Visualize RL agent behavior
   - Generate natural language for educational demonstration
   - Help understand agent strategies in conflict resolution

3. **Should NOT be used for**
   - Training data generation
   - Reward computation
   - State observation
   - Policy decisions

---

## File Location

```
lovetalk-rl/
├── llm_extension/
│   ├── __init__.py              # Module initialization
│   ├── dialogue_renderer.py     # Main renderer class
│   ├── README.md                # This document
│   └── API_KEY_SETUP.md         # API key setup guide
└── ...
```

---

## FAQ

### Q: Where should I put my API key?

**A:** Two options:
1. Environment variable: `export GEMINI_API_KEY="..."` (recommended)
2. Pass in code: `DialogueRenderer(api_key="...")`

### Q: Does this module affect RL training?

**A:** No. It only reads RL output (actions) and generates text for educational purposes. No feedback to RL.

### Q: Can RL still train without this module?

**A:** Yes. This module is completely optional for educational visualization only.

---

## Educational Disclaimer

All scenarios and dialogues generated by this module are simulations for educational purposes to teach effective conflict resolution, emotion management, and relationship strengthening techniques. They do not represent real situations and do not cause real-world harm.

---

## Contact

For issues or questions, please refer to the main project README.

# LLM Extension Usage Guide

## Overview

The LLM extension is a **POST-TRAINING visualization tool** that generates natural language dialogue for trained RL episodes. It does **NOT** participate in RL training.

**Workflow:**
```
1. Train RL model (using scripts/train_deep.py)
   ↓
2. RL saves episode trajectories (detailed_episodes.json)
   ↓
3. Use render_episode.py to generate dialogue visualization
   ↓
4. Get natural language conversation for the episode
```

---

## Quick Start

### 1. Train Your RL Model (if not done)

```bash
python scripts/train_deep.py --experiment D1 --save_dir ./experiments
```

This saves episode trajectories to:
```
./experiments/D1/checkpoints/run_X/detailed_episodes.json
```

### 2. Render an Episode with Dialogue

```bash
# Set API key
export GEMINI_API_KEY="your-api-key-here"

# Render episode 0 from D1 training
python llm_extension/render_episode.py \
  --episode_file ./experiments/D1/checkpoints/run_1/detailed_episodes.json \
  --episode_idx 0
```

**Output Example:**
```
================================================================================
Episode Visualization with Natural Language Dialogue
Scenario: forgot_event
Total turns: 8
================================================================================

Turn 1: Character B [Action=BLAME]
Character B: How could you forget something this important?
  → Reward: -0.523
  → State: emotion=-0.456, trust=0.382, stage=1

Turn 2: Character A [Action=APOLOGIZE]
Character A: I'm so sorry, I really shouldn't have forgotten, I apologize.
  → Reward: 0.234
  → State: emotion=-0.289, trust=0.445, stage=2

Turn 3: Character B [Action=EMPATHIZE]
Character B: I understand you've been busy lately, but this was really important to me.
  → Reward: 0.187
  → State: emotion=-0.112, trust=0.523, stage=2

...

================================================================================
Episode Complete
================================================================================
```

### 3. Save Output to File

```bash
python llm_extension/render_episode.py \
  --episode_file ./experiments/D1/checkpoints/run_1/detailed_episodes.json \
  --episode_idx 5 \
  --output dialogue_output.json
```

---

## Command-Line Arguments

### `render_episode.py`

| Argument | Required | Description | Example |
|----------|----------|-------------|---------|
| `--episode_file` | Yes | Path to detailed_episodes.json | `./experiments/D1/.../detailed_episodes.json` |
| `--episode_idx` | No | Episode index to render (default: 0) | `5` |
| `--scenario` | No | Dialogue scenario (auto-detected from path) | `forgot_event` |
| `--output` | No | Save dialogue to JSON file | `dialogue.json` |

---

## Scenario Mapping

The script automatically maps RL experiment IDs to dialogue scenarios:

| RL Experiment | Dialogue Scenario | Description |
|---------------|-------------------|-------------|
| D1 | forgot_event | Character A missed scheduled activity |
| D2 | busy_schedule | Character A occupied with tasks |
| D3 | past_connection | Questions about past connections |
| D4 | time_distribution | Time balance concerns |
| D5 | future_direction | Different future paths |

---

## Episode Data Format

The script expects episode data in this format:

```json
{
  "actions": [
    {"agent": "A", "action_type": 0},  // 0 = APOLOGIZE
    {"agent": "B", "action_type": 1},  // 1 = EMPATHIZE
    ...
  ],
  "rewards": [0.234, -0.123, ...],
  "states": [
    {"emotion": -0.3, "trust": 0.5, "stage": 1},
    ...
  ]
}
```

This is automatically saved by your RL training script.

---

## Action Mapping

RL actions are mapped to dialogue labels:

| RL ActionType | Dialogue Label | Description |
|---------------|----------------|-------------|
| APOLOGIZE (0) | apologize | Taking responsibility |
| EMPATHIZE (1) | empathize | Showing understanding |
| EXPLAIN (2) | explain | Calmly explaining |
| REASSURE (3) | reassure | Providing comfort |
| SUGGEST_SOLUTION (4) | suggest_solution | Proposing solutions |
| ASK_FOR_NEEDS (5) | ask_for_needs | Inquiring about needs |
| CHANGE_TOPIC (6) | change_topic | Shifting topic |
| DEFENSIVE (7) | defensive | Self-justification |
| BLAME (8) | blame | Pointing fault |
| WITHDRAW (9) | withdraw | Avoiding communication |

---

## Examples

### Example 1: Render First Episode from D1

```bash
python llm_extension/render_episode.py \
  --episode_file ./experiments/D1/checkpoints/run_1/detailed_episodes.json \
  --episode_idx 0
```

### Example 2: Render and Save Episode 10 from D3

```bash
python llm_extension/render_episode.py \
  --episode_file ./experiments/D3/checkpoints/run_1/detailed_episodes.json \
  --episode_idx 10 \
  --output d3_episode_10_dialogue.json
```

### Example 3: Render with Custom Scenario

```bash
python llm_extension/render_episode.py \
  --episode_file ./experiments/D2/checkpoints/run_1/detailed_episodes.json \
  --episode_idx 3 \
  --scenario busy_schedule
```

---

## Important Notes

1. **Post-Training Only**: This tool reads saved episode trajectories. It does NOT affect RL training.

2. **Episode Length**: The dialogue length matches the actual episode length (could be 3 turns, 10 turns, 20 turns, etc.)

3. **API Key Required**: You need a Gemini API key to generate dialogue.

4. **Fictional Characters**: All generated dialogue is for fictional characters A and B.

---

## Troubleshooting

### Error: "No module named 'llm_extension'"

Make sure you're running from the project root:
```bash
cd /path/to/lovetalk-rl
python llm_extension/render_episode.py ...
```

### Error: "Gemini API key required"

Set your API key:
```bash
export GEMINI_API_KEY="your-key-here"
```

### Error: "Episode file not found"

Check the path to your detailed_episodes.json file. It should be in:
```
./experiments/{experiment_id}/checkpoints/run_{X}/detailed_episodes.json
```

---

## Demo

To see a simple demo (not using real episode data):

```bash
python llm_extension/dialogue_renderer.py
```

This shows basic dialogue generation capabilities.


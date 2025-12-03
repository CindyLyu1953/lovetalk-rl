# LoveTalk-RL: Multi-Agent Conflict Resolution with Double DQN

Authors: Chenxi Guo, Jiayi Peng, Kexin Lyu, Yiran Tao

A multi-agent reinforcement learning system for relationship conflict resolution using Double DQN with soft target updates.

---

## Table of Contents

1. [Overview](#overview)
2. [Model Architecture](#model-architecture)
3. [Algorithm](#algorithm)
4. [Environment](#environment)
5. [Installation](#installation)
6. [Quick Start](#quick-start)
7. [LLM Dialogue Renderer](#llm-dialogue-renderer-optional)
8. [Project Structure](#project-structure)
9. [References](#references)

---

## Overview

This project implements a multi-agent reinforcement learning system where two agents learn to resolve relationship conflicts through dialogue. The system uses **Double DQN with Soft Target Updates (Polyak averaging)** for stable and efficient learning.

### Key Features

- **Double DQN:** Reduces Q-value overestimation bias
- **Soft Target Updates:** Polyak averaging for smooth learning
- **Large Replay Buffer:** 100,000 transitions for better sample efficiency
- **Repair Stage Inference:** Automatic stage detection (4 stages: Tension, Clarification, Problem-Solving, Closure)
- **Stage-Based Reward Shaping:** Contextual guidance for appropriate actions
- **Multi-Agent Coordination:** Team reward for cooperative learning
- **Personality System:** 5 personality types affecting behavior

---

## Model Architecture

### Double DQN Agent

**Network Architecture:**
```
Input: State (15D) = [emotion, trust, conflict, calmness, stage] + [history (10)]
  ↓
Hidden Layer 1: 128 neurons + ReLU
  ↓
Hidden Layer 2: 128 neurons + ReLU
  ↓
Output: Q-values for 10 actions
```

**Hyperparameters:**
```python
learning_rate = 3e-4
discount_factor = 0.99
epsilon_start = 1.0
epsilon_decay = 0.998
epsilon_min = 0.1
batch_size = 64
memory_size = 100000  # Large replay buffer
tau = 0.005  # Soft update parameter
```

---

## Algorithm

### Double DQN with Soft Target Updates

**1. Double DQN Target Computation:**
```python
# Standard DQN (Overestimates):
target = r + γ * max_a Q(s', a; θ_target)

# Double DQN (Reduced Bias):
a* = argmax_a Q(s', a; θ_online)  # Select with online network
target = r + γ * Q(s', a*; θ_target)  # Evaluate with target network
```

**2. Soft Target Update (Polyak Averaging):**
```python
# After each optimization step:
θ_target ← τ * θ_online + (1 - τ) * θ_target

# With τ = 0.005:
# - 0.5% from online network
# - 99.5% from target network
# - Smooth, continuous updates
```

**3. Team Reward for Multi-Agent Learning:**
```python
team_reward = state_change_reward + action_reward + stage_shaping + terminal_reward

# State change: Δemotion, Δtrust, Δconflict
# Action reward: +0.2 for cooperative, -0.2 for aggressive
# Stage shaping: Contextual guidance (±1.0)
# Terminal reward: +30 SUCCESS, -20 FAILURE, -10 NEUTRAL
```

---

## Environment

### State Space

**Core State (5D):**
- `emotion`: [-1, 1] - Emotional valence
- `trust`: [0, 1] - Trust level
- `conflict`: [0, 1] - Conflict intensity
- `calmness`: [0, 1] - Agent's calmness
- `stage`: [0, 1] - Repair stage (normalized)

**Full State (15D):** Core + Action History (10 recent actions)

### Action Space

10 discrete actions:
1. `APOLOGIZE` - Express apology
2. `EMPATHIZE` - Show empathy
3. `EXPLAIN` - Explain perspective
4. `REASSURE` - Provide reassurance
5. `SUGGEST_SOLUTION` - Propose solution
6. `ASK_FOR_NEEDS` - Inquire about needs
7. `CHANGE_TOPIC` - Change subject
8. `DEFENSIVE` - Defensive response
9. `BLAME` - Blame other
10. `WITHDRAW` - Withdraw from conversation

### Repair Stages

1. **Stage 1 - Tension/Eruption** (`emotion < -0.3`)
   - Optimal: EMPATHIZE, REASSURE
   - Avoid: EXPLAIN, SUGGEST_SOLUTION

2. **Stage 2 - Clarification** (`-0.3 ≤ emotion < 0`)
   - Optimal: EXPLAIN
   - Avoid: Withdrawal

3. **Stage 3 - Problem-Solving** (`emotion ≥ 0, trust < 0.6`)
   - Optimal: SUGGEST_SOLUTION
   - Avoid: Overthinking

4. **Stage 4 - Closure** (`emotion ≥ 0, trust ≥ 0.6`)
   - Optimal: APOLOGIZE, ASK_FOR_NEEDS
   - Maintain: REASSURE

### Termination Conditions

- **SUCCESS:** `emotion > 0.2 AND trust > 0.6`
- **FAILURE:** `emotion < -0.5 OR trust < 0.1`
- **NEUTRAL:** Max steps (50) reached without resolution

---

## Installation

### Requirements

```bash
Python 3.8+
PyTorch 1.9+
NumPy
Gymnasium
PyYAML
```

### Setup

```bash
# Clone repository
git clone <repository-url>
cd lovetalk-rl

# Install dependencies
pip install -r requirements.txt
```

---

## Quick Start

### Training Commands

**Set environment variable (avoid OMP error):**
```bash
export OMP_NUM_THREADS=1
```

**Train a single experiment (D1-D5):**
```bash
python scripts/train_deep.py --experiment D1 --save_dir ./experiments
```

**Train all experiments (D1-D5):**
```bash
python scripts/train_deep.py --all --save_dir ./experiments
```

**Experiments:**
- **D1:** Neutral × Neutral (baseline)
- **D2:** Neurotic × Agreeable (conflict scenario)
- **D3:** Neurotic × Neurotic (extreme conflict)
- **D4:** Neutral × Avoidant (cold war scenario)
- **D5:** Agreeable × Conscientious (cooperative scenario)

---

### Evaluation Commands

**Evaluate a single run:**
```bash
python scripts/evaluate_single_run.py \
  --experiment D1 \
  --run_id 1 \
  --num_episodes 100
```

**Evaluate all runs (D1-D5, run_1 to run_15):**
```bash
python scripts/evaluate_all_runs.py
```

---

### Generate Natural Language Dialogue

**Set Gemini API Key:**
```bash
export GEMINI_API_KEY="your-api-key-here"
```

**Render dialogue for a specific episode:**
```bash
python llm_extension/render_episode.py \
  --episode_file ./experiments/D1/checkpoints/run_1/detailed_episodes.json \
  --episode_idx 0
```

**Render and save to file:**
```bash
python llm_extension/render_episode.py \
  --episode_file ./experiments/D1/checkpoints/run_1/detailed_episodes.json \
  --episode_idx 0 \
  --output dialogue_output.json
```

**Specify scenario:**
```bash
python llm_extension/render_episode.py \
  --episode_file ./experiments/D1/checkpoints/run_1/detailed_episodes.json \
  --episode_idx 0 \
  --scenario forgot_event
```

**View LLM module demo:**
```bash
python llm_extension/dialogue_renderer.py
```

---

### Example Long Episodes

Based on trained models, here are some interesting long episodes you can render:

```bash
# D1 - Longest episode (16 steps, SUCCESS)
python llm_extension/render_episode.py \
  --episode_file ./experiments/D1/checkpoints/run_9/detailed_episodes.json \
  --episode_idx 41 \
  --scenario forgot_event

# D4 - Longest in project (25 steps, SUCCESS)
python llm_extension/render_episode.py \
  --episode_file ./experiments/D4/checkpoints/run_9/detailed_episodes.json \
  --episode_idx 35 \
  --scenario time_distribution

# D4 - Long failure case (22 steps, FAILURE)
python llm_extension/render_episode.py \
  --episode_file ./experiments/D4/checkpoints/run_9/detailed_episodes.json \
  --episode_idx 42 \
  --scenario time_distribution
```

---

### Results

Results are saved in:
```
experiments/
├── D1/
│   ├── checkpoints/
│   │   ├── run_1/
│   │   │   ├── agent_a_ep2000.pth
│   │   │   ├── agent_a_ep4000.pth
│   │   │   ├── agent_b_ep2000.pth
│   │   │   ├── agent_b_ep4000.pth
│   │   │   ├── train_stats.json
│   │   │   ├── detailed_episodes.json
│   │   │   └── evaluation_results.json
│   │   ├── run_2/...
│   │   └── run_15/...
│   ├── aggregated_evaluation_D1.json
│   └── evaluation_deep_D1.json
├── D2/...
├── D3/...
├── D4/...
└── D5/...
```

---

## LLM Dialogue Renderer (Optional Extension)

A standalone module for converting RL semantic actions into natural language dialogue using Gemini API.

**Important:** This module is **completely isolated** from RL training. It does NOT affect state transitions, rewards, or policy learning. It is purely for post-training visualization.

**Key Features:**
- **Completely Isolated:** Does NOT affect RL training, rewards, or state
- **LLM-Powered:** Uses Gemini 2.0 Flash for fast, natural text generation
- **Scenario-Aware:** 10 built-in fictional scenarios (e.g., forgot_event, busy_schedule, time_distribution)
- **Episode Rendering:** Converts full episode trajectories into natural dialogue
- **Conversation Context:** Maintains dialogue history for coherent multi-turn conversations
- **English Output:** Generates natural English conversational utterances

### What are "Scenarios"?

**Scenarios = Fixed background context for the entire episode**, not individual dialogue states.

Example:
- Scenario: "Character A forgot an important shared event. Character B notices this oversight." (Fixed premise)
- Not a scenario: "Currently arguing" (This is a dynamic state within the episode)

All dialogue turns in a single episode occur within the same fixed scenario context.

### Quick Example

```python
from llm_extension import DialogueRenderer

renderer = DialogueRenderer()  # Requires GEMINI_API_KEY env var

utterance = renderer.generate_reply(
    scenario_id="forgot_event",        # Fictional scenario background
    agent_role="A",                    # Character A
    action_label="apologize",          # RL action chosen by agent
    prev_message="How could you forget something this important?"  # Previous utterance
)
# Output: "I'm so sorry, I really shouldn't have forgotten this, I know it was important to you."
```

### Setup

```bash
# Install Gemini API
pip install google-generativeai

# Set API key (in your shell or .bashrc/.zshrc)
export GEMINI_API_KEY="your-api-key-here"

# Run demo
python llm_extension/dialogue_renderer.py
```

**Full Documentation:** See [`llm_extension/README.md`](llm_extension/README.md) and [`llm_extension/USAGE.md`](llm_extension/USAGE.md)

---

## Project Structure

```
lovetalk-rl/
├── README.md                      # This file
├── requirements.txt               # Python dependencies
├── .gitignore                     # Git ignore rules
│
├── agents/                        # RL Agents
│   └── deep_rl/
│       └── dqn.py                 # Double DQN implementation
│
├── environment/                   # Environment
│   ├── relationship_env.py        # Main environment class
│   ├── state.py                   # State representation
│   ├── actions.py                 # Action definitions
│   ├── transition_model.py        # State transition logic
│   └── action_feasibility.py      # Action selection constraints
│
├── personality/                   # Personality system
│   └── personality_policy.py      # Personality types and effects
│
├── training/                      # Training utilities
│   ├── trainer.py                 # Multi-agent trainer
│   └── evaluator.py               # Evaluation utilities
│
├── scripts/                       # Training & evaluation scripts
│   ├── train_deep.py              # Train Double DQN agents
│   ├── evaluate_deep.py           # Evaluate trained agents
│   ├── evaluate_single_run.py     # Evaluate single run
│   └── evaluate_all_runs.py       # Aggregate evaluation across all runs
│
├── llm_extension/                 # LLM Extension (optional, completely isolated)
│   ├── __init__.py
│   ├── dialogue_renderer.py       # Natural language generator (Gemini API)
│   ├── render_episode.py          # Render episode trajectories as dialogue
│   ├── README.md                  # Extension documentation
│   └── USAGE.md                   # Detailed usage guide
│
└── experiments/                   # Training results
    ├── D1/                        # Experiment 1 (Neutral × Neutral)
    │   ├── checkpoints/
    │   │   ├── run_1/...          # 15 independent runs
    │   │   └── run_15/...
    │   ├── aggregated_evaluation_D1.json
    │   └── evaluation_deep_D1.json
    ├── D2/...                     # Experiment 2 (Neurotic × Agreeable)
    ├── D3/...                     # Experiment 3 (Neurotic × Neurotic)
    ├── D4/...                     # Experiment 4 (Neutral × Avoidant)
    ├── D5/...                     # Experiment 5 (Agreeable × Conscientious)
    ├── all_results.json           # Combined results
    └── comparison_table.csv       # Comparison table
```

---

## Key Components

### 1. Double DQN Agent (`agents/deep_rl/dqn.py`)

Implements Double DQN with:
- Online network for action selection
- Target network for value evaluation
- Soft target updates (Polyak averaging)
- Large replay buffer (100,000 transitions)
- Epsilon-greedy exploration

### 2. Relationship Environment (`environment/relationship_env.py`)

Features:
- Turn-based multi-agent interaction
- Dynamic state updates (emotion, trust, conflict, calmness)
- Repair stage inference (4 stages)
- Stage-based reward shaping
- Team reward for cooperative learning
- Personality-specific transitions

### 3. Multi-Agent Trainer (`training/trainer.py`)

Supports:
- Self-play training
- Alternating agent turns
- Experience replay for both agents
- Periodic checkpointing
- Detailed episode logging

### 4. Personality System (`personality/personality_policy.py`)

Personality types:
- **NEUTRAL:** Balanced behavior
- **NEUROTIC:** High irritability, emotional
- **AGREEABLE:** Cooperative, low irritability
- **CONSCIENTIOUS:** Systematic, solution-focused
- **AVOIDANT:** Withdrawal tendency

Each personality affects:
- Action effect ranges
- Irritability (calmness decay)
- Action preferences

---

## Training Details

### Training Configuration

```yaml
num_episodes: 4000       # Training episodes per run
train_mode: self_play    # Both agents learn simultaneously
log_interval: 200        # Log every 200 episodes
save_interval: 2000      # Save checkpoints every 2000 episodes
repeats: 15              # 15 independent training runs per experiment
```

### Initial Conditions

```yaml
initial_emotion: -0.3    # Slight negative emotion (conflict situation)
initial_trust: 0.4       # Lower trust (challenging scenario)
initial_calmness: 0.4    # Moderate calmness
max_episode_steps: 50    # Maximum steps per episode
```

### Expected Performance

After training (4000 episodes):
- **Success Rate:** Varies by experiment (10-40%)
- **Average Episode Length:** 15-30 steps
- **Epsilon (final):** ~0.26 (with decay 0.998)
- **Training Time:** ~1-2 hours for all 5 experiments (15 runs each)

---

## References

### Reinforcement Learning

1. **Double DQN:**
   - van Hasselt, H., Guez, A., & Silver, D. (2016). "Deep Reinforcement Learning with Double Q-learning." *AAAI*.
   - Reduces Q-value overestimation by decoupling action selection and evaluation

2. **Soft Target Updates:**
   - Lillicrap, T. P., et al. (2015). "Continuous control with deep reinforcement learning." *ICLR*.
   - Polyak averaging for smooth target network updates

3. **Experience Replay:**
   - Mnih, V., et al. (2015). "Human-level control through deep reinforcement learning." *Nature*.
   - Breaks temporal correlations in training data

4. **Prioritized Experience Replay:**
   - Schaul, T., et al. (2015). "Prioritized Experience Replay." *ICLR*.
   - Improves sample efficiency (not implemented, but related work)

### Multi-Agent RL

5. **Value Decomposition (QMIX/VDN):**
   - Rashid, T., et al. (2018). "QMIX: Monotonic Value Function Factorisation for Decentralised Multi-Agent Reinforcement Learning." *ICML*.
   - Sunehag, P., et al. (2018). "Value-Decomposition Networks For Cooperative Multi-Agent Learning." *AAMAS*.
   - Inspiration for team reward design

6. **Counterfactual Multi-Agent (COMA):**
   - Foerster, J., et al. (2018). "Counterfactual Multi-Agent Policy Gradients." *AAAI*.
   - Inspiration for individual reward analysis

### Psychology & Relationship Research

7. **Gottman's Four Horsemen:**
   - Gottman, J. M., & Silver, N. (2015). "The Seven Principles for Making Marriage Work."
   - Foundation for negative action effects (BLAME, DEFENSIVE, WITHDRAW)

8. **Nonviolent Communication (NVC):**
   - Rosenberg, M. B. (2015). "Nonviolent Communication: A Language of Life."
   - Foundation for positive action effects (EMPATHIZE, ASK_FOR_NEEDS)

9. **Emotion Regulation:**
   - Gross, J. J. (1998). "The emerging field of emotion regulation: An integrative review." *Review of General Psychology*.
   - Basis for calmness mechanic and emotional dynamics

10. **Conflict Resolution Stages:**
    - Thomas, K. W., & Kilmann, R. H. (1974). "Thomas-Kilmann Conflict Mode Instrument."
    - Inspiration for repair stage design

---

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{lovetalk-rl-2025,
  title={LoveTalk-RL: Multi-Agent Conflict Resolution with Double DQN},
  author={Your Name},
  year={2025},
  howpublished={\url{https://github.com/yourusername/lovetalk-rl}}
}
```

---

## License

MIT License

---

## Acknowledgments

- Double DQN implementation inspired by van Hasselt et al. (2016)
- Multi-agent reward design inspired by QMIX (Rashid et al., 2018) and COMA (Foerster et al., 2018)
- Relationship dynamics grounded in Gottman's research and NVC principles
- Environment design influenced by emotion regulation theory (Gross, 1998)

---

## Contact

For questions or issues, please open an issue on GitHub or contact the authors.

---

**Last Updated:** December 3, 2024  
**Version:** 2.0 (Double DQN + Soft Target Updates + Stage-Based Reward Shaping)

# Code Structure Documentation

This document provides a comprehensive overview of the codebase structure for the Relationship Dynamics Simulator project.

## Table of Contents
1. [Project Overview](#project-overview)
2. [Directory Structure](#directory-structure)
3. [Core Components](#core-components)
4. [Key Modules](#key-modules)
5. [Data Flow](#data-flow)
6. [Experiment Configuration](#experiment-configuration)

---

## Project Overview

This is a **Multi-Agent Reinforcement Learning** project that simulates relationship communication dynamics between two agents. The project explores how agents with different personality types learn effective communication strategies to reduce conflict, improve emotional stability, and increase trust.

**Core Technologies:**
- Python 3.12+
- PyTorch (for Deep RL)
- NumPy (for numerical operations)
- Gym-like environment interface
- YAML configuration

---

## Directory Structure

```
lovetalk-rl/
├── environment/          # Core environment implementation
│   ├── __init__.py
│   ├── actions.py           # Action space definitions (10 discrete actions)
│   ├── state.py             # State space definitions (emotion, trust, conflict, calmness)
│   ├── transition_model.py  # State transition dynamics
│   ├── relationship_env.py  # Main Gym-like environment class
│   └── action_feasibility.py # Action feasibility based on calmness
│
├── agents/               # RL agent implementations
│   ├── __init__.py
│   ├── shallow_rl/       # Tabular RL methods (discrete state space)
│   │   ├── q_learning.py # Q-learning agent
│   │   └── sarsa.py      # SARSA agent
│   └── deep_rl/          # Deep RL methods (continuous state space)
│       ├── dqn.py        # Deep Q-Network (DQN) agent
│       └── ppo.py        # Proximal Policy Optimization (legacy, not used)
│
├── personality/          # Personality policy system
│   ├── __init__.py
│   └── personality_policy.py # Personality types and action bias
│
├── training/             # Training and evaluation utilities
│   ├── __init__.py
│   ├── trainer.py        # Multi-agent training loop
│   └── evaluator.py      # Agent evaluation utilities
│
├── scripts/              # Executable scripts
│   ├── train_shallow.py      # Train Shallow RL agents (Q-learning, SARSA)
│   ├── train_deep.py         # Train Deep RL agents (DQN)
│   ├── evaluate_shallow.py   # Evaluate Shallow RL agents
│   ├── evaluate_deep.py      # Evaluate Deep RL agents
│   ├── evaluate_all.py       # Batch evaluation for all experiments
│   ├── run_all_experiments.py # Batch training for all experiments
│   ├── collect_results.py    # Collect and aggregate results
│   ├── visualize_results.py  # Generate visualization plots
│   ├── debug_environment.py  # Debug environment behavior
│   └── clear_experiments.py  # Clean up experiment results
│
├── config/               # Configuration files
│   ├── __init__.py
│   └── config.yaml       # Main configuration file
│
├── data/                 # Data loading utilities (optional)
│   ├── __init__.py
│   ├── data_loader.py    # Load dialogue datasets
│   └── calibrator.py     # Calibrate transition model from data
│
├── utils/                # Utility modules
│   ├── __init__.py
│   └── visualizer.py     # Visualization utilities
│
├── experiments/          # Experiment results (generated)
│   ├── S1/ ... S6/       # Shallow RL experiment results
│   ├── D1/ ... D5/       # Deep RL experiment results
│   └── figures/          # Generated plots
│
├── checkpoints/          # Legacy checkpoints (deprecated)
│
├── QUICK_START.md        # Quick start guide
├── README.md             # Project overview
├── CODE_STRUCTURE.md     # This file
├── requirements.txt      # Python dependencies
└── RUN_EXPERIMENTS.sh    # Shell script to run all experiments
```

---

## Core Components

### 1. Environment (`environment/`)

The environment is a **Gym-like** turn-based two-agent communication simulator.

#### **`relationship_env.py`** - Main Environment Class
- **Class**: `RelationshipEnv`
- **Inherits**: `gym.Env`
- **Purpose**: Implements the core RL environment interface
- **Key Methods**:
  - `reset()`: Initialize environment state
  - `step(action)`: Execute action and return (observation, reward, terminated, truncated, info)
  - `_compute_reward()`: Calculate reward (supports both Shallow and Deep RL reward functions)
  - `_check_termination()`: Check termination conditions (SUCCESS, FAILURE, NEUTRAL/stalemate)

**State Representation:**
- **Shallow RL**: Discrete state space (quantized emotion, trust, conflict, calmness)
- **Deep RL**: Continuous state space (raw float values + optional history)

**Reward Functions:**
- **Shallow RL**: Simple reward based on state changes
- **Deep RL**: 4-part reward function:
  1. Continuous state change: `Δemotion + Δtrust - 0.5*Δconflict`
  2. Action-level: Cooperative (+0.05), Aggressive (-0.05), Withdraw (conditional ±0.02)
  3. Termination: Success (+2.0), Failure (-2.0), Stalemate (-0.2)
  4. Clipped to [-3.0, 3.0]

#### **`actions.py`** - Action Space
- **10 Discrete Actions**:
  - **Positive**: `APOLOGIZE`, `EMPATHIZE`, `EXPLAIN`, `REASSURE`, `SUGGEST_SOLUTION`, `ASK_FOR_NEEDS`
  - **Neutral**: `CHANGE_TOPIC`
  - **Negative**: `DEFENSIVE`, `BLAME`, `WITHDRAW`
- **Action Categories** (for Deep RL reward):
  - `COOPERATIVE_ACTIONS`, `AGGRESSIVE_ACTIONS`, `WITHDRAW_ACTIONS`

#### **`state.py`** - State Space
- **Class**: `RelationshipState`
- **Attributes**:
  - `emotion_level`: [-1, 1] (negative to positive)
  - `trust_level`: [0, 1]
  - `conflict_intensity`: [0, 1]
  - `calmness_a`, `calmness_b`: [0, 1] (per-agent calmness)

#### **`transition_model.py`** - State Dynamics
- **Class**: `TransitionModel`
- **Purpose**: Updates state based on actions and personality traits
- **Key Method**: `update_state(state, action_a, action_b, calmness_a, calmness_b, irritability_a, irritability_b)`
- Models the psychological effects of actions on relationship state
 - **Sampling**: Action effects are sampled per-run from personality-specific ranges using Beta distributions (configurable via `config/config.yaml`) and the environment supports seeded RNG for reproducibility.

#### **`action_feasibility.py`** - Action Constraints
- **Class**: `ActionFeasibility`
- **Purpose**: Modifies action probabilities based on agent's calmness
- When calmness is low, aggressive actions become more likely (emotional instability)

---

### 2. Agents (`agents/`)

#### **Shallow RL** (`agents/shallow_rl/`)

**Q-Learning** (`q_learning.py`):
- **Class**: `QLearningAgent`
- **State Space**: Discrete (quantized)
- **Learning**: Tabular Q-table updates
- **Formula**: `Q(s,a) ← Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]`

**SARSA** (`sarsa.py`):
- **Class**: `SarsaAgent`
- **State Space**: Discrete (quantized)
- **Learning**: On-policy Q-table updates
- **Formula**: `Q(s,a) ← Q(s,a) + α[r + γ Q(s',a') - Q(s,a)]`

#### **Deep RL** (`agents/deep_rl/`)

**DQN** (`dqn.py`):
- **Class**: `DQNAgent`
- **State Space**: Continuous (4-dimensional or 4+history)
- **Architecture**: Feedforward neural network (MLP)
- **Features**:
  - Experience replay buffer
  - Target network (updated every N steps)
  - Epsilon-greedy exploration
  - **Default Hyperparameters** (from config):
    - Learning rate: 3e-4
    - Discount factor: 0.99
    - Epsilon: 1.0 → 0.05 (decay: 0.997)
    - Batch size: 64
    - Memory size: 20000
    - Target update frequency: 200

**PPO** (`ppo.py`):
- **Status**: Legacy (not actively used)
- Available but not part of current experiments

---

### 3. Personality System (`personality/`)

**`personality_policy.py`**:
- **Enum**: `PersonalityType` (NEUTRAL, IMPULSIVE, SENSITIVE, AVOIDANT)
- **Class**: `PersonalityPolicy`
- **Purpose**: 
  - Modifies state perception (bias)
  - Adds action bias to Q-values
  - Affects action feasibility

**Personality Types:**
- **Neutral**: No bias, balanced behavior
- **Impulsive**: High irritability, tendency toward aggressive actions
- **Sensitive**: Reactive to negative states, volatile trust
- **Avoidant**: Tendency to withdraw or change topic

---

### 4. Training System (`training/`)

**`trainer.py`**:
- **Class**: `Trainer`
- **Purpose**: Manages multi-agent training loop
- **Training Modes**:
  - `self_play`: Both agents learn simultaneously
  - `fixed_opponent`: Agent A learns against fixed Agent B
- **Features**:
  - Checkpoint saving
  - Training metrics logging
  - Progress tracking

**`evaluator.py`**:
- **Class**: `Evaluator`
- **Purpose**: Evaluates trained agents
- **Metrics Collected**:
  - Episode rewards
  - Episode lengths
  - Termination reasons (SUCCESS/FAILURE/NEUTRAL)
  - Action distributions
  - State trajectories

---

## Key Modules

### Scripts (`scripts/`)

#### Training Scripts

**`train_shallow.py`**:
- Train Shallow RL agents (Q-learning or SARSA)
- Arguments: algorithm, episodes, personalities, train_mode
- Outputs: Checkpoints, training logs

**`train_deep.py`**:
- Train Deep RL agents (DQN)
- Supports training single experiment (`--experiment D1`) or all (`--all`)
- **5 Experiments**: D1 (neutral×neutral), D2 (impulsive×sensitive), D3 (impulsive×impulsive), D4 (neutral×avoidant), D5 (sensitive×sensitive)
- Outputs: Checkpoints (ep2000, ep4000, ..., ep8000), metadata, training logs

**`run_all_experiments.py`**:
- Batch training for all 12 experiments (7 Shallow + 5 Deep)
- Handles experiment orchestration

#### Evaluation Scripts

**`evaluate_shallow.py`**:
- Evaluate Shallow RL agents
- Loads checkpoints and runs evaluation episodes
- Outputs: JSON results with metrics

**`evaluate_deep.py`**:
- Evaluate Deep RL agents
- Supports single or batch evaluation (D1-D5)
- **Outputs**:
  - Per-experiment: `experiments/D{1-5}/evaluation_deep_D{1-5}.json`
  - Aggregated: `experiments/deep_rl_evaluation_results.json`
  - **Metrics**: Success/Failure/Stalemate rates, reward statistics, episode lengths, convergence curves

**`evaluate_all.py`**:
- Batch evaluation for all experiments
- Automatically detects agent type and calls appropriate evaluator

#### Utility Scripts

**`collect_results.py`**:
- Aggregates results from all experiments
- Generates comparison tables

**`visualize_results.py`**:
- Generates plots (learning curves, termination rates, comparisons)
- Outputs: PDF and PNG files in `experiments/figures/`

**`debug_environment.py`**:
- Debug tool to inspect environment behavior
- Tests initial states, termination conditions, action effects

**`clear_experiments.py`**:
- Clean up experiment results directory

---

## Data Flow

### Training Flow

```
1. Initialize Environment
   ├── Set initial state (emotion, trust, conflict, calmness)
   └── Configure termination thresholds

2. Initialize Agents
   ├── Load personalities
   ├── Initialize Q-table (Shallow) or Neural Network (Deep)
   └── Set exploration parameters (epsilon)

3. Training Loop (for each episode)
   ├── Reset environment
   ├── Episode Loop (until termination)
   │   ├── Agent A selects action (epsilon-greedy)
   │   ├── Agent B selects action (epsilon-greedy or fixed)
   │   ├── Environment step (action_a, action_b)
   │   │   ├── Update state via TransitionModel
   │   │   ├── Check termination
   │   │   └── Compute reward
   │   ├── Agents observe (state, reward, done)
   │   └── Update Q-values / Neural Networks
   ├── Log episode metrics
   └── Save checkpoints (periodically)

4. Save Final Model
   └── Save checkpoint to experiments/{exp_id}/checkpoints/
```

### Evaluation Flow

```
1. Load Trained Agents
   ├── Load checkpoints
   └── Set evaluation mode (no exploration)

2. Evaluation Loop (N episodes)
   ├── Reset environment
   ├── Episode Loop (until termination)
   │   ├── Agents select actions (greedy)
   │   ├── Environment step
   │   └── Record state trajectory
   ├── Record episode metrics
   │   ├── Reward
   │   ├── Episode length
   │   ├── Termination reason
   │   └── Final state
   └── Aggregate statistics

3. Generate Results
   ├── Success/Failure/Stalemate rates
   ├── Average rewards and episode lengths
   ├── Action distributions
   └── Convergence curves (for Deep RL)
```

---

## Experiment Configuration

### 12 Core Experiments

**Shallow RL (7 experiments):**
- **S1**: Q-learning, neutral × neutral (Baseline)
- **S2**: Q-learning, impulsive × sensitive (Intense conflict)
- **S3**: Q-learning, impulsive × impulsive (Extreme conflict)
- **S4**: Q-learning, neutral × avoidant (Cold war)
- **S5**: Q-learning, sensitive × sensitive (Mutual misunderstanding)
- **S6**: Q-learning, fixed_opponent, impulsive × sensitive
- **S2_SARSA**: SARSA, impulsive × sensitive (Algorithm comparison)

**Deep RL (5 experiments):**
- **D1**: DQN, neutral × neutral (Baseline)
- **D2**: DQN, impulsive × sensitive (Intense conflict)
- **D3**: DQN, impulsive × impulsive (Extreme conflict)
- **D4**: DQN, neutral × avoidant (Cold war)
- **D5**: DQN, sensitive × sensitive (Mutual misunderstanding)

### Configuration File (`config/config.yaml`)

**Environment Settings:**
- `max_episode_steps`: 20
- Initial state values (emotion, trust, calmness)
- Termination thresholds
- Reward weights

**Deep RL Settings:**
- Learning rate: 3e-4
- Discount factor: 0.99
- Epsilon decay: 0.997
- Batch size: 64
- Memory size: 20000
- Target update frequency: 200

**Training Settings:**
- `num_episodes`: 8000 (Deep RL), 5000 (Shallow RL)
- `train_mode`: self_play
- `log_interval`: 200
- `save_interval`: 2000

---

## File Locations Summary

### Input/Configuration
- **Config**: `config/config.yaml`
- **Experiment Plans**: Defined in `scripts/train_deep.py`, `scripts/run_all_experiments.py`

### Output/Results
- **Checkpoints**: `experiments/{exp_id}/checkpoints/agent_{a,b}_ep{N}.pth`
- **Training Logs**: `experiments/{exp_id}/training_log_{exp_id}.txt`
- **Metadata**: `experiments/{exp_id}/metadata.json`
- **Evaluation Results**: 
  - Individual: `experiments/{exp_id}/evaluation_deep_{exp_id}.json`
  - Aggregated: `experiments/deep_rl_evaluation_results.json`
  - All experiments: `experiments/all_results.json`
- **Figures**: `experiments/figures/*.pdf`, `*.png`

---

## Key Design Decisions

1. **Dual Reward Functions**: Separate reward functions for Shallow and Deep RL to optimize for different objectives
2. **Modular Architecture**: Clear separation between environment, agents, training, and evaluation
3. **Experiment-Driven**: Scripts organized around experiment IDs (S1-S6, D1-D5)
4. **Personality as Bias**: Personality affects perception and action selection, not environment dynamics
5. **Action Feasibility**: Low calmness increases probability of negative actions (emotional instability)

---

## Extension Points

1. **New RL Algorithms**: Add to `agents/deep_rl/` or `agents/shallow_rl/`
2. **New Personality Types**: Extend `PersonalityType` enum in `personality/personality_policy.py`
3. **New Actions**: Add to `ActionType` enum in `environment/actions.py`
4. **Custom Reward Functions**: Modify `_compute_reward()` in `relationship_env.py`
5. **New Experiments**: Add to experiment configs in training scripts

---

## Quick Reference

**Train Deep RL (all experiments):**
```bash
python scripts/train_deep.py --all --save_dir ./experiments
```

**Evaluate Deep RL (all experiments):**
```bash
python scripts/evaluate_deep.py --checkpoint_dir ./experiments --num_episodes 100
```

**View results:**
- Individual: `experiments/D{1-5}/evaluation_deep_D{1-5}.json`
- Summary: `experiments/deep_rl_evaluation_results.json`

For more details, see `QUICK_START.md`.

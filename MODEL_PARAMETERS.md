# 🎯 Complete Model Parameters

**Last Updated**: November 29, 2025  
**Model Type**: Multi-Agent Deep Reinforcement Learning (DQN with Team Reward)

---

## 📊 Executive Summary

| Category | Key Value |
|----------|-----------|
| **Algorithm** | Deep Q-Network (DQN) |
| **Training Mode** | Self-Play |
| **Total Episodes** | 8,000 per run |
| **Independent Runs** | 15 (different seeds) |
| **Total Training** | 120,000 episodes per scenario |
| **State Dimension** | 14 (with history) |
| **Action Dimension** | 10 (discrete) |
| **Network Architecture** | 14 → 128 → 128 → 10 |

---

## 🏗️ 1. Environment Parameters

### State Space (14 dimensions)
```yaml
Core State (5 dimensions):
  - emotion_level: [-1.0, 1.0]         # Emotional valence
  - trust_level: [0.0, 1.0]            # Trust level
  - conflict_intensity: [0.0, 1.0]     # Conflict intensity
  - calmness_a: [0.0, 1.0]             # Agent A's calmness
  - calmness_b: [0.0, 1.0]             # Agent B's calmness

Action History (9 dimensions):
  - last_10_actions: one-hot encoded and compressed
```

### Action Space (10 discrete actions)
```yaml
Positive Actions (6):
  0. APOLOGIZE          - Taking responsibility
  1. EMPATHIZE          - Expressing understanding
  2. EXPLAIN            - Calmly explaining
  3. REASSURE           - Providing comfort
  4. SUGGEST_SOLUTION   - Proposing solutions
  5. ASK_FOR_NEEDS      - Inquiring about needs

Neutral Actions (1):
  6. CHANGE_TOPIC       - Shifting conversation

Negative Actions (3):
  7. DEFENSIVE          - Self-defense
  8. BLAME              - Blaming partner
  9. WITHDRAW           - Silent treatment
```

### Initial State
```yaml
initial_emotion: -0.2              # Slightly negative (conflict scenario)
initial_trust: 0.6                 # Moderate trust
initial_calmness_a: 0.6            # More calm (prevents immediate termination)
initial_calmness_b: 0.6            # More calm
initial_conflict: 0.4              # Calculated from emotion
```

### Environment Dynamics
```yaml
max_episode_steps: 50              # Maximum steps per episode
history_length: 10                 # Action history length
use_history: true                  # Include history in state (for Deep RL)

recovery_rate: 0.02                # Automatic calmness recovery per step
cross_agent_calmness_factor: 0.6   # Cross-agent calmness influence (60%)

feasibility_alpha: 1.0             # Weight for calmness in action feasibility
feasibility_beta: 1.0              # Weight for action difficulty
```

### Termination Conditions
```yaml
SUCCESS (episode ends):
  emotion > 0.4 AND trust > 0.6    # Moderate repair achieved

FAILURE (episode ends):
  emotion < -0.5 OR trust < 0.1    # Extreme conflict / very low trust

NEUTRAL/STALEMATE (episode ends):
  step >= 50                       # Max steps reached without resolution
```

---

## 🧠 2. DQN Network Architecture

### Q-Network Structure
```python
Input Layer:    14 neurons  (state dimension)
                ↓
Hidden Layer 1: 128 neurons + ReLU
                ↓
Hidden Layer 2: 128 neurons + ReLU
                ↓
Output Layer:   10 neurons  (Q-values for 10 actions)

Total Parameters: ~18,000
```

### Network Details
```yaml
Activation Function: ReLU
Optimizer: Adam
Loss Function: MSE (Mean Squared Error)
Gradient Clipping: 1.0
Device: CUDA (if available), else CPU
```

---

## 🎓 3. DQN Algorithm Parameters

### Core DQN Hyperparameters
```yaml
learning_rate: 3e-4                # Adam optimizer learning rate
discount_factor: 0.99              # Gamma (γ) for future reward discounting
batch_size: 64                     # Mini-batch size for experience replay
memory_size: 20000                 # Replay buffer capacity
target_update_freq: 200            # Target network update frequency (steps)
```

### Exploration Strategy (ε-greedy)
```yaml
epsilon_initial: 1.0               # Initial exploration rate (100%)
epsilon_min: 0.05                  # Minimum exploration rate (5%)
epsilon_decay: 0.997               # Decay rate per episode
                                   # ε(t) = max(ε_min, ε(t-1) * decay)

Decay Schedule:
  Episode 0:    ε = 1.000 (100% random)
  Episode 1000: ε ≈ 0.050 (5% random)
  Episode 2000: ε = 0.050 (5% random, stays at min)
  Episode 8000: ε = 0.050 (5% random)
```

### Experience Replay
```yaml
Replay Buffer Type: Deque (FIFO)
Buffer Size: 20,000 transitions
Sampling: Uniform random sampling
Minimum Buffer Size: 64 (= batch_size)
```

### Target Network
```yaml
Update Type: Hard update (full copy)
Update Frequency: Every 200 steps
Purpose: Stabilize training by providing fixed Q-targets
```

---

## 💰 4. Reward Function (Team Reward)

### Team Reward Formula
```python
team_reward = r_state + r_action + r_terminal + emotion_crossing_bonus + emotion_progress_bonus
team_reward = clip(team_reward, -5.0, 5.0)
```

### Component 1: State Change Reward
```python
# Dynamic emotion weight
emotion_weight = 3.0 if emotion < 0 else 2.5

r_state = emotion_weight * Δemotion + 1.0 * Δtrust - 0.5 * Δconflict
```

### Component 2: Emotion Crossing Bonus
```python
if prev_emotion < 0 and curr_emotion >= 0:
    emotion_crossing_bonus = 0.6
else:
    emotion_crossing_bonus = 0.0
```

### Component 3: Emotion Progress Bonus
```python
if prev_emotion < 0 and curr_emotion < 0 and Δemotion > 0:
    progress_bonus_scale = abs(prev_emotion) * 0.3
    emotion_progress_bonus = progress_bonus_scale * (Δemotion / 0.3)
    emotion_progress_bonus = min(emotion_progress_bonus, 0.2)
else:
    emotion_progress_bonus = 0.0
```

### Component 4: Action-Level Reward
```python
Cooperative Actions (APOLOGIZE, EMPATHIZE, REASSURE, SUGGEST_SOLUTION, ASK_FOR_NEEDS):
    base_reward = 0.20
    if emotion < 0:
        r_action = base_reward + 0.25  # Total: +0.45
    else:
        r_action = base_reward          # Total: +0.20

Aggressive Actions (DEFENSIVE, BLAME):
    r_action = -0.20

Withdraw Actions (WITHDRAW, CHANGE_TOPIC):
    if conflict >= 0.6:
        r_action = +0.02   # Strategic withdrawal in high conflict
    else:
        r_action = -0.05   # Avoidance in moderate conflict
```

### Component 5: Termination Reward
```python
if terminated:
    if termination_reason == "SUCCESS":
        r_terminal = +4.0
    elif termination_reason == "FAILURE":
        r_terminal = -4.0
    elif termination_reason == "NEUTRAL":
        r_terminal = -0.2
else:
    r_terminal = 0.0
```

### Reward Bounds
```yaml
Total Team Reward: [-5.0, 5.0]
Individual Reward: [-2.0, 2.0]   # For evaluation only, not training
```

---

## 🏋️ 5. Training Configuration

### Training Schedule
```yaml
num_episodes: 8000                 # Episodes per independent run
num_repeats: 15                    # Independent runs with different seeds
total_episodes: 120000             # Per scenario (8000 × 15)

train_mode: self_play              # Both agents learn simultaneously
log_interval: 200                  # Log statistics every 200 episodes
save_interval: 2000                # Save checkpoints every 2000 episodes
```

### Random Seeds
```yaml
base_seed: 42
run_seeds: [42, 43, 44, ..., 56]   # base_seed + run_index

Purpose:
  - Ensures reproducibility
  - Provides statistical robustness (15 independent runs)
  - Allows variance estimation
```

### Checkpoints Saved
```yaml
Checkpoints at Episodes: [2000, 4000, 6000, 8000]

Per Checkpoint:
  - agent_a_ep{N}.pth    # Agent A's Q-network + optimizer state
  - agent_b_ep{N}.pth    # Agent B's Q-network + optimizer state
  - train_stats.json     # Training statistics
  - detailed_episodes.json  # Detailed episode data (selected episodes)
```

---

## 🧪 6. Experiment Scenarios (D1-D5)

| ID | Personality A | Personality B | Irritability A | Irritability B | Description |
|----|---------------|---------------|----------------|----------------|-------------|
| **D1** | neutral | neutral | 0.4 | 0.4 | Baseline |
| **D2** | neurotic | agreeable | 0.5 | 0.3 | Conflict scenario |
| **D3** | neurotic | neurotic | 0.5 | 0.5 | Extreme conflict |
| **D4** | neutral | avoidant | 0.4 | 0.3 | Cold war |
| **D5** | agreeable | conscientious | 0.3 | 0.3 | Cooperative |

### Personality Effects
```yaml
Irritability:
  - Controls emotional reactivity
  - Higher = more volatile calmness changes
  - Affects action effect sampling (via personality-specific ranges)

Personality-Specific Action Effects:
  - Each personality has different effect ranges for each action
  - Example: Neurotic has larger emotion swings from same action
  - Implemented via Beta distribution sampling from personality-specific ranges
```

---

## 📐 7. Transition Model Parameters

### Beta Distribution Sampling
```yaml
Positive Actions:
  alpha: 3.0    # Bias toward upper end of effect range
  beta: 2.0

Negative Actions:
  alpha: 2.0    # Bias toward lower end of effect range
  beta: 3.0

Mixed Actions:
  alpha: 2.5    # Uniform distribution
  beta: 2.5
```

### Calmness Update
```python
# Acting agent's calmness
new_calmness_acting = clip(
    old_calmness + delta_calmness + recovery_rate,
    0.0, 1.0
)

# Other agent's calmness (cross-agent influence)
new_calmness_other = clip(
    old_calmness + delta_calmness * 0.6 + recovery_rate,
    0.0, 1.0
)
```

### Action Feasibility (Calmness-Based)
```python
# Probability modification based on calmness
P(action_i) ∝ exp(α * calmness - β * difficulty(action_i))

Where:
  α = 1.0 (feasibility_alpha)
  β = 1.0 (feasibility_beta)
  difficulty(action_i) = predefined difficulty score per action
```

---

## 📊 8. Evaluation Parameters

```yaml
num_episodes: 100                  # Episodes per evaluation
render: false                      # No visualization during evaluation
alternate_first_move: true         # Eliminate first-mover advantage

Metrics Recorded:
  - Team rewards (training objective)
  - Individual rewards (contribution analysis)
  - Episode length
  - Final state (emotion, trust, conflict, calmness)
  - Termination rates (SUCCESS/FAILURE/NEUTRAL)
  - Action distribution
  - Cooperation scores
```

---

## 🔢 9. Computational Requirements

### Training Time Estimates
```yaml
Per Episode:
  - Average steps: 25-35
  - Average time: 0.5-1 second (CPU), 0.2-0.4 second (GPU)

Per Scenario (1 run, 8000 episodes):
  - CPU: ~2-3 hours
  - GPU: ~1-1.5 hours

All Scenarios (5 × 15 runs = 75 runs):
  - CPU: ~150-225 hours (6-9 days)
  - GPU: ~75-112 hours (3-5 days)
```

### Memory Requirements
```yaml
Replay Buffer: ~50 MB (20000 transitions × 14 dims × 8 bytes)
Q-Network: ~0.5 MB (18000 parameters × 4 bytes × 2 networks)
Total per Agent: ~100 MB
Total per Scenario: ~200 MB
```

---

## 📈 10. Performance Metrics

### Training Metrics (Logged Every 200 Episodes)
```yaml
- Average Team Reward (Agent A)
- Average Team Reward (Agent B)
- Average Episode Length
- Final State Statistics:
  * Mean Emotion
  * Mean Trust
  * Mean Conflict
- Loss (Q-network training loss)
- Epsilon (exploration rate)
```

### Evaluation Metrics
```yaml
Primary Metrics:
  - Success Rate
  - Failure Rate
  - Neutral/Stalemate Rate
  - Average Episode Length

Reward Metrics:
  - Team Reward (mean ± std)
  - Individual Reward A (mean ± std)
  - Individual Reward B (mean ± std)
  - Cooperation Scores

State Metrics:
  - Final Emotion (mean ± std)
  - Final Trust (mean ± std)
  - Final Conflict (mean ± std)
  - Final Calmness A (mean ± std)
  - Final Calmness B (mean ± std)

Behavioral Metrics:
  - Action Distribution (Agent A)
  - Action Distribution (Agent B)
  - Positive Action Ratio
  - Negative Action Ratio
```

---

## 🎯 11. MARL-Specific Parameters

### Team Reward Design (QMIX/VDN-inspired)
```yaml
Principle: Shared team reward for cooperative learning
Both agents receive: Same team_reward at each step
Training objective: Maximize joint team reward
```

### Individual Reward Design (COMA-inspired)
```yaml
Purpose: Evaluation and analysis only (NOT for training)
Formula: Based on action contribution + alignment with state improvement
Usage: Identify individual agent contributions
```

### Credit Assignment
```yaml
Strategy: Per-step team reward
Advantage: Avoids sparse reward problem
Benefit: Clear credit assignment in turn-based environment
```

---

## 📝 12. Key Design Decisions

### Why These Parameters?

**DQN Learning Rate (3e-4)**:
- Standard for DQN in moderate-complexity tasks
- Higher than typical (1e-4) for faster learning in short episodes

**Discount Factor (0.99)**:
- High value for long-term planning
- Relationship repair requires considering future consequences

**Epsilon Min (0.05)**:
- Higher than typical (0.01) to maintain "emotionally unstable" behavior
- Prevents complete determinism, more human-like

**Memory Size (20000)**:
- Large buffer for diverse experiences
- ~250 full episodes worth of transitions

**Target Update Freq (200)**:
- Slower updates for stability
- Prevents oscillation in Q-values

**Max Episode Steps (50)**:
- Increased from 20 to give agents more time
- Reduces premature stalemates

**Team Reward Range (±5.0)**:
- Larger than typical (±3.0) to accommodate termination rewards (±4.0)
- Clear signal for SUCCESS/FAILURE

---

## 🔗 References

For detailed explanations:
- **Algorithm**: See `agents/deep_rl/dqn.py`
- **Environment**: See `environment/relationship_env.py`
- **Reward Design**: See `MARL_REWARD_DESIGN.md`
- **Training**: See `scripts/train_deep.py`
- **Evaluation**: See `scripts/evaluate_deep.py`

---

**Total Parameter Count**: ~18,000 trainable + ~50 hyperparameters

**Last Updated**: Based on current implementation as of November 29, 2025


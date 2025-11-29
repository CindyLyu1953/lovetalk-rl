# Multi-Agent Reinforcement Learning Reward Design

## Overview

This document describes our Multi-Agent Reinforcement Learning (MARL) reward design for the relationship dynamics simulator, based on principles from leading MARL papers: **QMIX**, **VDN**, and **COMA**.

---

## Design Philosophy

### Core Insight
Relationship repair is a **pure cooperative multi-agent task**. Both agents share the same goal:
- Improve emotion (emotion ↑)
- Increase trust (trust ↑)
- Reduce conflict (conflict ↓)

Therefore, we adopt a **shared team reward** framework where both agents receive the same reward during training.

---

## Reward Types

### 1. Team Reward (Training)

**Purpose**: Used for training both agents simultaneously

**Formula**: Based on shared relationship state changes

```python
team_reward = f(Δemotion, Δtrust, Δconflict, action, termination)
```

**Components**:

#### A. State Change Reward
```python
# Dynamic emotion weight based on current state
emotion_weight = 3.0 if emotion < 0 else 2.5

r_state = emotion_weight * Δemotion + 1.0 * Δtrust - 0.5 * Δconflict
```

#### B. Emotion Crossing Bonus
```python
if prev_emotion < 0 and curr_emotion >= 0:
    bonus = +0.6  # Reward crossing from negative to positive
```

#### C. Emotion Progress Bonus
```python
if prev_emotion < 0 and curr_emotion < 0 and Δemotion > 0:
    # Reward gradual improvement even when still negative
    bonus = min(0.2, abs(prev_emotion) * 0.3 * (Δemotion / 0.3))
```

#### D. Action-Level Reward
```python
Cooperative actions (apologize, empathize, reassure, suggest_solution, ask_for_needs):
    base = +0.20
    if emotion < 0:
        reward = base + 0.25  # Total: +0.45
    else:
        reward = base  # Total: +0.20

Aggressive actions (defensive, blame):
    reward = -0.20

Withdraw actions (withdraw, change_topic):
    if conflict >= 0.6:
        reward = +0.02  # Strategic withdrawal in high conflict
    else:
        reward = -0.05  # Avoidance in moderate conflict
```

#### E. Termination Reward
```python
SUCCESS (emotion > 0.4 AND trust > 0.6):  +4.0
FAILURE (emotion < -0.5 OR trust < 0.1):  -4.0
NEUTRAL/STALEMATE (max_steps = 50):       -0.2
```

#### F. Reward Clipping
```python
final_reward = clip(total_reward, -5.0, 5.0)
```

**Theoretical Foundation**:
- **QMIX**: Monotonic value function factorization enables decentralized execution with centralized training
- **VDN**: Simple additive value decomposition for fully cooperative tasks
- Both agents optimize the same team objective, encouraging natural cooperation

---

### 2. Individual Reward (Evaluation Only)

**Purpose**: Analyze individual agent contributions (NOT used for training)

**Formula**: Based on action contribution and alignment with state improvement

```python
individual_reward = action_contribution + alignment_reward + termination_contribution
```

**Components**:

#### A. Action Contribution Score
```python
Cooperative actions:
    base = +0.15
    if emotion < 0:
        contribution = base + 0.10  # Total: +0.25
    else:
        contribution = base  # Total: +0.15

Aggressive actions:
    base = -0.15
    if emotion < 0:
        contribution = base - 0.10  # Total: -0.25 (extra penalty for being aggressive when already negative)
    else:
        contribution = base  # Total: -0.15

Withdraw actions:
    if conflict >= 0.6:
        contribution = +0.05  # Strategic withdrawal
    else:
        contribution = -0.05  # Avoidance
```

#### B. Alignment with State Improvement
```python
if cooperative_action and positive_state_change:
    alignment = +0.10  # Agent's cooperation helped

elif aggressive_action and negative_state_change:
    alignment = -0.10  # Agent's aggression hurt

elif cooperative_action and negative_state_change:
    alignment = +0.05  # Tried to help but didn't work (still get some credit)

elif aggressive_action and positive_state_change:
    alignment = -0.05  # State improved despite aggression (still penalize)
```

#### C. Termination Contribution
```python
SUCCESS:
    contribution = +0.5  # Both agents contributed

FAILURE:
    if action was aggressive:
        contribution = -0.5  # This agent's aggression contributed to failure
    else:
        contribution = -0.2  # Shared responsibility

NEUTRAL:
    contribution = -0.1  # Small penalty for stalemate
```

#### D. Clipping
```python
individual_reward = clip(total, -2.0, 2.0)  # Smaller range than team reward
```

**Theoretical Foundation**:
- **COMA (Counterfactual Multi-Agent Policy Gradients)**: Distinguishes between global team reward and individual counterfactual contribution
- Individual rewards quantify "what would happen if this agent acted differently?"
- Used for:
  - Identifying which agent contributes more to success/failure
  - Analyzing personality-specific cooperative/antagonistic tendencies
  - Diagnosing training issues and behavioral patterns

---

## Training vs. Evaluation

| Aspect | Training | Evaluation |
|--------|----------|------------|
| **Reward Used** | Team Reward (shared) | Team Reward (for metrics) |
| **Individual Reward** | Not used | Recorded for analysis |
| **Purpose** | Optimize joint objective | Analyze individual contributions |
| **Both Agents Receive** | Same team reward | Team reward + individual metrics |

---

## Implementation Details

### Environment (`environment/relationship_env.py`)

```python
def step(self, action: int):
    # ... state transition ...
    
    # Compute TEAM REWARD (both agents receive this)
    team_reward = self._compute_team_reward(prev_state, curr_state, action, ...)
    
    # Compute INDIVIDUAL REWARDS (for evaluation only)
    individual_reward_acting = self._compute_individual_reward(
        acting_agent_id, prev_state, curr_state, action, ...
    )
    
    # Return team_reward as main reward (used for training)
    # Store individual rewards in info dict (for evaluation)
    info["team_reward"] = team_reward
    info["individual_reward_a"] = individual_reward_a
    info["individual_reward_b"] = individual_reward_b
    
    return obs, team_reward, done, truncated, info
```

### Trainer (`training/trainer.py`)

```python
# Both agents store transitions with the same team_reward
agent_a.store_transition(state, action_a, team_reward, next_state, done)
agent_b.store_transition(state, action_b, team_reward, next_state, done)
```

### Evaluator (`training/evaluator.py`)

```python
# During evaluation, record both team and individual rewards
metrics = {
    "total_team_reward_a": sum(team_rewards_a),
    "total_team_reward_b": sum(team_rewards_b),
    "total_individual_reward_a": sum(individual_rewards_a),
    "total_individual_reward_b": sum(individual_rewards_b),
    ...
}
```

---

## Key Insights from Literature

### 1. QMIX & VDN: Shared Team Reward
- **Problem**: Credit assignment in cooperative multi-agent settings
- **Solution**: Decompose team value function into individual value functions
- **Our Approach**: Since our task is fully cooperative, we use shared team reward directly
- **Benefit**: Simpler than value decomposition, faster convergence

### 2. COMA: Counterfactual Baselines
- **Problem**: How to attribute success/failure to individual agents?
- **Solution**: Compare actual outcome vs. counterfactual (what if agent acted differently?)
- **Our Approach**: Individual reward approximates this by measuring action-specific contribution
- **Benefit**: Provides interpretable analysis without complex counterfactual computation

### 3. Per-Step vs. Episode-End Reward
- **Literature**: Per-step rewards converge faster in step-to-step cooperative tasks
- **Our Approach**: Team reward computed at every step
- **Benefit**: Avoids sparse reward problem, clearer credit assignment

---

## Expected Outcomes

### Training Metrics
- Both agents should learn cooperative behaviors
- Team reward should increase over training
- Success rate should increase, failure rate should decrease

### Evaluation Metrics
With individual rewards, we can answer:
1. **Which agent is more cooperative?**
   - Compare `avg_individual_reward_a` vs `avg_individual_reward_b`
   - Higher individual reward = more cooperative/helpful

2. **Which personality contributes more to relationship repair?**
   - Analyze across different personality combinations (D1-D5)
   - Example: Does "agreeable" contribute more than "neurotic"?

3. **Are there free-riders?**
   - Agent with high team reward but low individual reward
   - Indicates agent benefits from partner's cooperation without contributing

4. **Which agent is more antagonistic?**
   - Agent with negative individual reward
   - Indicates frequent use of aggressive actions

---

## Validation & Sanity Checks

### 1. Team Reward Consistency
```python
# Both agents should receive the same team_reward at each step
assert info["team_reward"] == reward
```

### 2. Individual Reward Bounds
```python
# Individual rewards should be smaller in magnitude than team rewards
assert -2.0 <= individual_reward <= 2.0
assert -5.0 <= team_reward <= 5.0
```

### 3. Cooperation Alignment
```python
# For cooperative actions, individual reward should align with team reward direction
if action in COOPERATIVE_ACTIONS and team_reward > 0:
    assert individual_reward > 0
```

---

## References

1. **QMIX**: Rashid, T., et al. (2018). "QMIX: Monotonic Value Function Factorisation for Decentralised Multi-Agent Reinforcement Learning." ICML 2018.
   - https://arxiv.org/abs/1803.11485

2. **VDN**: Sunehag, P., et al. (2017). "Value-Decomposition Networks For Cooperative Multi-Agent Learning." arXiv:1706.05296.
   - https://arxiv.org/abs/1706.05296

3. **COMA**: Foerster, J., et al. (2018). "Counterfactual Multi-Agent Policy Gradients." AAAI 2018.
   - https://arxiv.org/abs/1705.08926

4. **QPLEX**: Wang, J., et al. (2020). "QPLEX: Duplex Dueling Multi-Agent Q-Learning." ICLR 2021.
   - https://arxiv.org/abs/2006.04742

---

## Summary

- **Training**: Both agents receive **Team Reward** (shared objective)
- **Evaluation**: Record **Individual Rewards** (analyze contributions)
- **Theoretical Foundation**: QMIX/VDN (cooperative learning) + COMA (contribution analysis)
- **Benefits**: Faster convergence, clearer cooperation, interpretable analysis


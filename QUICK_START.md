# Quick Start Guide

A concise guide for running experiments and understanding the codebase.

## Quick Start

### Run All Experiments (Recommended)
```bash
# Clear old results and run all 13 experiments
bash RUN_EXPERIMENTS.sh 5000 --clear-force
```

This will:
1. Train all 13 experiments (5000 episodes each)
2. Evaluate all trained models (100 episodes each)
3. Collect results into comparison tables
4. Generate visualization figures

**Time**: ~6-12 hours (CPU) or ~2-4 hours (GPU)

---

## Experiment Overview

### 13 Core Experiments

**Table A: Shallow RL (6 experiments)**
- **S1**: Q-learning, neutral vs neutral (Baseline)
- **S2**: Q-learning, impulsive vs sensitive (Most intense conflict)
- **S3**: Q-learning, impulsive vs impulsive (Extreme conflict)
- **S4**: Q-learning, neutral vs avoidant (Cold war mode)
- **S5**: Q-learning, sensitive vs sensitive (Both sensitive)
- **S6**: Q-learning, fixed_opponent, impulsive vs sensitive

**Table B: Deep RL (4 experiments)**
- **D1**: DQN, neutral vs neutral (Deep baseline)
- **D2**: DQN, impulsive vs sensitive (Shallow vs Deep comparison)
- **D3**: PPO, impulsive vs sensitive (PPO stable strategy)
- **D4**: PPO, sensitive vs sensitive (Emotionally delicate interaction)

**Table C: Algorithm Comparison (1 additional)**
- **S2_SARSA**: SARSA, impulsive vs sensitive (Q-learning vs SARSA)

---

## Common Commands

### Training

```bash
# Train single experiment (Shallow RL)
python scripts/train_shallow.py \
    --algorithm q_learning \
    --episodes 5000 \
    --personality_a neutral \
    --personality_b neutral \
    --train_mode self_play \
    --save_dir ./experiments/S1/checkpoints

# Train single experiment (Deep RL)
python scripts/train_deep.py \
    --algorithm dqn \
    --episodes 5000 \
    --personality_a neutral \
    --personality_b neutral \
    --train_mode self_play \
    --save_dir ./experiments/D1/checkpoints \
    --history_length 10

# Train all experiments (batch)
python scripts/run_all_experiments.py \
    --episodes 5000 \
    --clear_first --clear_force
```

### Evaluation

```bash
# Evaluate single experiment
python scripts/evaluate.py \
    --agent_type q_learning \
    --checkpoint_a ./experiments/S1/checkpoints/agent_a_ep5000.pth \
    --checkpoint_b ./experiments/S1/checkpoints/agent_b_ep5000.pth \
    --personality_a neutral \
    --personality_b neutral \
    --num_episodes 100

# Evaluate all experiments
python scripts/evaluate_all.py --num_episodes 100
```

### Results & Visualization

```bash
# Collect results
python scripts/collect_results.py

# Generate visualizations
python scripts/visualize_results.py

# Debug environment
python scripts/debug_environment.py
```

### Utilities

```bash
# Clear experiments directory
python scripts/clear_experiments.py --experiment_dir ./experiments --force

# Or with confirmation prompt
python scripts/clear_experiments.py --experiment_dir ./experiments
```

---

## Output Structure

```
experiments/
├── experiment_plan.json          # Overall experiment plan
├── experiment_summary.json       # Training summary
├── comparison_table.csv          # Comparison table (all metrics)
├── all_results.json              # Full results (JSON)
│
├── S1/                           # Experiment S1
│   ├── metadata.json
│   ├── training_log_S1.txt
│   ├── evaluation_S1.txt
│   └── checkpoints/
│       ├── agent_a_ep1000.pth
│       └── agent_b_ep1000.pth
│
├── S2/, S3/, ...                 # Other experiments
│
└── figures/                      # Generated visualizations
    ├── learning_curves.pdf
    ├── termination_rates.pdf
    └── algorithm_comparison.pdf
```

---

## Key Configuration

### Initial State
- **Emotion**: `-0.2` (slightly negative, reflects conflict scenario)
- **Trust**: `0.6` (moderate trust)
- **Calmness**: `0.6` (more calm, prevents immediate termination)

### Termination Conditions
- **SUCCESS**: `emotion > 0.7 AND trust > 0.75`
- **FAILURE**: `emotion < -0.9 OR trust < 0.1`
- **NEUTRAL**: Max steps (20) reached

### Algorithms
- **Shallow RL**: Q-learning, SARSA (tabular methods)
- **Deep RL**: DQN, PPO (neural network methods)

---

## Quick Troubleshooting

### Check Initial State
```bash
python scripts/debug_environment.py
```

### Test Single Episode
```bash
python scripts/train_shallow.py --algorithm q_learning --episodes 1 --personality_a neutral --personality_b neutral
```

### Verify Checkpoints
```bash
ls experiments/S1/checkpoints/
```

---

## Understanding Results

### Key Metrics
- **Episode Length**: Should be > 1 step (typically 3-15 steps)
- **Termination Rates**: Mix of SUCCESS/FAILURE/NEUTRAL
- **Rewards**: Should improve over training
- **Action Distribution**: Should show learning (positive actions increase)

### Expected Outcomes
- Episode length > 1 step
- Learning curves show improvement
- Termination rates have variety
- Agents learn to use positive actions

---

## Documentation Files

- `QUICK_START.md` (this file) - Quick reference guide
- `README.md` - Project documentation

---

## Tips

1. **Start Small**: Test with `--episodes 10` before full runs
2. **Check Logs**: Monitor `training_log_*.txt` files
3. **Use Clear**: Use `--clear-force` for fresh starts
4. **Monitor Progress**: Check `experiment_summary.json` periodically
5. **GPU Speedup**: Deep RL runs 3-4x faster on GPU

---

## For Teammates

**To run experiments:**
```bash
bash RUN_EXPERIMENTS.sh 5000 --clear-force
```

**To check results:**
```bash
# View comparison table
cat experiments/comparison_table.csv

# View visualizations
open experiments/figures/
```

**To understand code structure:**
- `environment/` - Environment definition
- `agents/` - Agent implementations (shallow & deep RL)
- `training/` - Training loop
- `scripts/` - Command-line scripts

**Need help?** Check the detailed documentation files or run `python scripts/debug_environment.py` for diagnostics.


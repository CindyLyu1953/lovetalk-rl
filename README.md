# Multi-Agent RL: Relationship Dynamics Simulator

A reinforcement learning project that simulates relationship communication dynamics between two agents with different personality policies. The project explores how agents learn effective communication strategies through multi-agent reinforcement learning to reduce conflict, improve emotional stability, and increase trust.

## Project Overview

This project implements a turn-based two-agent communication environment where partners (agents) take turns expressing actions based on their learned policies. The environment models relationship state through three core metrics:
- **Emotion Level**: Current emotional valence [-1, 1] (negative to positive)
- **Trust Level**: Relationship trust level [0, 1]
- **Conflict Intensity**: Intensity of current conflict [0, 1]

### Core Research Question

> Can two intelligent agents with different personality characteristics learn effective communication strategies through multi-agent reinforcement learning to reduce conflict, improve emotional stability, and increase trust in simulated relationship conflict scenarios?

## Features

- **Rule-based Environment**: Turn-based two-agent communication environment grounded in psychological theories
- **Multiple RL Algorithms**: 
  - Shallow RL: Q-learning, SARSA (tabular methods)
  - Deep RL: DQN, PPO (neural network-based methods)
- **Personality System**: Different personality types (impulsive, sensitive, avoidant, neutral) affecting agent behavior
- **Psychology-Grounded Actions**: 10 discrete communication actions based on Gottman's Four Horsemen and Nonviolent Communication (NVC) models
- **Data-Driven Calibration**: Optional calibration of transition model using real dialogue datasets (DailyDialog, EmpatheticDialogues)

## Project Structure

```
lovetalk-rl/
├── environment/          # Core environment implementation
│   ├── __init__.py
│   ├── actions.py        # Action space definitions
│   ├── state.py          # State space definitions
│   ├── transition_model.py  # State transition model
│   └── relationship_env.py  # Main environment class
├── agents/               # RL agent implementations
│   ├── shallow_rl/       # Tabular RL methods
│   │   ├── q_learning.py
│   │   └── sarsa.py
│   └── deep_rl/          # Deep RL methods
│       ├── dqn.py
│       └── ppo.py
├── personality/          # Personality policy system
│   ├── __init__.py
│   └── personality_policy.py
├── training/             # Training and evaluation utilities
│   ├── __init__.py
│   ├── trainer.py
│   └── evaluator.py
├── scripts/              # Training and evaluation scripts
│   ├── train_shallow.py
│   ├── train_deep.py
│   └── evaluate.py
├── data/                 # Data loading and calibration utilities
│   ├── __init__.py
│   ├── data_loader.py
│   └── calibrator.py
├── utils/                # Visualization utilities
│   ├── __init__.py
│   └── visualizer.py
├── config/               # Configuration files
│   ├── __init__.py
│   └── config.yaml
├── requirements.txt
└── README.md
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd lovetalk-rl
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training Shallow RL Agents

Train Q-learning or SARSA agents:
```bash
python scripts/train_shallow.py \
    --algorithm q_learning \
    --episodes 5000 \
    --personality_a neutral \
    --personality_b neutral \
    --train_mode self_play \
    --save_dir ./checkpoints/shallow
```

### Training Deep RL Agents

Train DQN or PPO agents:
```bash
python scripts/train_deep.py \
    --algorithm dqn \
    --episodes 5000 \
    --personality_a neutral \
    --personality_b sensitive \
    --train_mode self_play \
    --save_dir ./checkpoints/deep
```

### Evaluating Trained Agents

Evaluate trained agents:
```bash
python scripts/evaluate.py \
    --agent_type q_learning \
    --checkpoint_a ./checkpoints/shallow/agent_a_ep5000.pth \
    --checkpoint_b ./checkpoints/shallow/agent_b_ep5000.pth \
    --num_episodes 100
```

## Action Space

The environment defines 10 discrete communication actions:

**Positive Actions (NVC-based)**:
- `APOLOGIZE`: Taking responsibility and apologizing
- `EMPATHIZE`: Expressing understanding and empathy
- `EXPLAIN`: Calmly explaining facts without blame
- `REASSURE`: Providing emotional comfort and reassurance
- `SUGGEST_SOLUTION`: Proposing constructive solutions
- `ASK_FOR_NEEDS`: Inquiring about partner's needs and feelings

**Neutral Actions**:
- `CHANGE_TOPIC`: Shifting conversation topic

**Negative Actions (Gottman's Four Horsemen)**:
- `DEFENSIVE`: Self-defense and justification
- `BLAME`: Blaming the partner
- `WITHDRAW`: Silent treatment or avoidance

## Personality Types

- **Neutral**: Balanced behavior, no bias
- **Impulsive**: Tendency to use blame/defensive actions
- **Sensitive**: More reactive to negative states, trust more volatile
- **Avoidant**: Tendency to withdraw or change topic

## Reward Function

The reward function combines:
1. **Immediate rewards**: Based on state changes (emotion improvement, trust increase, conflict reduction)
2. **Action quality bonus**: Positive actions get bonuses, negative actions get penalties
3. **Final episode reward**: Weighted combination of final relationship state metrics

## Experiments

The project is designed to explore:
1. **Tabular RL vs Deep RL**: Performance differences in convergence, total reward, conflict resolution success
2. **Personality effects**: How different personality types affect strategy learning
3. **Multi-agent vs single-agent**: Comparison of training modes (self-play vs fixed-opponent)
4. **Reward shaping**: Impact of different reward function designs

## Theoretical Foundations

The environment design is grounded in:
1. **Gottman's Four Horsemen**: Model of relationship conflict (criticism, contempt, defensiveness, stonewalling)
2. **Nonviolent Communication (NVC)**: Framework for empathetic communication
3. **Emotion Regulation & Repair Research**: Models of relationship repair and trust building

## Data Calibration (Optional)

The transition model can be calibrated using real dialogue datasets:
- **DailyDialog**: Provides emotion labels and dialog acts
- **EmpatheticDialogues**: Focused on empathy and emotional understanding

See `data/calibrator.py` for calibration utilities.

## Configuration

Configuration can be customized in `config/config.yaml`:
- Environment parameters
- RL algorithm hyperparameters
- Training settings
- Personality configurations

## Citation

If you use this code, please cite:

```bibtex
@misc{relationship-dynamics-simulator,
  title={Multi-Agent RL: Relationship Dynamics Simulator},
  author={Your Name},
  year={2024},
  howpublished={\url{<repository-url>}}
}
```

## License

[Specify your license]

## Contributing

[Specify contribution guidelines]

## Acknowledgments

- DailyDialog dataset
- EmpatheticDialogues dataset
- Gottman Institute's relationship research
- Nonviolent Communication framework

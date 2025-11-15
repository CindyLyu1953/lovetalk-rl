#!/bin/bash
# Complete script to run all 13 experiments, evaluate, collect results, and visualize
# Usage: bash RUN_EXPERIMENTS.sh [num_episodes] [--clear]

NUM_EPISODES=${1:-5000}
LOG_INTERVAL=100
SAVE_INTERVAL=1000

# Check if --clear flag is provided
CLEAR_FLAG=""
if [[ "$*" == *"--clear"* ]] || [[ "$*" == *"--clear-force"* ]]; then
    CLEAR_FLAG="--clear_first"
    if [[ "$*" == *"--clear-force"* ]]; then
        CLEAR_FLAG="--clear_first --clear_force"
    fi
fi

echo "=========================================="
echo "Relationship Dynamics Simulator - Full Experiment Suite"
echo "=========================================="
echo ""
echo "Running $NUM_EPISODES episodes per experiment"
if [[ -n "$CLEAR_FLAG" ]]; then
    echo "Will clear existing experiments directory first"
fi
echo ""

# Step 1: Run all training experiments
echo "Step 1/4: Training all 13 experiments..."
echo "=========================================="
python scripts/run_all_experiments.py \
    --episodes $NUM_EPISODES \
    --log_interval $LOG_INTERVAL \
    --save_interval $SAVE_INTERVAL \
    --output_dir ./experiments \
    $CLEAR_FLAG

echo ""
echo "Step 2/4: Evaluating all trained models..."
echo "=========================================="
python scripts/evaluate_all.py \
    --num_episodes 100 \
    --output_dir ./experiments

echo ""
echo "Step 3/4: Collecting results..."
echo "=========================================="
python scripts/collect_results.py \
    --experiment_dir ./experiments \
    --output ./experiments/comparison_table.csv

echo ""
echo "Step 4/4: Generating visualizations..."
echo "=========================================="
python scripts/visualize_results.py \
    --results_file ./experiments/all_results.json \
    --output_dir ./experiments/figures

echo ""
echo "=========================================="
echo "All experiments completed!"
echo "=========================================="
echo ""
echo "Results saved in:"
echo "  - Comparison table: ./experiments/comparison_table.csv"
echo "  - Full results: ./experiments/all_results.json"
echo "  - Figures: ./experiments/figures/"
echo ""

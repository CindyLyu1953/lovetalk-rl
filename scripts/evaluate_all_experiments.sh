#!/bin/bash
#
# Batch evaluation script for all D1-D5 experiments
# Evaluates the trained models (run_1) for each experiment
#

echo "=========================================="
echo "Evaluating All Experiments (D1-D5)"
echo "=========================================="
echo ""

# Array of experiments
EXPERIMENTS=("D1" "D2" "D3" "D4" "D5")

# Track success/failure
SUCCESS_COUNT=0
FAILURE_COUNT=0

# Evaluate each experiment
for exp in "${EXPERIMENTS[@]}"; do
    echo ""
    echo "=========================================="
    echo "Evaluating Experiment: $exp"
    echo "=========================================="
    
    # Check if checkpoint exists
    CHECKPOINT_DIR="./experiments/$exp/checkpoints/run_1"
    
    if [ ! -f "$CHECKPOINT_DIR/agent_a_ep4000.pth" ]; then
        echo "❌ ERROR: Agent A checkpoint not found for $exp"
        echo "   Expected: $CHECKPOINT_DIR/agent_a_ep4000.pth"
        FAILURE_COUNT=$((FAILURE_COUNT + 1))
        continue
    fi
    
    if [ ! -f "$CHECKPOINT_DIR/agent_b_ep4000.pth" ]; then
        echo "❌ ERROR: Agent B checkpoint not found for $exp"
        echo "   Expected: $CHECKPOINT_DIR/agent_b_ep4000.pth"
        FAILURE_COUNT=$((FAILURE_COUNT + 1))
        continue
    fi
    
    # Run evaluation
    OMP_NUM_THREADS=1 python scripts/evaluate_single_run.py \
        --checkpoint_dir "$CHECKPOINT_DIR" \
        --experiment "$exp" \
        --num_episodes 100
    
    # Check if evaluation succeeded
    if [ $? -eq 0 ]; then
        echo "✅ $exp evaluation completed successfully"
        echo "   Results saved to: ./experiments/$exp/checkpoints/run_1/evaluation_results.json"
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
    else
        echo "❌ $exp evaluation failed"
        FAILURE_COUNT=$((FAILURE_COUNT + 1))
    fi
    
    echo ""
done

echo ""
echo "=========================================="
echo "Evaluation Summary"
echo "=========================================="
echo "Total experiments: 5"
echo "Successful: $SUCCESS_COUNT"
echo "Failed: $FAILURE_COUNT"
echo ""

if [ $SUCCESS_COUNT -eq 5 ]; then
    echo "🎉 All evaluations completed successfully!"
    echo ""
    echo "Results saved in:"
    for exp in "${EXPERIMENTS[@]}"; do
        echo "  - ./experiments/$exp/checkpoints/run_1/evaluation_results.json"
    done
else
    echo "⚠️ Some evaluations failed. Please check the output above."
fi

echo ""


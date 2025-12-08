#!/bin/bash
# ==============================================================================
# Investigation: Segment Length (5s vs 20s)
# ==============================================================================
# Compares CNN-only and CNN-LSTM performance on 5-second vs 20-second segments.
# All results are automatically logged to checkpoints/ with timestamps.
#
# Usage:
#     chmod +x ./investigations/run_segment_length_investigation.sh
#     ./investigations/run_segment_length_investigation.sh
# ==============================================================================

set -e  # Exit on error

SEEDS="42 123 456"
MODELS="cnn_only cnn_lstm"

echo "======================================================================"
echo "SEGMENT LENGTH INVESTIGATION: 5s vs 20s"
echo "======================================================================"
echo "Seeds: $SEEDS"
echo "Models: $MODELS"
echo "Started: $(date)"
echo ""

# ------------------------------------------------------------------------------
# Step 1: Create 5-second data (if not exists)
# ------------------------------------------------------------------------------
echo "======================================================================"
echo "STEP 1: Preparing 5-second segment data"
echo "======================================================================"

if [ ! -d "data/splits_5s" ]; then
    echo "Creating 5s processed data..."
    python preprocessing.py --segment_sec 5 --save_dir data/processed_5s
    
    echo "Creating 5s splits..."
    python create_splits.py --processed_dir data/processed_5s --split_dir data/splits_5s --seed 42
else
    echo "5s splits already exist, skipping..."
fi

# ------------------------------------------------------------------------------
# Step 2: Create 20-second data (if not exists)
# ------------------------------------------------------------------------------
echo ""
echo "======================================================================"
echo "STEP 2: Preparing 20-second segment data"
echo "======================================================================"

if [ ! -d "data/splits_20s" ]; then
    echo "Creating 20s processed data..."
    python preprocessing.py --segment_sec 20 --save_dir data/processed_20s
    
    echo "Creating 20s splits..."
    python create_splits.py --processed_dir data/processed_20s --split_dir data/splits_20s --seed 42
else
    echo "20s splits already exist, skipping..."
fi

# ------------------------------------------------------------------------------
# Step 3: Train models on 5-second segments
# ------------------------------------------------------------------------------
echo ""
echo "======================================================================"
echo "STEP 3: Training on 5-second segments"
echo "======================================================================"

for seed in $SEEDS; do
    for model in $MODELS; do
        echo ""
        echo "----------------------------------------------------------------------"
        echo "Training $model (seed=$seed) on 5s segments..."
        echo "----------------------------------------------------------------------"
        python train.py --model $model --seed $seed --split_dir data/splits_5s
    done
done

# ------------------------------------------------------------------------------
# Step 4: Train models on 20-second segments
# ------------------------------------------------------------------------------
echo ""
echo "======================================================================"
echo "STEP 4: Training on 20-second segments"
echo "======================================================================"

for seed in $SEEDS; do
    for model in $MODELS; do
        echo ""
        echo "----------------------------------------------------------------------"
        echo "Training $model (seed=$seed) on 20s segments..."
        echo "----------------------------------------------------------------------"
        python train.py --model $model --seed $seed --split_dir data/splits_20s
    done
done

# ------------------------------------------------------------------------------
# Done
# ------------------------------------------------------------------------------
echo ""
echo "======================================================================"
echo "INVESTIGATION COMPLETE"
echo "======================================================================"
echo "Finished: $(date)"
echo ""
echo "Results saved to checkpoints/ with automatic timestamps."
echo ""
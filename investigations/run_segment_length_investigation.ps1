# ==============================================================================
# Investigation: Segment Length (5s vs 20s)
# ==============================================================================
# Compares CNN-only and CNN-LSTM performance on 5-second vs 20-second segments.
# All results are automatically logged to checkpoints/ with timestamps.
#
# Usage:
#     Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
#     .\investigations\run_segment_length_investigation.ps1
# ==============================================================================

$ErrorActionPreference = "Stop"

$SEEDS = @(42, 123, 456)
$MODELS = @("cnn_only", "cnn_lstm")

Write-Host "======================================================================"
Write-Host "SEGMENT LENGTH INVESTIGATION: 5s vs 20s"
Write-Host "======================================================================"
Write-Host "Seeds: $($SEEDS -join ', ')"
Write-Host "Models: $($MODELS -join ', ')"
Write-Host "Started: $(Get-Date)"
Write-Host ""

# ------------------------------------------------------------------------------
# Step 1: Create 5-second data (if not exists)
# ------------------------------------------------------------------------------
Write-Host "======================================================================"
Write-Host "STEP 1: Preparing 5-second segment data"
Write-Host "======================================================================"

if (-not (Test-Path "data/splits_5s")) {
    Write-Host "Creating 5s processed data..."
    python preprocessing.py --segment_sec 5 --save_dir data/processed_5s
    
    Write-Host "Creating 5s splits..."
    python create_splits.py --processed_dir data/processed_5s --split_dir data/splits_5s --seed 42
} else {
    Write-Host "5s splits already exist, skipping..."
}

# ------------------------------------------------------------------------------
# Step 2: Create 20-second data (if not exists)
# ------------------------------------------------------------------------------
Write-Host ""
Write-Host "======================================================================"
Write-Host "STEP 2: Preparing 20-second segment data"
Write-Host "======================================================================"

if (-not (Test-Path "data/splits_20s")) {
    Write-Host "Creating 20s processed data..."
    python preprocessing.py --segment_sec 20 --save_dir data/processed_20s
    
    Write-Host "Creating 20s splits..."
    python create_splits.py --processed_dir data/processed_20s --split_dir data/splits_20s --seed 42
} else {
    Write-Host "20s splits already exist, skipping..."
}

# ------------------------------------------------------------------------------
# Step 3: Train models on 5-second segments
# ------------------------------------------------------------------------------
Write-Host ""
Write-Host "======================================================================"
Write-Host "STEP 3: Training on 5-second segments"
Write-Host "======================================================================"

foreach ($seed in $SEEDS) {
    foreach ($model in $MODELS) {
        Write-Host ""
        Write-Host "----------------------------------------------------------------------"
        Write-Host "Training $model (seed=$seed) on 5s segments..."
        Write-Host "----------------------------------------------------------------------"
        python train.py --model $model --seed $seed --split_dir data/splits_5s
    }
}

# ------------------------------------------------------------------------------
# Step 4: Train models on 20-second segments
# ------------------------------------------------------------------------------
Write-Host ""
Write-Host "======================================================================"
Write-Host "STEP 4: Training on 20-second segments"
Write-Host "======================================================================"

foreach ($seed in $SEEDS) {
    foreach ($model in $MODELS) {
        Write-Host ""
        Write-Host "----------------------------------------------------------------------"
        Write-Host "Training $model (seed=$seed) on 20s segments..."
        Write-Host "----------------------------------------------------------------------"
        python train.py --model $model --seed $seed --split_dir data/splits_20s
    }
}

# ------------------------------------------------------------------------------
# Done
# ------------------------------------------------------------------------------
Write-Host ""
Write-Host "======================================================================"
Write-Host "INVESTIGATION COMPLETE"
Write-Host "======================================================================"
Write-Host "Finished: $(Get-Date)"
Write-Host ""
Write-Host "Results saved to checkpoints/ with automatic timestamps."
Write-Host ""
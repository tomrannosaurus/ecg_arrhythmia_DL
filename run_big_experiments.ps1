#!/usr/bin/env pwsh
# big experiment - random hyperparam search across models & segment lengths
# run prepare_segment_splits.ps1 first!

$ErrorActionPreference = "Stop"

# --- config ---
$MODELS = @(
    "cnn_lstm",
    "cnn_only",
    "cnn_lstm_ln",
    "cnn_lstm_seq16",
    "cnn_bilstm_seq24",
    "cnn_lstm_meanpool",
    "bilstm",
    "simple_lstm",
    "lstm_only",
    "residual",
    "gru",
    "attention"
)

# models that support differential lr (have get_param_groups)
$DIFF_LR_MODELS = @(
    "cnn_lstm",
    "cnn_lstm_ln",
    "cnn_lstm_seq16",
    "cnn_bilstm_seq24",
    "cnn_lstm_meanpool",
    "bilstm",
    "simple_lstm",
    "residual",
    "gru",
    "attention"
)

# models that support freeze_cnn
$FREEZE_MODELS = @(
    "cnn_lstm",
    "cnn_lstm_ln",
    "cnn_lstm_seq16",
    "cnn_bilstm_seq24",
    "cnn_lstm_meanpool",
    "bilstm",
    "simple_lstm",
    "residual",
    "gru",
    "attention"
)

$RUNS_PER_MODEL = 3
$SPLIT_DIRS = @("data/splits", "data/splits_10s", "data/splits_20s")
$BATCH_SIZES = @(32, 64, 128, 192, 256)

# lr ranges (log-uniform)
$LR_MIN = 1e-5
$LR_MAX = 1e-3
$RNN_LR_MIN = 1e-6
$RNN_LR_MAX = 1e-4
$WD_MIN = 1e-6
$WD_MAX = 1e-3

$FREEZE_PROB = 0.3  # prob of freezing cnn

# --- helpers ---

function Get-LogUniform {
    param($min, $max)
    $logMin = [Math]::Log10($min)
    $logMax = [Math]::Log10($max)
    $logValue = $logMin + (Get-Random -Minimum 0.0 -Maximum 1.0) * ($logMax - $logMin)
    $value = [Math]::Pow(10, $logValue)
    return "{0:e2}" -f $value
}

function Get-RandomChoice {
    param($array)
    return $array | Get-Random
}

function Get-RandomBool {
    param($probability)
    return (Get-Random -Minimum 0.0 -Maximum 1.0) -lt $probability
}

function Get-RandomSeed {
    return Get-Random -Minimum 0 -Maximum 100000
}

# --- main loop ---

Write-Host "=== big experiment ==="
Write-Host "models: $($MODELS -join ', ')"
Write-Host "runs per model: $RUNS_PER_MODEL"
Write-Host ""

# check splits exist
foreach ($sd in $SPLIT_DIRS) {
    if (-not (Test-Path $sd)) {
        Write-Host "ERROR: $sd not found. run prepare_segment_splits.ps1 first"
        exit 1
    }
}

# create logs dir
if (-not (Test-Path "logs")) {
    New-Item -ItemType Directory -Path "logs" | Out-Null
}

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$LOG_FILE = "logs/experiment_log_$timestamp.txt"
"RUN,MODEL,SPLIT_DIR,SEED,LR,RNN_LR,WD,BS,FREEZE" | Out-File -FilePath $LOG_FILE -Encoding utf8

$RUN_NUM = 0
$TOTAL = $MODELS.Count * $RUNS_PER_MODEL

foreach ($model in $MODELS) {
    Write-Host "--- $model ---"
    
    # check model capabilities
    $supportsDiffLR = $DIFF_LR_MODELS -contains $model
    $supportsFreeze = $FREEZE_MODELS -contains $model
    
    for ($i = 1; $i -le $RUNS_PER_MODEL; $i++) {
        $RUN_NUM++
        
        # sample random hyperparams
        $SPLIT_DIR = Get-RandomChoice $SPLIT_DIRS
        $SEED = Get-RandomSeed
        $LR = Get-LogUniform $LR_MIN $LR_MAX
        $WD = Get-LogUniform $WD_MIN $WD_MAX
        $BS = Get-RandomChoice $BATCH_SIZES
        
        # only sample rnn_lr if model supports it
        if ($supportsDiffLR) {
            $RNN_LR = Get-LogUniform $RNN_LR_MIN $RNN_LR_MAX
        } else {
            $RNN_LR = $null
        }
        
        # only consider freezing if model supports it
        if ($supportsFreeze) {
            $FREEZE = Get-RandomBool $FREEZE_PROB
        } else {
            $FREEZE = $false
        }
        
        $rnnLrDisplay = if ($RNN_LR) { $RNN_LR } else { "n/a" }
        Write-Host "[$RUN_NUM/$TOTAL] $model | split=$SPLIT_DIR | lr=$LR | rnn_lr=$rnnLrDisplay | wd=$WD | bs=$BS | freeze=$FREEZE"
        
        "$RUN_NUM,$model,$SPLIT_DIR,$SEED,$LR,$rnnLrDisplay,$WD,$BS,$FREEZE" | Out-File -FilePath $LOG_FILE -Append -Encoding utf8
        
        # build command
        $CMD = "python train.py --model $model --seed $SEED --split_dir $SPLIT_DIR --lr $LR --weight_decay $WD --batch_size $BS --num_epochs 100 --patience 10"
        
        if ($RNN_LR) {
            $CMD += " --rnn_lr $RNN_LR"
        }
        
        if ($FREEZE) {
            $CMD += " --freeze_cnn"
        }
        
        try {
            Invoke-Expression $CMD
        } catch {
            Write-Host "  ^ run failed, continuing"
        }
        Write-Host ""
    }
}

Write-Host "=== done ==="
Write-Host "log: $LOG_FILE"
Write-Host "results: checkpoints/"
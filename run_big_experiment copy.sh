#!/bin/bash
# big experiment - random hyperparam search across models & segment lengths
# run prepare_segment_splits.sh first!

set -e

# --- config ---
MODELS=(
    "cnn_lstm"
    "cnn_lstm_ln"
    "cnn_lstm_seq16"
    "cnn_bilstm_seq24"
    "cnn_lstm_meanpool"
    "bilstm"
    "simple_lstm"
    "lstm_only"
    "residual"
    "gru"
    "attention"
)

# models that support differential lr (have get_param_groups)
DIFF_LR_MODELS="cnn_lstm cnn_lstm_ln cnn_lstm_seq16 cnn_bilstm_seq24 cnn_lstm_meanpool bilstm simple_lstm residual gru attention"

# models that support freeze_cnn
FREEZE_MODELS="cnn_lstm cnn_lstm_ln cnn_lstm_seq16 cnn_bilstm_seq24 cnn_lstm_meanpool bilstm simple_lstm residual gru attention"

RUNS_PER_MODEL=3
SPLIT_DIRS=("data/splits" "data/splits_10s" "data/splits_20s")
BATCH_SIZES=(32 64 128 192 256)

# lr ranges (log-uniform)
LR_MIN="1e-5"
LR_MAX="1e-3"
RNN_LR_MIN="1e-6"
RNN_LR_MAX="1e-4"
WD_MIN="1e-6"
WD_MAX="1e-3"

FREEZE_PROB=0.3  # prob of freezing cnn

# --- helpers ---

log_uniform() {
    python3 -c "
import random, math
log_min, log_max = math.log10($1), math.log10($2)
print(f'{10**random.uniform(log_min, log_max):.2e}')"
}

random_choice() {
    local arr=("$@")
    echo "${arr[$((RANDOM % ${#arr[@]}))]}"
}

random_bool() {
    python3 -c "import random; print('true' if random.random() < $1 else 'false')"
}

random_seed() {
    echo $((RANDOM * RANDOM % 100000))
}

supports_diff_lr() {
    [[ " $DIFF_LR_MODELS " =~ " $1 " ]]
}

supports_freeze() {
    [[ " $FREEZE_MODELS " =~ " $1 " ]]
}

# --- main loop ---

echo "=== big experiment ==="
echo "models: ${MODELS[*]}"
echo "runs per model: $RUNS_PER_MODEL"
echo ""

# check splits exist
for sd in "${SPLIT_DIRS[@]}"; do
    if [ ! -d "$sd" ]; then
        echo "ERROR: $sd not found. run prepare_segment_splits.sh first"
        exit 1
    fi
done

# create logs dir
mkdir -p logs

LOG_FILE="logs/experiment_log_$(date +%Y%m%d_%H%M%S).txt"
echo "RUN,MODEL,SPLIT_DIR,SEED,LR,RNN_LR,WD,BS,FREEZE" > "$LOG_FILE"

RUN_NUM=0
TOTAL=$((${#MODELS[@]} * RUNS_PER_MODEL))

for model in "${MODELS[@]}"; do
    echo "--- $model ---"
    
    for ((i=1; i<=RUNS_PER_MODEL; i++)); do
        RUN_NUM=$((RUN_NUM + 1))
        
        # sample random hyperparams
        SPLIT_DIR=$(random_choice "${SPLIT_DIRS[@]}")
        SEED=$(random_seed)
        LR=$(log_uniform $LR_MIN $LR_MAX)
        WD=$(log_uniform $WD_MIN $WD_MAX)
        BS=$(random_choice "${BATCH_SIZES[@]}")
        
        # only sample rnn_lr if model supports it
        if supports_diff_lr "$model"; then
            RNN_LR=$(log_uniform $RNN_LR_MIN $RNN_LR_MAX)
        else
            RNN_LR="n/a"
        fi
        
        # only consider freezing if model supports it
        if supports_freeze "$model"; then
            FREEZE=$(random_bool $FREEZE_PROB)
        else
            FREEZE="false"
        fi
        
        echo "[$RUN_NUM/$TOTAL] $model | split=$SPLIT_DIR | lr=$LR | rnn_lr=$RNN_LR | wd=$WD | bs=$BS | freeze=$FREEZE"
        echo "$RUN_NUM,$model,$SPLIT_DIR,$SEED,$LR,$RNN_LR,$WD,$BS,$FREEZE" >> "$LOG_FILE"
        
        # build command
        CMD="python train.py --model $model --seed $SEED --split_dir $SPLIT_DIR --lr $LR --weight_decay $WD --batch_size $BS --num_epochs 100 --patience 10"
        
        # add rnn_lr if supported
        if supports_diff_lr "$model"; then
            CMD="$CMD --rnn_lr $RNN_LR"
        fi
        
        # add freeze_cnn if supported and selected
        if [ "$FREEZE" = "true" ]; then
            CMD="$CMD --freeze_cnn"
        fi
        
        $CMD || echo "  ^ run failed, continuing"
        echo ""
    done
done

echo "=== done ==="
echo "log: $LOG_FILE"
echo "results: checkpoints/"
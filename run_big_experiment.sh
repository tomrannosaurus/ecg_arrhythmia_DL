#!/bin/bash
# big experiment - random hyperparam search across models & segment lengths
# run prepare_segment_splits.sh first!

set -e

# --- config ---
MODELS=(
    "cnn_lstm"
    "cnn_only"
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

LOG_FILE="experiment_log_$(date +%Y%m%d_%H%M%S).txt"
echo "RUN,MODEL,SPLIT_DIR,SEED,LR,RNN_LR,WD,BS,FREEZE" > "$LOG_FILE"

RUN_NUM=0
TOTAL=$((${#MODELS[@]} * RUNS_PER_MODEL))
count=0

for model in "${MODELS[@]}"; do
    echo "--- $model ---"
    
    for ((i=1; i<=RUNS_PER_MODEL; i++)); do
        RUN_NUM=$((RUN_NUM + 1))
        
        # sample random hyperparams
        SPLIT_DIR=$(random_choice "${SPLIT_DIRS[@]}")
        SEED=$(random_seed)
        LR=$(log_uniform $LR_MIN $LR_MAX)
        RNN_LR=$(log_uniform $RNN_LR_MIN $RNN_LR_MAX)
        WD=$(log_uniform $WD_MIN $WD_MAX)
        BS=$(random_choice "${BATCH_SIZES[@]}")
        FREEZE=$(random_bool $FREEZE_PROB)
        
        echo "[$RUN_NUM/$TOTAL] $model | split=$SPLIT_DIR | lr=$LR | rnn_lr=$RNN_LR | wd=$WD | bs=$BS | freeze=$FREEZE"
        
        echo "$RUN_NUM,$model,$SPLIT_DIR,$SEED,$LR,$RNN_LR,$WD,$BS,$FREEZE" >> "$LOG_FILE"
        
        CMD="python train.py --model $model --seed $SEED --split_dir $SPLIT_DIR \
            --lr $LR --rnn_lr $RNN_LR --weight_decay $WD --batch_size $BS \
            --num_epochs 100 --patience 10"
        [ "$FREEZE" = "true" ] && CMD="$CMD --freeze_cnn"
        
        $CMD || echo "  ^ run failed, continuing"

        count=$((count+1))
        if (( count % 10 == 0 )); then
            echo "Committing after $count runs..."

            BRANCH="main"

            # 1) Stage ALL changes (code + checkpoints)
            git add -A

            # 2) Commit them (if there's anything to commit)
            git commit -m "Auto-commit: Completed $count runs" || echo "No changes to commit"

            # 3) Fetch + rebase onto origin/main
            git fetch origin || echo "git fetch failed (network issue?)"

            if ! git pull --rebase origin "$BRANCH"; then
                echo "git pull --rebase failed (probably merge conflict)."
                echo "Please resolve conflicts manually, then rerun the script."
                exit 1
            fi

            # 4) Push our updated main
            if ! git push origin "$BRANCH"; then
                echo "First git push failed; trying WIP-before-rebase flow..."

                # Make sure everything is staged again (in case anything changed)
                git add -A
                git commit -m "WIP before rebase (auto)" || echo "No WIP changes to commit"

                if ! git pull --rebase origin "$BRANCH"; then
                    echo "git pull --rebase failed again (conflicts?)."
                    echo "Please resolve conflicts manually, then rerun the script."
                    exit 1
                fi

                if ! git push origin "$BRANCH"; then
                    echo "Second git push failed; please fix manually (network/permissions?)."
                    exit 1
                fi
            fi
        fi

        echo ""
    done
done

echo "=== done ==="
echo "log: $LOG_FILE"
echo "results: checkpoints/"

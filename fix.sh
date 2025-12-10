#!/bin/bash

EXPERIMENT_SCRIPT="./run_big_experiment.sh"

echo "=== Supervisor for $EXPERIMENT_SCRIPT ==="
echo "Will restart on failure. Ctrl+C to stop."
echo

while true; do
    echo "[$(date)] Starting experiment..."
    bash "$EXPERIMENT_SCRIPT"
    EXIT_CODE=$?

    if [ $EXIT_CODE -eq 0 ]; then
        echo "[$(date)] Experiment finished normally (exit code 0). Not restarting."
        break
    else
        echo "[$(date)] Experiment crashed/failed with exit code $EXIT_CODE."
        echo "Restarting in 30 seconds..."
        git add -A 
        git commit -m "comitting runs!" --rebase origin main git push origin main
        sleep 30
    fi
done

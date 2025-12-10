#!/bin/bash

# Auto-sync loop for git main branch
# Runs forever until you Ctrl+C

BRANCH="main"

echo "=== Starting auto git sync loop (branch: $BRANCH) ==="
echo "Press Ctrl+C to stop."

while true; do
    echo ""
    echo "----- Sync cycle at $(date) -----"

    # 1) Stage ALL changes
    git add -A

    # 2) Commit (if nothing to commit, ignore)
    git commit -m "WIP before rebase" >/dev/null 2>&1 || echo "Nothing to commit"

    # 3) Fetch + rebase
    if ! git pull --rebase origin "$BRANCH"; then
        echo "ERROR: rebase failed (probably merge conflict)."
        echo "Fix manually, then restart this script."
        exit 1
    fi

    # 4) Push
    if ! git push origin "$BRANCH"; then
        echo "Push failed (network or permissions issue)."
        echo "Will retry on next loop..."
    else
        echo "Push OK."
    fi

    # Sleep between cycles (adjust delay as needed)
    sleep 10
done

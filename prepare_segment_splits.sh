#!/bin/bash
# prep 10s and 20s splits (5s already exists in data/splits)

set -e

echo "=== preparing 10s and 20s segment splits ==="

# 10s
if [ ! -d "data/splits_10s" ]; then
    echo "creating 10s data..."
    python preprocessing.py --segment_sec 10 --save_dir data/processed_10s
    python create_splits.py --processed_dir data/processed_10s --split_dir data/splits_10s --seed 42
else
    echo "10s splits exist, skipping"
fi

# 20s
if [ ! -d "data/splits_20s" ]; then
    echo "creating 20s data..."
    python preprocessing.py --segment_sec 20 --save_dir data/processed_20s
    python create_splits.py --processed_dir data/processed_20s --split_dir data/splits_20s --seed 42
else
    echo "20s splits exist, skipping"
fi

echo ""
echo "done. splits at: data/splits (5s), data/splits_10s, data/splits_20s"

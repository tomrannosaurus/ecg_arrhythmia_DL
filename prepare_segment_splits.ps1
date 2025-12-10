#!/usr/bin/env pwsh
# prep splits

$ErrorActionPreference = "Stop"

Write-Host "=== preparing segment splits ==="

# 5s
if (-not (Test-Path "data/splits")) {
    Write-Host "creating 5s data..."
    python preprocessing.py --segment_sec 5 --save_dir data/processed
    python create_splits.py --processed_dir data/processed --split_dir data/splits --seed 42
} else {
    Write-Host "5s splits exist, skipping"
}

# 10s
if (-not (Test-Path "data/splits_10s")) {
    Write-Host "creating 10s data..."
    python preprocessing.py --segment_sec 10 --save_dir data/processed_10s
    python create_splits.py --processed_dir data/processed_10s --split_dir data/splits_10s --seed 42
} else {
    Write-Host "10s splits exist, skipping"
}

# 20s
if (-not (Test-Path "data/splits_20s")) {
    Write-Host "creating 20s data..."
    python preprocessing.py --segment_sec 20 --save_dir data/processed_20s
    python create_splits.py --processed_dir data/processed_20s --split_dir data/splits_20s --seed 42
} else {
    Write-Host "20s splits exist, skipping"
}

Write-Host ""
Write-Host "done. splits at: data/splits (5s), data/splits_10s, data/splits_20s"

"""
Investigation T10.0: 20-Second Segments
========================================

Tests whether increasing segment length from 5s to 20s improves CNN-LSTM performance.

Hypothesis: Longer segments provide more temporal context for LSTM.
Result: Both models performed WORSE with 20s segments.
Conclusion: Segment length is NOT the issue; focus on LSTM architecture/training.

Usage:
    python investigate_T10.0.py
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime

# Import pipeline functions
from preprocessing import main as preprocess
from create_splits import main as create_splits  
from train import main as train_model


def run_investigation(investigation_id, segment_sec, seeds=[42, 123, 456]):
    """
    Run investigation with specified segment length.
    
    Args:
        investigation_id: Unique ID for this investigation (e.g., "T10.0")
        segment_sec: Segment length in seconds
        seeds: List of random seeds for reproducibility
    
    Returns:
        dict: Results with per-seed and aggregated metrics
    """
    
    print("="*70)
    print(f"INVESTIGATION {investigation_id}: {segment_sec}-Second Segments")
    print("="*70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Seeds: {seeds}\n")
    
    # Paths
    processed_dir = f"data/processed_{segment_sec}s"
    split_dir = f"data/splits_{segment_sec}s"
    
    # Step 1: Preprocess
    print("STEP 1: Preprocessing")
    print("-"*70)
    preprocess(segment_sec=segment_sec, save_dir=processed_dir)
    
    # Step 2: Create splits
    print("\nSTEP 2: Creating Splits")
    print("-"*70)
    create_splits(processed_dir=processed_dir, split_dir=split_dir, seed=42)
    
    # Step 3: Train models
    print("\nSTEP 3: Training Models")
    print("-"*70)
    
    results = {
        'investigation_id': investigation_id,
        'segment_sec': segment_sec,
        'seeds': seeds,
        'timestamp': datetime.now().isoformat(),
        'cnn_only': [],
        'cnn_lstm': []
    }
    
    for seed in seeds:
        print(f"\n{'='*70}")
        print(f"SEED {seed}")
        print('='*70)
        
        # CNN-only
        print("\n[1/2] Training CNN-only...")
        cnn_results = train_model(model_name='cnn_only', seed=seed, split_dir=split_dir, num_epochs=30)
        results['cnn_only'].append(cnn_results)
        print(f"CNN-only: F1={cnn_results['test_f1']:.4f}, AUROC={cnn_results['test_auroc']:.4f}")
        
        # CNN-LSTM
        print("\n[2/2] Training CNN-LSTM...")
        lstm_results = train_model(model_name='cnn_lstm', seed=seed, split_dir=split_dir, num_epochs=50, patience=10)
        results['cnn_lstm'].append(lstm_results)
        print(f"CNN-LSTM: F1={lstm_results['test_f1']:.4f}, AUROC={lstm_results['test_auroc']:.4f}")
    
    # Aggregate results
    results['summary'] = {
        'cnn_only': {
            'f1_mean': float(np.mean([r['test_f1'] for r in results['cnn_only']])),
            'f1_std': float(np.std([r['test_f1'] for r in results['cnn_only']])),
            'auroc_mean': float(np.mean([r['test_auroc'] for r in results['cnn_only']])),
            'auroc_std': float(np.std([r['test_auroc'] for r in results['cnn_only']]))
        },
        'cnn_lstm': {
            'f1_mean': float(np.mean([r['test_f1'] for r in results['cnn_lstm']])),
            'f1_std': float(np.std([r['test_f1'] for r in results['cnn_lstm']])),
            'auroc_mean': float(np.mean([r['test_auroc'] for r in results['cnn_lstm']])),
            'auroc_std': float(np.std([r['test_auroc'] for r in results['cnn_lstm']]))
        }
    }
    
    # Save results
    output_dir = Path('investigations') / investigation_id
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    s = results['summary']
    print(f"CNN-only:  F1={s['cnn_only']['f1_mean']:.4f}±{s['cnn_only']['f1_std']:.4f}, "
          f"AUROC={s['cnn_only']['auroc_mean']:.4f}±{s['cnn_only']['auroc_std']:.4f}")
    print(f"CNN-LSTM:  F1={s['cnn_lstm']['f1_mean']:.4f}±{s['cnn_lstm']['f1_std']:.4f}, "
          f"AUROC={s['cnn_lstm']['auroc_mean']:.4f}±{s['cnn_lstm']['auroc_std']:.4f}")
    print(f"\nResults saved: investigations/{investigation_id}/results.json")
    
    return results


if __name__ == "__main__":
    # T10.0: Test 20-second segments
    run_investigation(
        investigation_id="T10.0",
        segment_sec=20,
        seeds=[42, 123, 456]
    )

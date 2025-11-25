"""
Investigation T10 Baseline: 5-Second Segments
==============================================

Establishes baseline performance for CNN-only and CNN-LSTM on standard 5s segments.
All other T10.x investigations compare against these results.

Usage:
    python investigate_T10.py
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime

from train import main as train_model


def run_baseline(seeds=[42, 123, 456]):
    """
    Run baseline experiments: CNN-only and CNN-LSTM on 5s segments.
    
    Args:
        seeds: List of random seeds for reproducibility
    
    Returns:
        dict: Results with per-seed and aggregated metrics
    """
    
    investigation_id = "T10_baseline"
    split_dir = "data/splits"  # Standard 5s segments
    
    print("="*70)
    print("BASELINE: CNN-only vs CNN-LSTM (5-Second Segments)")
    print("="*70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Seeds: {seeds}")
    print(f"Split directory: {split_dir}\n")
    
    results = {
        'investigation_id': investigation_id,
        'segment_sec': 5,
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
        cnn_results = train_model(
            model_name='cnn_only',
            seed=seed,
            split_dir=split_dir,
            num_epochs=30,
            patience=10
        )
        results['cnn_only'].append(cnn_results)
        print(f"CNN-only: F1={cnn_results['test_f1']:.4f}, AUROC={cnn_results['test_auroc']:.4f}")
        
        # CNN-LSTM
        print("\n[2/2] Training CNN-LSTM...")
        lstm_results = train_model(
            model_name='cnn_lstm',
            seed=seed,
            split_dir=split_dir,
            num_epochs=50,
            patience=10
        )
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
    run_baseline()

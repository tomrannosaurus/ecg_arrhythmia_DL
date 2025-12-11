#!/usr/bin/env python3
"""
Find optimal candidate runs using existing results_table infrastructure.

Usage:
    python optimal_runs.py
    python optimal_runs.py --output optimal_models.txt
"""

import argparse
import numpy as np
from results_table import generate_table

# Target configs: (method, model, batch_size, lr, rnn_lr, weight_decay, seed)
TARGET_CONFIGS = [
    ('Greedy', 'attention', 64, 5.00e-04, 5.00e-04, 1.00e-04, 42),
    ('Linear', 'bilstm',    24, 5.00e-03, 1.00e-03, 1.00e-06, 42),
    ('RF',     'cnn_lstm',  48, 1.10e-03, None,     7.70e-05, 42),
    ('NN',     'bilstm',    24, 1.18e-03, 1.40e-04, 1.24e-06, 42),
]


def approx_eq(a, b, tol=0.05):
    """Check approximate equality (within 5%)."""
    if a is None or b is None:
        return a is None and b is None
    if a == 0 and b == 0:
        return True
    try:
        return abs(a - b) / max(abs(a), abs(b)) < tol
    except:
        return False


def find_matching_run(df, model, batch_size, lr, rnn_lr, weight_decay, seed):
    """Find run matching the specified hyperparameters."""
    mask = (
        (df['Model'] == model) &
        (df['Batch_Size'] == batch_size) &
        (df['Seed'] == seed)
    )
    
    candidates = df[mask].copy()
    if candidates.empty:
        return None
    
    # Filter by LR
    candidates = candidates[candidates['LR'].apply(lambda x: approx_eq(x, lr))]
    if candidates.empty:
        return None
    
    # Filter by RNN_LR if specified
    if rnn_lr is not None:
        candidates = candidates[candidates['RNN_LR'].apply(lambda x: approx_eq(x, rnn_lr))]
        if candidates.empty:
            return None
    
    # Filter by weight decay
    candidates = candidates[candidates['Weight_Decay'].apply(lambda x: approx_eq(x, weight_decay))]
    if candidates.empty:
        return None
    
    # Return best F1 among matches
    return candidates.loc[candidates['Test_F1'].idxmax()]


def main(checkpoint_dir="checkpoints", output=None):
    # Load all runs
    df = generate_table(checkpoint_dir=checkpoint_dir, sort_by='Test_F1', ascending=False)
    if df is None:
        print("No runs found.")
        return
    
    results = []
    
    # Find each optimizer's config
    for method, model, bs, lr, rnn_lr, wd, seed in TARGET_CONFIGS:
        print(f"\nSearching for {method}: {model}...")
        match = find_matching_run(df, model, bs, lr, rnn_lr, wd, seed)
        
        if match is not None:
            row = match.to_dict()
            row['Method'] = method
            results.append(row)
            print(f"  Found: {row['Run_ID']} (F1={row['Test_F1']:.4f})")
        else:
            print("  NOT FOUND - needs training")
    
    # Find empirical best
    print("\nSearching for Empirical Best...")
    best_idx = df['Test_F1'].idxmax()
    best_row = df.loc[best_idx].to_dict()
    best_row['Method'] = 'Empirical'
    
    # Check if empirical best already in results
    is_dup = any(r.get('Run_ID') == best_row['Run_ID'] for r in results)
    if is_dup:
        for r in results:
            if r.get('Run_ID') == best_row['Run_ID']:
                r['Method'] = r['Method'] + '*'
        print(f"  Same as {best_row['Run_ID']}")
    else:
        results.append(best_row)
        print(f"  Found: {best_row['Run_ID']} (F1={best_row['Test_F1']:.4f})")
    
    if not results:
        print("\nNo matching runs found.")
        return
    
    # Build output table
    import pandas as pd
    results_df = pd.DataFrame(results)
    
    # Select and order columns for output
    cols = ['Method', 'Model', 'Batch_Size', 'LR', 'RNN_LR', 'Weight_Decay', 
            'Test_F1', 'Test_AUROC', 'Test_Accuracy', 'Seed']
    cols = [c for c in cols if c in results_df.columns]
    results_df = results_df[cols]
    
    # Format for display
    def fmt_lr(x):
        if x is None:
            return "-"
        try:
            if np.isnan(x):
                return "-"
            return f"{x:.2e}"
        except:
            return "-"
    
    display_df = results_df.copy()
    for col in ['LR', 'RNN_LR', 'Weight_Decay']:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(fmt_lr)
    for col in ['Test_F1', 'Test_AUROC', 'Test_Accuracy']:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(lambda x: f"{x:.4f}")
    
    # Print table
    print("\n" + "=" * 110)
    print("Table: Optimal Model Configurations and Performance Metrics")
    print("=" * 110)
    print(display_df.to_string(index=False))
    print("=" * 110)
    print("\n* Method also achieved Empirical Best F1 score")
    
    # Save if requested
    if output:
        with open(output, 'w') as f:
            f.write("Table: Optimal Model Configurations and Performance Metrics\n")
            f.write("=" * 110 + "\n\n")
            f.write(display_df.to_string(index=False))
            f.write("\n" + "=" * 110 + "\n")
            f.write("\n* Method also achieved Empirical Best F1 score\n")
        print(f"\nSaved to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", default="checkpoints")
    parser.add_argument("--output", "-o", help="Output txt file")
    args = parser.parse_args()
    main(args.checkpoint_dir, args.output)
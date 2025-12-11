#!/usr/bin/env python3
"""
find optimal candidate runs using existing results_table infrastructure.

Usage:
    python print_optimal_runs.py
    python print_optimal_runs.py --output optimal_models.txt
"""

import argparse
import numpy as np
from results_table import generate_table

# target configs: (method, model, batch_size, lr, rnn_lr, weight_decay, seed)
TARGET_CONFIGS = [
    ('Greedy',    'attention', 64, 5.00e-04, 5.00e-04, 1.00e-04, 42),
    ('Linear',    'bilstm',    24, 5.00e-03, 1.00e-03, 1.00e-06, 42),
    ('RF',        'cnn_lstm',  48, 1.10e-03, None,     7.70e-05, 42),
    ('NN',        'bilstm',    24, 1.18e-03, 1.40e-04, 1.24e-06, 42),
    ('Empirical', 'cnn_lstm',  64, 1.00e-04, None, 1.00e-04, 1),
]


def approx_eq(a, b, tol=0.02):
    """check approximate equality (within 2%)."""
    if a is None or b is None:
        return a is None and b is None
    if a == 0 and b == 0:
        return True
    try:
        return abs(a - b) / max(abs(a), abs(b)) < tol
    except:
        return False


def find_matching_runs(df, model, batch_size, lr, rnn_lr, weight_decay, seed):
    """find all runs matching the specified hyperparameters."""
    mask = (
        (df['Model'] == model) &
        (df['Batch_Size'] == batch_size) &
        (df['Seed'] == seed)
    )
    
    candidates = df[mask].copy()
    if candidates.empty:
        return None
    
    # filter by lr
    candidates = candidates[candidates['LR'].apply(lambda x: approx_eq(x, lr))]
    if candidates.empty:
        return None
    
    # filter by rnn_lr if specified
    if rnn_lr is not None:
        candidates = candidates[candidates['RNN_LR'].apply(lambda x: approx_eq(x, rnn_lr))]
        if candidates.empty:
            return None
    
    # filter by weight decay
    candidates = candidates[candidates['Weight_Decay'].apply(lambda x: approx_eq(x, weight_decay))]
    if candidates.empty:
        return None
    
    # return all matches (sorted by f1 descending)
    return candidates.sort_values('Test_F1', ascending=False)


def main(checkpoint_dir="checkpoints", output=None):
    # load all runs
    df = generate_table(checkpoint_dir=checkpoint_dir, sort_by='Test_F1', ascending=False)
    if df is None:
        print("No runs found.")
        return
    
    # filter out mps backend runs (lstm bug)
    if 'Device' in df.columns:
        rnn_models = df['Model'].str.contains('lstm|gru|bilstm', case=False, na=False)
        mps_device = df['Device'] == 'mps'
        affected = rnn_models & mps_device
        n_affected = affected.sum()
        if n_affected > 0:
            print(f"filtering {n_affected} mps backend runs (lstm bug)")
            df = df[~affected].copy()
    
    results = []
    
    # find each optimizer's config (may have multiple matches)
    for method, model, bs, lr, rnn_lr, wd, seed in TARGET_CONFIGS:
        print(f"\nSearching for {method}: {model}...")
        matches = find_matching_runs(df, model, bs, lr, rnn_lr, wd, seed)
        
        if matches is not None and not matches.empty:
            # add individual runs
            for _, match in matches.iterrows():
                row = match.to_dict()
                row['Method'] = method
                results.append(row)
            print(f"  Found {len(matches)} run(s): {', '.join(matches['Run_ID'].tolist())}")
            
            # add average row if multiple runs
            if len(matches) > 1:
                avg_row = {
                    'Method': f"{method} (avg)",
                    'Model': model,
                    'Batch_Size': bs,
                    'LR': lr,
                    'RNN_LR': rnn_lr,
                    'Weight_Decay': wd,
                    'Test_F1': matches['Test_F1'].mean(),
                    'Test_AUROC': matches['Test_AUROC'].mean(),
                    'Test_Accuracy': matches['Test_Accuracy'].mean(),
                    'Seed': seed,
                    'Platform': 'avg',
                }
                results.append(avg_row)
        else:
            print(f"  NOT FOUND - needs training")
    
    if not results:
        print("\nNo matching runs found.")
        return
    
    # build output table
    import pandas as pd
    results_df = pd.DataFrame(results)
    
    # select and order columns for output (including Platform/OS)
    cols = ['Method', 'Model', 'Batch_Size', 'LR', 'RNN_LR', 'Weight_Decay', 
            'Test_F1', 'Test_AUROC', 'Test_Accuracy', 'Platform', 'Seed']
    cols = [c for c in cols if c in results_df.columns]
    results_df = results_df[cols]
    
    # format for display
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
    
    # print table
    print("\n" + "=" * 120)
    print("Table: Optimal Model Configurations and Performance Metrics")
    print("=" * 120)
    print(display_df.to_string(index=False))
    print("=" * 120)
    print("\n(avg) = average across multiple runs of same configuration")
    
    # save if requested
    if output:
        with open(output, 'w') as f:
            f.write("Table: Optimal Model Configurations and Performance Metrics\n")
            f.write("=" * 120 + "\n\n")
            f.write(display_df.to_string(index=False))
            f.write("\n" + "=" * 120 + "\n")
            f.write("\n(avg) = average across multiple runs of same configuration\n")
        print(f"\nSaved to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", default="checkpoints")
    parser.add_argument("--output", "-o", help="Output txt file")
    args = parser.parse_args()
    main(args.checkpoint_dir, args.output)
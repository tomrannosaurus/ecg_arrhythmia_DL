"""
Search & Compare Training Runs
===============================

Search through logged training runs to find optimal hyperparameters.
All runs are automatically logged w/ complete config for easy searching.

Usage:
    # List all runs
    python search_runs.py --list
    
    # Find best F1
    python search_runs.py --best f1
    
    # Find best AUROC
    python search_runs.py --best auroc
    
    # Search by model
    python search_runs.py --model cnn_lstm
    
    # Search by LR range
    python search_runs.py --lr-min 1e-5 --lr-max 1e-3
    
    # Compare specific runs
    python search_runs.py --compare run1 run2 run3
"""

import json
import argparse
from pathlib import Path
from datetime import datetime
import pandas as pd


def load_run(json_path):
    """Load run results from JSON file."""
    with open(json_path, 'r') as f:
        return json.load(f)


def find_all_runs(checkpoint_dir="checkpoints"):
    """Find all *_results.json files in checkpoint dir."""
    checkpoint_dir = Path(checkpoint_dir)
    return list(checkpoint_dir.glob("*_results.json"))


def extract_run_summary(results):
    """Extract key info from results for comparison."""
    config = results['config']
    test = results['test']
    history = results['history']
    
    return {
        # ID
        'run_id': config['run_id'],
        'timestamp': config['timestamp'],
        'datetime': config['datetime'],
        
        # Model
        'model': config['model_name'],
        'model_file': config['model_file'],
        'params': config['model_parameters'],
        
        # Hyperparams
        'lr': config['learning_rate'],
        'wd': config['weight_decay'],
        'bs': config['batch_size'],
        'seed': config['seed'],
        
        # Training
        'epochs': history['epochs_completed'],
        'best_epoch': history['best_epoch'],
        'early_stop': history['stopped_early'],
        
        # Performance
        'val_f1': history['best_val_f1'],
        'test_f1': test['f1'],
        'test_auroc': test['auroc'],
        'test_acc': test['accuracy'],
        'test_loss': test['loss']
    }


def list_all_runs(checkpoint_dir="checkpoints"):
    """List all training runs w/ key metrics."""
    run_files = find_all_runs(checkpoint_dir)
    
    if not run_files:
        print(f"No runs found in {checkpoint_dir}/")
        return
    
    print(f"\nFound {len(run_files)} runs in {checkpoint_dir}/\n")
    
    summaries = []
    for run_file in run_files:
        try:
            results = load_run(run_file)
            summaries.append(extract_run_summary(results))
        except Exception as e:
            print(f"Warning: Could not load {run_file}: {e}")
    
    # Create DataFrame for nice display
    df = pd.DataFrame(summaries)
    
    # Sort by test F1 (best first)
    df = df.sort_values('test_f1', ascending=False)
    
    # Print
    print("="*100)
    print("ALL TRAINING RUNS (sorted by test F1)")
    print("="*100)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 150)
    print(df.to_string(index=False))
    print()
    
    return df


def find_best_run(metric='f1', checkpoint_dir="checkpoints"):
    """Find best run by specified metric."""
    run_files = find_all_runs(checkpoint_dir)
    
    if not run_files:
        print(f"No runs found in {checkpoint_dir}/")
        return
    
    best_run = None
    best_score = -float('inf')
    
    for run_file in run_files:
        try:
            results = load_run(run_file)
            score = results['test'][metric]
            
            if score > best_score:
                best_score = score
                best_run = results
        except Exception as e:
            print(f"Warning: Could not load {run_file}: {e}")
    
    if best_run is None:
        print("No valid runs found")
        return
    
    print("\n" + "="*70)
    print(f"BEST RUN (by test {metric.upper()})")
    print("="*70)
    
    config = best_run['config']
    test = best_run['test']
    
    print(f"\nRun ID: {config['run_id']}")
    print(f"Date: {config['datetime']}")
    print(f"\nModel: {config['model_name']} ({config['model_file']})")
    print(f"Parameters: {config['model_parameters']:,}")
    
    print("\nHyperparameters:")
    print(f"  LR: {config['learning_rate']}")
    print(f"  Weight decay: {config['weight_decay']}")
    print(f"  Batch size: {config['batch_size']}")
    print(f"  Seed: {config['seed']}")
    
    print("\nTraining:")
    print(f"  Epochs: {best_run['history']['epochs_completed']}")
    print(f"  Best epoch: {best_run['history']['best_epoch']}")
    print(f"  Early stop: {best_run['history']['stopped_early']}")
    
    print("\nTest Performance:")
    print(f"  F1:     {test['f1']:.4f}")
    print(f"  AUROC:  {test['auroc']:.4f}")
    print(f"  Acc:    {test['accuracy']:.4f}")
    print(f"  Loss:   {test['loss']:.4f}")
    
    print("\nFiles:")
    print(f"  Model:   {config['model_path']}")
    print(f"  History: {config['log_path']}")
    print(f"  Results: {config['results_path']}")
    
    return best_run


def search_runs(model=None, lr_min=None, lr_max=None, seed=None, 
                checkpoint_dir="checkpoints"):
    """Search runs by criteria."""
    run_files = find_all_runs(checkpoint_dir)
    
    if not run_files:
        print(f"No runs found in {checkpoint_dir}/")
        return
    
    matching = []
    
    for run_file in run_files:
        try:
            results = load_run(run_file)
            config = results['config']
            
            # Apply filters
            if model and config['model_name'] != model:
                continue
            if lr_min and config['learning_rate'] < lr_min:
                continue
            if lr_max and config['learning_rate'] > lr_max:
                continue
            if seed and config['seed'] != seed:
                continue
            
            matching.append(extract_run_summary(results))
            
        except Exception as e:
            print(f"Warning: Could not load {run_file}: {e}")
    
    if not matching:
        print("No runs match the search criteria")
        return
    
    # Create DataFrame
    df = pd.DataFrame(matching)
    df = df.sort_values('test_f1', ascending=False)
    
    print("\n" + "="*100)
    print(f"MATCHING RUNS: {len(matching)}")
    print("="*100)
    if model:
        print(f"Model: {model}")
    if lr_min or lr_max:
        print(f"LR range: {lr_min or 'any'} - {lr_max or 'any'}")
    if seed:
        print(f"Seed: {seed}")
    print()
    
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 150)
    print(df.to_string(index=False))
    print()
    
    return df


def compare_runs(run_ids, checkpoint_dir="checkpoints"):
    """Compare specific runs side-by-side."""
    checkpoint_dir = Path(checkpoint_dir)
    
    runs = []
    for run_id in run_ids:
        # Find matching file
        pattern = f"{run_id}*_results.json"
        matches = list(checkpoint_dir.glob(pattern))
        
        if not matches:
            print(f"Warning: No run found matching '{run_id}'")
            continue
        
        if len(matches) > 1:
            print(f"Warning: Multiple runs match '{run_id}', using first")
        
        try:
            results = load_run(matches[0])
            runs.append(extract_run_summary(results))
        except Exception as e:
            print(f"Warning: Could not load {matches[0]}: {e}")
    
    if not runs:
        print("No valid runs to compare")
        return
    
    # Create comparison DataFrame
    df = pd.DataFrame(runs)
    
    print("\n" + "="*100)
    print(f"RUN COMPARISON ({len(runs)} runs)")
    print("="*100)
    
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 150)
    print(df.to_string(index=False))
    print()
    
    # Highlight best for each metric
    print("\nBest per metric:")
    for metric in ['test_f1', 'test_auroc', 'test_acc']:
        best_idx = df[metric].idxmax()
        print(f"  {metric}: {df.loc[best_idx, 'run_id']} ({df.loc[best_idx, metric]:.4f})")
    
    return df


def main():
    parser = argparse.ArgumentParser(
        description='Search & compare training runs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all runs
  python search_runs.py --list
  
  # Find best by F1
  python search_runs.py --best f1
  
  # Search CNN-LSTM runs
  python search_runs.py --model cnn_lstm
  
  # Search LR range
  python search_runs.py --lr-min 1e-5 --lr-max 1e-3
  
  # Compare specific runs
  python search_runs.py --compare cnn_only_seed42 cnn_lstm_seed42
        """
    )
    
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                       help='Checkpoint directory (default: checkpoints/)')
    parser.add_argument('--list', action='store_true',
                       help='List all runs')
    parser.add_argument('--best', type=str, choices=['f1', 'auroc', 'accuracy', 'loss'],
                       help='Find best run by metric')
    parser.add_argument('--model', type=str,
                       help='Filter by model name')
    parser.add_argument('--lr-min', type=float,
                       help='Min learning rate')
    parser.add_argument('--lr-max', type=float,
                       help='Max learning rate')
    parser.add_argument('--seed', type=int,
                       help='Filter by seed')
    parser.add_argument('--compare', nargs='+',
                       help='Compare specific runs by ID')
    
    args = parser.parse_args()
    
    if args.list:
        list_all_runs(args.checkpoint_dir)
    
    elif args.best:
        find_best_run(args.best, args.checkpoint_dir)
    
    elif args.compare:
        compare_runs(args.compare, args.checkpoint_dir)
    
    elif args.model or args.lr_min or args.lr_max or args.seed:
        search_runs(
            model=args.model,
            lr_min=args.lr_min,
            lr_max=args.lr_max,
            seed=args.seed,
            checkpoint_dir=args.checkpoint_dir
        )
    
    else:
        # Default: list all
        list_all_runs(args.checkpoint_dir)


if __name__ == "__main__":
    main()

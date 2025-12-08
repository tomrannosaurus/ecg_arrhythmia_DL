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
    
    # Search by LR range (searches CNN/main LR)
    python search_runs.py --lr-min 1e-5 --lr-max 1e-3
    
    # Search by LSTM LR range
    python search_runs.py --lstm-lr-min 1e-6 --lstm-lr-max 1e-4
    
    # Compare specific runs
    python search_runs.py --compare run1 run2 run3
"""

import json
import argparse
from pathlib import Path
import pandas as pd


def load_run(json_path):
    """Load run results from JSON file."""
    with open(json_path, 'r') as f:
        return json.load(f)


def find_all_runs(checkpoint_dir="checkpoints"):
    """Find all *_results.json files in checkpoint dir."""
    checkpoint_dir = Path(checkpoint_dir)
    return list(checkpoint_dir.glob("*_results.json"))


def format_lr_dict(lr_dict):
    """Format learning rate dict for display."""
    if not lr_dict:
        return "N/A"
    if len(lr_dict) == 1:
        return f"{list(lr_dict.values())[0]:.2e}"
    # Multiple LRs - show all
    return ", ".join([f"{k}={v:.2e}" for k, v in sorted(lr_dict.items())])


def extract_run_summary(results):
    """Extract key info from results for comparison."""
    config = results['config']
    test = results['test']
    history = results['history']
    
    # Extract LR info - handle both old and new format
    main_lr = config.get('learning_rate', None)
    lstm_lr = config.get('lstm_learning_rate', None)
    lr_groups = config.get('learning_rate_groups', {})
    freeze_cnn = config.get('freeze_cnn', False)
    
    # Create LR display string
    if lr_groups:
        lr_display = format_lr_dict(lr_groups)
    elif lstm_lr is not None:
        lr_display = f"cnn={main_lr:.2e}, lstm={lstm_lr:.2e}"
    else:
        lr_display = f"{main_lr:.2e}" if main_lr else "N/A"
    
    return {
        # ID
        'run_id': config['run_id'],
        'timestamp': config['timestamp'],
        'datetime': config['datetime'],
        
        # Model
        'model': config['model_name'],
        'model_file': config['model_file'],
        'params': config['model_parameters'],
        
        # Hyperparams - NEW: separate main and LSTM LRs
        'lr': main_lr,  # Main/CNN LR
        'lstm_lr': lstm_lr,  # LSTM LR (None if uniform)
        'lr_display': lr_display,  # Formatted string for display
        'diff_lr': config.get('differential_lr', False),  # Flag
        'freeze_cnn': freeze_cnn,  # NEW: CNN frozen flag
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
    
    print(f"Found {len(run_files)} training runs")
    
    all_runs = []
    for run_file in run_files:
        try:
            results = load_run(run_file)
            all_runs.append(extract_run_summary(results))
        except Exception as e:
            print(f"Warning: Could not load {run_file}: {e}")
    
    if not all_runs:
        print("No valid runs found")
        return
    
    # Create DataFrame
    df = pd.DataFrame(all_runs)
    
    # Select columns to display
    display_cols = ['run_id', 'model', 'lr_display', 'freeze_cnn', 'test_f1', 
                    'test_auroc', 'test_acc', 'epochs', 'best_epoch', 'datetime']
    df_display = df[display_cols].sort_values('test_f1', ascending=False)
    
    print("\n" + "="*120)
    print("ALL TRAINING RUNS (sorted by Test F1)")
    print("="*120)
    
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 150)
    pd.set_option('display.max_colwidth', 40)
    print(df_display.to_string(index=False))
    print()
    
    return df


def find_best_run(metric='f1', checkpoint_dir="checkpoints"):
    """Find best run by specified metric."""
    run_files = find_all_runs(checkpoint_dir)
    
    if not run_files:
        print(f"No runs found in {checkpoint_dir}/")
        return
    
    all_runs = []
    for run_file in run_files:
        try:
            results = load_run(run_file)
            all_runs.append(extract_run_summary(results))
        except Exception as e:
            print(f"Warning: Could not load {run_file}: {e}")
    
    if not all_runs:
        print("No valid runs found")
        return
    
    df = pd.DataFrame(all_runs)
    
    # Map metric name to column
    metric_col = f'test_{metric}'
    if metric_col not in df.columns:
        print(f"Unknown metric: {metric}. Available: f1, auroc, accuracy, loss")
        return
    
    # Find best (max for f1/auroc/accuracy, min for loss)
    if metric == 'loss':
        best_idx = df[metric_col].idxmin()
    else:
        best_idx = df[metric_col].idxmax()
    
    best_run = df.loc[best_idx]
    
    print("\n" + "="*70)
    print(f"BEST RUN BY {metric.upper()}")
    print("="*70)
    print(f"\nRun ID: {best_run['run_id']}")
    print(f"Model: {best_run['model']}")
    print(f"Learning Rate(s): {best_run['lr_display']}")
    print(f"CNN Frozen: {best_run['freeze_cnn']}")
    print(f"Weight Decay: {best_run['wd']:.2e}")
    print(f"Batch Size: {best_run['bs']}")
    print(f"Seed: {best_run['seed']}")
    print("\nPerformance:")
    print(f"  Test F1:      {best_run['test_f1']:.4f}")
    print(f"  Test AUROC:   {best_run['test_auroc']:.4f}")
    print(f"  Test Accuracy: {best_run['test_acc']:.4f}")
    print(f"  Test Loss:    {best_run['test_loss']:.4f}")
    print("\nTraining:")
    print(f"  Epochs: {best_run['epochs']}")
    print(f"  Best Epoch: {best_run['best_epoch']}")
    print(f"  Early Stop: {best_run['early_stop']}")
    print(f"  Date: {best_run['datetime']}")
    print()
    
    return best_run


def search_runs(model=None, lr_min=None, lr_max=None, lstm_lr_min=None, lstm_lr_max=None,
                seed=None, freeze_cnn=None, device=None, checkpoint_dir="checkpoints"):
    """Search runs with filters."""
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
            
            # Main LR filter
            main_lr = config.get('learning_rate')
            if lr_min and (main_lr is None or main_lr < lr_min):
                continue
            if lr_max and (main_lr is None or main_lr > lr_max):
                continue
            
            # LSTM LR filter 
            lstm_lr = config.get('lstm_learning_rate')
            if lstm_lr_min and (lstm_lr is None or lstm_lr < lstm_lr_min):
                continue
            if lstm_lr_max and (lstm_lr is None or lstm_lr > lstm_lr_max):
                continue
            
            # Seed filter
            if seed and config['seed'] != seed:
                continue
            
            # CNN frozen filter 
            if freeze_cnn is not None:
                if config.get('freeze_cnn', False) != freeze_cnn:
                    continue
            
            # Device filter
            if device and config.get('device') != device: 
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
    
    # Display columns
    display_cols = ['run_id', 'model', 'lr_display', 'freeze_cnn', 'test_f1', 
                    'test_auroc', 'epochs', 'best_epoch']
    df_display = df[display_cols]
    
    print("\n" + "="*100)
    print(f"MATCHING RUNS: {len(matching)}")
    print("="*100)
    if model:
        print(f"Model: {model}")
    if lr_min or lr_max:
        print(f"Main/CNN LR range: {lr_min or 'any'} - {lr_max or 'any'}")
    if lstm_lr_min or lstm_lr_max:
        print(f"LSTM LR range: {lstm_lr_min or 'any'} - {lstm_lr_max or 'any'}")
    if seed:
        print(f"Seed: {seed}")
    if freeze_cnn is not None:
        print(f"CNN Frozen: {freeze_cnn}")
    print()
    
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 150)
    print(df_display.to_string(index=False))
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
    
    # Display columns for comparison
    display_cols = ['run_id', 'model', 'lr_display', 'freeze_cnn', 'wd', 'bs', 
                    'test_f1', 'test_auroc', 'test_acc', 'test_loss',
                    'epochs', 'best_epoch', 'early_stop']
    df_display = df[display_cols]
    
    print("\n" + "="*100)
    print(f"RUN COMPARISON ({len(runs)} runs)")
    print("="*100)
    
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 150)
    print(df_display.to_string(index=False))
    print()
    
    # Highlight best for each metric
    print("\nBest per metric:")
    for metric in ['test_f1', 'test_auroc', 'test_acc']:
        best_idx = df[metric].idxmax()
        print(f"  {metric}: {df.loc[best_idx, 'run_id']} ({df.loc[best_idx, metric]:.4f})")
    
    # Best (lowest) loss
    best_idx = df['test_loss'].idxmin()
    print(f"  test_loss: {df.loc[best_idx, 'run_id']} ({df.loc[best_idx, 'test_loss']:.4f})")
    print()
    
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
  
  # Search by main/CNN LR range
  python search_runs.py --lr-min 1e-5 --lr-max 1e-3
  
  # Search by LSTM LR range
  python search_runs.py --lstm-lr-min 1e-6 --lstm-lr-max 1e-4
  
  # Search runs with frozen CNN
  python search_runs.py --freeze-cnn
  
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
                       help='Min main/CNN learning rate')
    parser.add_argument('--lr-max', type=float,
                       help='Max main/CNN learning rate')
    parser.add_argument('--lstm-lr-min', type=float,
                       help='Min LSTM learning rate')
    parser.add_argument('--lstm-lr-max', type=float,
                       help='Max LSTM learning rate')
    parser.add_argument('--seed', type=int,
                       help='Filter by seed')
    parser.add_argument('--freeze-cnn', action='store_true',
                       help='Filter for runs with frozen CNN')
    parser.add_argument('--no-freeze-cnn', action='store_true',
                       help='Filter for runs without frozen CNN')
    parser.add_argument('--compare', nargs='+',
                       help='Compare specific runs by ID')
    parser.add_argument('--device', type=str, help='Filter by device (mps, cuda, cpu)')
    
    args = parser.parse_args()
    
    # Handle freeze_cnn filter
    freeze_cnn_filter = None
    if args.freeze_cnn:
        freeze_cnn_filter = True
    elif args.no_freeze_cnn:
        freeze_cnn_filter = False
    
    if args.list:
        list_all_runs(args.checkpoint_dir)
    
    elif args.best:
        find_best_run(args.best, args.checkpoint_dir)
    
    elif args.compare:
        compare_runs(args.compare, args.checkpoint_dir)
    
    elif args.model or args.lr_min or args.lr_max or args.lstm_lr_min or args.lstm_lr_max or args.seed or freeze_cnn_filter is not None:
        search_runs(
            model=args.model,
            lr_min=args.lr_min,
            lr_max=args.lr_max,
            lstm_lr_min=args.lstm_lr_min,
            lstm_lr_max=args.lstm_lr_max,
            seed=args.seed,
            freeze_cnn=freeze_cnn_filter,
            device=args.device,  
            checkpoint_dir=args.checkpoint_dir
        )
    
    else:
        # Default: list all
        list_all_runs(args.checkpoint_dir)


if __name__ == "__main__":
    main()
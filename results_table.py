#!/usr/bin/env python3
"""
Generate Results Table
======================

Create comprehensive table of all trained models sorted by test F1 score.

Usage:
    python results_table.py
    python results_table.py --checkpoint_dir checkpoints
    python results_table.py --output results_table.csv
    python results_table.py --format markdown
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
    """Find all *_results.json files in checkpoint directory."""
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        return []
    return sorted(checkpoint_dir.glob("*_results.json"))


def extract_model_info(results):
    """Extract comprehensive info from results for the table."""
    config = results['config']
    test = results['test']
    history = results['history']
    
    info = {
        # Model identification
        'Model': config['model_name'],
        
        # Hyperparameters
        'LR': config['learning_rate'],
        'Weight_Decay': config['weight_decay'],
        'Batch_Size': config['batch_size'],
        'Seed': config['seed'],
        'Params': config['model_parameters'],
        
        # Training details
        'Epochs': history['epochs_completed'],
        'Best_Epoch': history['best_epoch'],
        'Early_Stop': history['stopped_early'],
        
        # Performance metrics (sorted by importance)
        'Test_F1': test['f1'],
        'Test_AUROC': test['auroc'],
        'Test_Accuracy': test['accuracy'],
        'Test_Loss': test['loss'],
        'Val_F1': history['best_val_f1'],
        
        # Additional info
        'Run_ID': config['run_id'],
        'Date': config['datetime'],
    }
    
    return info


def generate_table(checkpoint_dir="checkpoints", sort_by='Test_F1', ascending=False):
    """
    Generate comprehensive table of all models.
    
    Args:
        checkpoint_dir: Directory containing checkpoint files
        sort_by: Column to sort by (default: Test_F1)
        ascending: Sort order (default: False for best first)
    
    Returns:
        pandas.DataFrame with all model results
    """
    run_files = find_all_runs(checkpoint_dir)
    
    if not run_files:
        print(f"No runs found in {checkpoint_dir}/")
        return None
    
    print(f"Found {len(run_files)} training runs")
    
    ##############################
    # Extract info from all runs
    all_models = []
    for run_file in run_files:
        try:
            results = load_run(run_file)
            model_info = extract_model_info(results)
            all_models.append(model_info)
        except Exception as e:
            print(f"Warning: Could not load {run_file.name}: {e}")
    
    if not all_models:
        print("No valid runs found")
        return None
    ##############################

    df = pd.DataFrame(all_models) # make DataFrame
    
    df = df.sort_values(sort_by, ascending=ascending)
    
    df = df.reset_index(drop=True) # Reset index for cleaner display
    
    return df


def format_dataframe(df):
    """Format numeric columns for better display."""
    df_display = df.copy()
    
    # format floating point columns
    float_cols = ['LR', 'Weight_Decay', 'Test_F1', 'Test_AUROC', 'Test_Accuracy', 'Test_Loss', 'Val_F1']
    for col in float_cols:
        if col in df_display.columns:
            if col in ['LR', 'Weight_Decay']:
                # scientific notation for small values
                df_display[col] = df_display[col].apply(lambda x: f'{x:.2e}')
            else:
                # 4 decimal places for metrics
                df_display[col] = df_display[col].apply(lambda x: f'{x:.4f}')
    
    # format integer columns
    int_cols = ['Batch_Size', 'Seed', 'Params', 'Epochs', 'Best_Epoch']
    for col in int_cols:
        if col in df_display.columns:
            if col == 'Params':
                # add commas to parameter counts
                df_display[col] = df_display[col].apply(lambda x: f'{x:,}')
    
    return df_display


def print_table(df, format='text'):
    """
    Print the table in specified format.
    
    Args:
        df: pandas DataFrame
        format: 'text', 'markdown', or 'latex'
    """
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 200)
    pd.set_option('display.max_colwidth', 30)
    
    print("\n" + "="*150)
    print("MODEL COMPARISON TABLE (Sorted by Test F1 Score)")
    print("="*150)
    print()
    
    # format for display
    df_display = format_dataframe(df) if format == 'text' else df
    
    if format == 'markdown':
        print(df.to_markdown(index=False, floatfmt='.4f'))
    elif format == 'latex':
        print(df.to_latex(index=False, float_format='%.4f'))
    else:
        print(df_display.to_string(index=False))
    
    print()
    print("="*150)
    print(f"Total models: {len(df)}")
    print(f"Best F1: {df['Test_F1'].max():.4f} (Model: {df.loc[df['Test_F1'].idxmax(), 'Model']})")
    print(f"Best AUROC: {df['Test_AUROC'].max():.4f} (Model: {df.loc[df['Test_AUROC'].idxmax(), 'Model']})")
    print(f"Best Accuracy: {df['Test_Accuracy'].max():.4f} (Model: {df.loc[df['Test_Accuracy'].idxmax(), 'Model']})")
    print("="*150)
    print()


def save_table(df, output_path, format='csv'):
    """
    Save the table to a file.
    
    Args:
        df: pandas DataFrame
        output_path: Path to save file
        format: 'csv', 'excel', or 'latex'
    """
    output_path = Path(output_path)
    
    if format == 'excel' or output_path.suffix in ['.xlsx', '.xls']:
        df.to_excel(output_path, index=False)
        print(f"Table saved to {output_path}")
    elif format == 'latex' or output_path.suffix == '.tex':
        df.to_latex(output_path, index=False)
        print(f"Table saved to {output_path}")
    else:  # default to CSV
        df.to_csv(output_path, index=False)
        print(f"Table saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate comprehensive table of all model results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate and display table
  python results_table.py
  
  # Save to CSV
  python results_table.py --output results.csv
  
  # Display as markdown
  python results_table.py --format markdown
  
  # Sort by AUROC instead of F1
  python results_table.py --sort-by Test_AUROC
        """
    )
    
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                       help='Checkpoint directory (default: checkpoints/)')
    parser.add_argument('--output', type=str,
                       help='Output file path (CSV, Excel, or LaTeX)')
    parser.add_argument('--format', type=str, 
                       choices=['text', 'markdown', 'latex', 'csv', 'excel'],
                       default='text',
                       help='Display format (default: text)')
    parser.add_argument('--sort-by', type=str, default='Test_F1',
                       help='Column to sort by (default: Test_F1)')
    parser.add_argument('--ascending', action='store_true',
                       help='Sort in ascending order (default: descending)')
    
    args = parser.parse_args()
    
    df = generate_table(
        checkpoint_dir=args.checkpoint_dir,
        sort_by=args.sort_by,
        ascending=args.ascending
    )
    
    if df is None:
        return
    
    print_table(df, format=args.format)
    
    if args.output:
        save_table(df, args.output, format=args.format)


if __name__ == "__main__":
    main()
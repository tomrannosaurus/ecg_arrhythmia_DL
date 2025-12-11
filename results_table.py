#!/usr/bin/env python3
"""
generate results table
======================

usage:
    python results_table.py
    python results_table.py --checkpoint_dir checkpoints
    python results_table.py --output results_table.csv
"""

import json
import argparse
from pathlib import Path
import pandas as pd


def load_run(json_path):
    """load run results from json file."""
    with open(json_path, 'r') as f:
        return json.load(f)


def find_all_runs(checkpoint_dir="checkpoints"):
    """find all *_results.json files in checkpoint directory."""
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        return []
    return sorted(checkpoint_dir.glob("*_results.json"))


def normalize_split_dir(split_dir: str) -> str:
    """normalize split directory name for display."""
    if split_dir:
        # remove the _5s suffix since 5s is the default
        return split_dir.replace('_5s', '')
    return split_dir


def format_lr_for_table(config, model_name):
    """
    format learning rate(s) for table display.
    
    special handling for lstm_only model:
    - lstm_only has no cnn, so its single lr is the rnn lr
    - for cnn+rnn models, main lr is cnn lr, rnn_learning_rate is rnn lr
    
    backward compatible: checks both 'rnn_learning_rate' and 'lstm_learning_rate'
    
    returns:
        tuple: (cnn_lr, rnn_lr, lr_display_string)
    """
    main_lr = config.get('learning_rate', None)
    # check new key first, fall back to old key for backward compatibility
    rnn_lr_from_config = config.get('rnn_learning_rate', config.get('lstm_learning_rate', None))
    lr_groups = config.get('learning_rate_groups', {})
    
    # special case: lstm_only model has no cnn
    if model_name == 'lstm_only':
        cnn_lr = None
        rnn_lr = main_lr  # the single lr is for the rnn
        lr_display = f"RNN={main_lr:.2e}" if main_lr else "N/A"
    elif lr_groups and len(lr_groups) > 1:
        # multiple lrs from param groups
        cnn_lr = lr_groups.get('cnn', main_lr)
        rnn_lr = lr_groups.get('rnn', lr_groups.get('lstm', rnn_lr_from_config))
        lr_display = ", ".join([f"{k}={v:.2e}" for k, v in sorted(lr_groups.items())])
    elif rnn_lr_from_config is not None:
        # differential lr
        cnn_lr = main_lr
        rnn_lr = rnn_lr_from_config
        lr_display = f"CNN={main_lr:.2e}, RNN={rnn_lr_from_config:.2e}"
    elif main_lr is not None:
        # single lr for all parameters
        cnn_lr = main_lr
        rnn_lr = None
        lr_display = f"{main_lr:.2e}"
    else:
        cnn_lr = None
        rnn_lr = None
        lr_display = "N/A"
    
    return cnn_lr, rnn_lr, lr_display


def extract_model_info(results):
    """extract comprehensive info from results for the table."""
    config = results['config']
    test = results['test']
    history = results['history']
    model_name = config['model_name']
    
    # extract lr info with model-aware handling
    cnn_lr, rnn_lr, lr_display = format_lr_for_table(config, model_name)
    
    # normalize split directory
    split_dir = normalize_split_dir(config.get('split_dir', 'N/A'))
    
    info = {
        # model identification
        'Model': model_name,
        
        # data split info (normalized)
        'Split_Dir': split_dir,
        
        # hyperparameters
        'LR': cnn_lr,          # cnn lr (none for lstm_only)
        'RNN_LR': rnn_lr,      # rnn/lstm/gru lr
        'LR_Display': lr_display,
        'Diff_LR': config.get('differential_lr', False),
        'CNN_Frozen': config.get('freeze_cnn', False),
        'Weight_Decay': config['weight_decay'],
        'Batch_Size': config['batch_size'],
        'Seed': config['seed'],
        'Params': config['model_parameters'],
        
        # training details
        'Epochs': history['epochs_completed'],
        'Best_Epoch': history['best_epoch'],
        'Early_Stop': history['stopped_early'],
        
        # performance metrics
        'Test_F1': test['f1'],
        'Test_AUROC': test['auroc'],
        'Test_Accuracy': test['accuracy'],
        'Test_Loss': test['loss'],
        'Val_F1': history['best_val_f1'],
        
        # metadata
        'Run_ID': config['run_id'],
        'Date': config['datetime'],
        'Device': config.get('device', 'N/A'),
        'User': config.get('user', 'N/A'),
        'Platform': config.get('platform', 'N/A'),
        'Torch_Vers': config.get('torch_version', 'N/A'),
    }
    
    return info


def generate_table(checkpoint_dir="checkpoints", sort_by='Test_F1', ascending=False):
    """generate comprehensive table of all models."""
    run_files = find_all_runs(checkpoint_dir)
    
    if not run_files:
        print(f"no runs found in {checkpoint_dir}/")
        return None
    
    print(f"found {len(run_files)} training runs")
    
    all_models = []
    for run_file in run_files:
        try:
            results = load_run(run_file)
            model_info = extract_model_info(results)
            all_models.append(model_info)
        except Exception as e:
            print(f"warning: could not load {run_file.name}: {e}")
    
    if not all_models:
        print("no valid runs found")
        return None

    df = pd.DataFrame(all_models)
    df = df.sort_values(sort_by, ascending=ascending)
    df = df.reset_index(drop=True)
    
    return df


def format_dataframe(df):
    """format numeric columns for better display."""
    df_display = df.copy()
    
    # format floating point columns
    float_cols = ['LR', 'RNN_LR', 'Weight_Decay', 'Test_F1', 'Test_AUROC', 
                  'Test_Accuracy', 'Test_Loss', 'Val_F1']
    for col in float_cols:
        if col in df_display.columns:
            if col in ['LR', 'RNN_LR', 'Weight_Decay']:
                df_display[col] = df_display[col].apply(
                    lambda x: f'{x:.2e}' if x is not None else 'N/A'
                )
            else:
                df_display[col] = df_display[col].apply(lambda x: f'{x:.4f}')
    
    # format integer columns
    int_cols = ['Batch_Size', 'Seed', 'Params', 'Epochs', 'Best_Epoch']
    for col in int_cols:
        if col in df_display.columns:
            if col == 'Params':
                df_display[col] = df_display[col].apply(lambda x: f'{x:,}')
    
    return df_display


def print_table(df):
    """print the table to console."""
    if df is None:
        return
    
    display_cols = ['Model', 'Split_Dir', 'LR_Display', 'Diff_LR', 'CNN_Frozen', 
                    'Test_F1', 'Test_AUROC', 'Test_Accuracy', 'Test_Loss',
                    'Epochs', 'Best_Epoch', 'Seed', 'Device']
    
    display_cols = [col for col in display_cols if col in df.columns]
    df_display = df[display_cols]
    df_formatted = format_dataframe(df_display)
    
    print("\n" + "="*120)
    print("model results table")
    print("="*120)
    print(f"total models: {len(df)}")
    print("="*120)
    
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 150)
    pd.set_option('display.max_colwidth', 50)
    print(df_formatted.to_string(index=False))
    
    print("="*120)
    print()


def save_table(df, output_path):
    """save table to csv file."""
    output_path = Path(output_path)
    df.to_csv(output_path, index=False)
    print(f"table saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='generate results table')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    parser.add_argument('--output', type=str, help='output csv file path')
    parser.add_argument('--sort-by', type=str, default='Test_F1')
    parser.add_argument('--ascending', action='store_true')
    
    args = parser.parse_args()
    
    df = generate_table(
        checkpoint_dir=args.checkpoint_dir,
        sort_by=args.sort_by,
        ascending=args.ascending
    )
    
    if df is None:
        return
    
    print_table(df)
    
    if args.output:
        save_table(df, args.output)


if __name__ == "__main__":
    main()
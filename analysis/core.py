#!/usr/bin/env python3
"""
core.py - data loading, feature encoding, and shared utilities
==============================================================

this is the ROOT module - it has NO dependencies on other project files.
all other modules import from here.

contents:
- data loading (load_all_runs, filter_mps_runs)
- feature engineering (add_derived_features, decompose_model_features)
- encoding for ML (encode_features, decode_candidate, generate_candidates)
- basic utilities (get_segment_length)
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd


# =============================================================================
# utilities
# =============================================================================

def get_segment_length(split_dir: str) -> int:
    """extract segment length from split directory path."""
    if split_dir is None:
        return 5
    if '20s' in split_dir:
        return 20
    elif '10s' in split_dir:
        return 10
    elif '5s' in split_dir:
        return 5
    return 5


def decompose_model_features(model_name: str) -> Dict:
    """
    break down model name into architectural features.
    
    features:
    - has_cnn: uses cnn feature extractor
    - has_rnn: uses any recurrent layer
    - rnn_type: 'lstm', 'gru', or 'none'
    - is_bidirectional: uses bidirectional rnn
    - has_attention: uses attention mechanism
    - has_residual: uses residual connections
    - has_ln_after_rnn: has layer norm after rnn output
    - pooling: 'last_state', 'mean_pool', 'global_avg', etc.
    """
    name = model_name.lower()
    
    features = {
        'has_cnn': 'cnn' in name or name in ['residual'],
        'has_rnn': any(x in name for x in ['lstm', 'gru', 'bilstm']),
        'rnn_type': 'none',
        'is_bidirectional': 'bi' in name,
        'has_attention': 'attention' in name,
        'has_residual': 'residual' in name,
        'has_ln_after_rnn': '_ln' in name,
        'pooling': 'last_state'
    }
    
    # determine rnn type
    if 'gru' in name:
        features['rnn_type'] = 'gru'
    elif 'lstm' in name or 'bilstm' in name:
        features['rnn_type'] = 'lstm'
    
    # determine pooling
    if 'meanpool' in name:
        features['pooling'] = 'mean_pool'
    elif not features['has_rnn']:
        features['pooling'] = 'global_avg'
    
    return features


# =============================================================================
# data loading
# =============================================================================

def load_all_runs(checkpoint_dir: str = "checkpoints") -> Optional[pd.DataFrame]:
    """
    load all training runs from checkpoint directory.
    
    expects files named *_results.json with structure:
    {
        "config": {...},
        "test": {...},      # NOT test_metrics!
        "history": {...}
    }
    """
    checkpoint_path = Path(checkpoint_dir)
    if not checkpoint_path.exists():
        print(f"error: checkpoint directory '{checkpoint_dir}' not found")
        return None
    
    result_files = list(checkpoint_path.glob("*_results.json"))
    if not result_files:
        print(f"error: no *_results.json files found in '{checkpoint_dir}'")
        return None
    
    print(f"loading {len(result_files)} training runs...")
    
    runs = []
    skipped = 0
    for f in result_files:
        try:
            with open(f) as fp:
                data = json.load(fp)
            
            config = data.get('config', {})
            # CORRECT: 'test' not 'test_metrics'
            test = data.get('test', {})
            
            # model name: config['model_name'] not config['model']
            model = config.get('model_name')
            if model is None:
                model = config.get('model', 'unknown')
            
            # test metrics: test['f1'] not test['test_f1']
            test_f1 = test.get('f1')
            if test_f1 is None:
                test_f1 = test.get('test_f1')
            
            test_auroc = test.get('auroc')
            if test_auroc is None:
                test_auroc = test.get('test_auroc')
            
            test_acc = test.get('accuracy')
            if test_acc is None:
                test_acc = test.get('test_accuracy')
            
            # learning rates: config['learning_rate'] not config['lr']
            lr = config.get('learning_rate')
            if lr is None:
                lr = config.get('lr')
            
            # rnn_learning_rate or lstm_learning_rate
            rnn_lr = config.get('rnn_learning_rate')
            if rnn_lr is None:
                rnn_lr = config.get('lstm_learning_rate')
            if rnn_lr is None:
                rnn_lr = config.get('rnn_lr')
            
            weight_decay = config.get('weight_decay')
            batch_size = config.get('batch_size')
            
            # differential_lr not diff_lr
            diff_lr = config.get('differential_lr', False)
            if diff_lr is None:
                diff_lr = config.get('diff_lr', False)
            
            # freeze_cnn not cnn_frozen
            cnn_frozen = config.get('freeze_cnn', False)
            if cnn_frozen is None:
                cnn_frozen = config.get('cnn_frozen', False)
            
            seed = config.get('seed')
            device = config.get('device', 'unknown')
            split_dir = config.get('split_dir')
            
            run = {
                'run_id': f.stem.replace('_results', ''),
                'model': model if model else 'unknown',
                'segment_length': get_segment_length(split_dir),
                'lr': lr,
                'rnn_lr': rnn_lr,
                'weight_decay': weight_decay,
                'batch_size': batch_size,
                'diff_lr': diff_lr if diff_lr is not None else False,
                'cnn_frozen': cnn_frozen if cnn_frozen is not None else False,
                'seed': seed,
                'device': device if device else 'unknown',
                'test_f1': test_f1,
                'test_auroc': test_auroc,
                'test_acc': test_acc,
            }
            
            # store original rnn_lr before modification
            run['rnn_lr_raw'] = run['rnn_lr']
            
            # if diff_lr is false, rnn uses same lr as main
            if not run['diff_lr'] and run['rnn_lr'] is None:
                run['rnn_lr'] = run['lr']
            
            # skip runs with no test metrics at all
            if run['test_f1'] is None and run['test_auroc'] is None:
                skipped += 1
                continue
            
            runs.append(run)
            
        except Exception as e:
            print(f"  warning: failed to load {f.name}: {e}")
            skipped += 1
    
    if not runs:
        print("error: no valid runs loaded")
        if skipped > 0:
            print(f"  ({skipped} files skipped due to missing data or errors)")
        return None
    
    df = pd.DataFrame(runs)
    print(f"loaded {len(df)} valid runs")
    if skipped > 0:
        print(f"  ({skipped} files skipped)")
    
    return df


def filter_mps_runs(df: pd.DataFrame) -> pd.DataFrame:
    """
    filter out runs affected by mps backend lstm bug.
    
    there's a known pytorch bug where lstm/gru on mps backend
    produces incorrect gradients. these runs should be excluded.
    """
    if 'device' not in df.columns:
        return df
    
    # identify affected runs: mps device + lstm/gru model
    rnn_models = df['model'].str.contains('lstm|gru|bilstm', case=False, na=False)
    mps_device = df['device'] == 'mps'
    affected = rnn_models & mps_device
    
    n_affected = affected.sum()
    if n_affected > 0:
        print(f"filtering {n_affected} mps backend runs (lstm bug)")
    
    return df[~affected].copy()


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    add derived features for analysis.
    
    - log-scale continuous variables
    - model architecture decomposition
    """
    df = df.copy()
    
    # log-scale continuous hps (better for regression)
    for col, log_col in [('lr', 'log_lr'), ('rnn_lr', 'log_rnn_lr'), ('weight_decay', 'log_wd')]:
        if col in df.columns:
            # convert to numeric, coercing errors (None, strings) to NaN
            numeric_col = pd.to_numeric(df[col], errors='coerce')
            with np.errstate(divide='ignore', invalid='ignore'):
                df[log_col] = np.log10(numeric_col)
            df[log_col] = df[log_col].replace([np.inf, -np.inf], np.nan)
    
    # decompose model architectures
    if 'model' in df.columns:
        features_list = df['model'].apply(decompose_model_features)
        features_df = pd.DataFrame(features_list.tolist())
        for col in features_df.columns:
            df[col] = features_df[col]
    
    return df


# =============================================================================
# feature encoding (for ML models)
# =============================================================================

def encode_features(df: pd.DataFrame, 
                    categorical_cols: List[str],
                    continuous_cols: List[str]) -> Tuple[np.ndarray, Dict]:
    """
    encode mixed categorical/continuous features for ML models.
    
    categorical: one-hot encoding
    continuous: standardization (z-score)
    
    returns:
        X: encoded feature matrix
        encoder: dict with encoding info for decoding
    """
    encoded_parts = []
    encoder = {
        'categorical_cols': categorical_cols,
        'continuous_cols': continuous_cols,
        'categorical_mappings': {},
        'continuous_stats': {}
    }
    
    # encode categorical features (one-hot)
    for col in categorical_cols:
        if col not in df.columns:
            continue
        
        unique_vals = sorted([v for v in df[col].dropna().unique() if pd.notna(v)])
        if not unique_vals:
            continue
            
        encoder['categorical_mappings'][col] = list(unique_vals)
        
        for val in unique_vals:
            encoded_parts.append((df[col] == val).astype(float).values.reshape(-1, 1))
    
    # encode continuous features (standardize)
    for col in continuous_cols:
        if col not in df.columns:
            continue
        
        vals = df[col].values.astype(float)
        valid_vals = vals[~np.isnan(vals)]
        
        if len(valid_vals) == 0:
            continue
        
        mean_val = float(np.mean(valid_vals))
        std_val = float(np.std(valid_vals))
        if std_val < 1e-10:
            std_val = 1.0
        
        encoder['continuous_stats'][col] = {'mean': mean_val, 'std': std_val}
        
        standardized = (vals - mean_val) / std_val
        standardized = np.nan_to_num(standardized, nan=0.0)
        encoded_parts.append(standardized.reshape(-1, 1))
    
    if not encoded_parts:
        return np.zeros((len(df), 0)), encoder
    
    X = np.hstack(encoded_parts)
    return X, encoder


def decode_candidate(candidate: np.ndarray, encoder: Dict) -> Dict:
    """decode encoded candidate back to original feature space."""
    config = {}
    idx = 0
    
    for col in encoder['categorical_cols']:
        if col not in encoder['categorical_mappings']:
            continue
        
        vals = encoder['categorical_mappings'][col]
        n_vals = len(vals)
        one_hot = candidate[idx:idx+n_vals]
        best_idx = int(np.argmax(one_hot))
        config[col] = vals[best_idx]
        idx += n_vals
    
    for col in encoder['continuous_cols']:
        if col not in encoder['continuous_stats']:
            continue
        
        stat = encoder['continuous_stats'][col]
        standardized = candidate[idx]
        original = standardized * stat['std'] + stat['mean']
        
        if col.startswith('log_'):
            config[col.replace('log_', '')] = 10 ** original
        else:
            config[col] = original
        idx += 1
    
    return config


def generate_candidates(encoder: Dict, n_candidates: int = 10000,
                        seed: int = 42) -> np.ndarray:
    """generate candidate configurations for optimization."""
    np.random.seed(seed)
    
    n_features = 0
    for col in encoder['categorical_cols']:
        if col in encoder['categorical_mappings']:
            n_features += len(encoder['categorical_mappings'][col])
    for col in encoder['continuous_cols']:
        if col in encoder['continuous_stats']:
            n_features += 1
    
    candidates = np.zeros((n_candidates, n_features))
    
    for i in range(n_candidates):
        idx = 0
        
        for col in encoder['categorical_cols']:
            if col not in encoder['categorical_mappings']:
                continue
            
            vals = encoder['categorical_mappings'][col]
            n_vals = len(vals)
            one_hot = np.zeros(n_vals)
            one_hot[np.random.randint(n_vals)] = 1.0
            candidates[i, idx:idx+n_vals] = one_hot
            idx += n_vals
        
        for col in encoder['continuous_cols']:
            if col not in encoder['continuous_stats']:
                continue
            candidates[i, idx] = np.random.uniform(-2, 2)
            idx += 1
    
    return candidates


def get_feature_names(encoder: Dict) -> List[str]:
    """get list of feature names in encoded order."""
    names = []
    for col in encoder['categorical_cols']:
        if col in encoder['categorical_mappings']:
            for val in encoder['categorical_mappings'][col]:
                names.append(f"{col}={val}")
    for col in encoder['continuous_cols']:
        if col in encoder['continuous_stats']:
            names.append(col)
    return names


# =============================================================================
# summary utilities
# =============================================================================

def summarize_data(df: pd.DataFrame) -> Dict:
    """compute basic summary statistics for the dataset."""
    summary = {
        'n_runs': len(df),
        'n_models': df['model'].nunique() if 'model' in df.columns else 0,
        'n_seeds': df['seed'].nunique() if 'seed' in df.columns else 0,
        'models': df['model'].value_counts().to_dict() if 'model' in df.columns else {},
        'segment_lengths': df['segment_length'].value_counts().to_dict() if 'segment_length' in df.columns else {},
        'devices': df['device'].value_counts().to_dict() if 'device' in df.columns else {},
    }
    
    for metric in ['test_f1', 'test_auroc', 'test_acc']:
        if metric in df.columns:
            summary[f'{metric}_range'] = (df[metric].min(), df[metric].max())
            summary[f'{metric}_mean'] = df[metric].mean()
    
    return summary

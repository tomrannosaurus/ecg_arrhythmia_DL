#!/usr/bin/env python3
"""
optimizer.py - hyperparameter optimization methods
===================================================

depends on: core.py, stats.py

contents:
- greedy optimization (categorical filtering + continuous regression)
- random forest surrogate
- neural network surrogate
- unified interface
"""

import warnings
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

# import from core (data encoding)
from core import encode_features, decode_candidate, generate_candidates, get_feature_names

# import from stats (statistical tests)
from stats import compute_anova_oneway


# =============================================================================
# continuous factor estimation
# =============================================================================

def estimate_optimal_continuous(df: pd.DataFrame, factor: str, 
                                 response: str = 'test_f1') -> Dict:
    """
    estimate optimal value for continuous hp via regression.
    handles edge cases: insufficient data, constant input, etc.
    """
    # get valid x,y pairs (both non-null)
    valid_mask = df[factor].notna() & df[response].notna()
    if valid_mask.sum() == 0:
        return {
            'factor': factor, 'optimal': np.nan, 'correlation': np.nan,
            'p_value': np.nan, 'method': 'insufficient_data', 'n_samples': 0,
            'observed_range': (np.nan, np.nan)
        }
    
    # extract values and convert to float, handling None
    x_raw = df.loc[valid_mask, factor].values
    y_raw = df.loc[valid_mask, response].values
    
    # convert to float arrays, replacing None with nan
    try:
        x = np.array([float(v) if v is not None else np.nan for v in x_raw])
        y = np.array([float(v) if v is not None else np.nan for v in y_raw])
    except (TypeError, ValueError):
        return {
            'factor': factor, 'optimal': np.nan, 'correlation': np.nan,
            'p_value': np.nan, 'method': 'conversion_failed', 'n_samples': 0,
            'observed_range': (np.nan, np.nan)
        }
    
    # filter out any remaining NaN/inf
    finite_mask = np.isfinite(x) & np.isfinite(y)
    x = x[finite_mask]
    y = y[finite_mask]
    
    n_samples = len(x)
    
    if n_samples < 5:
        mean_val = float(np.mean(x)) if n_samples > 0 else np.nan
        return {
            'factor': factor, 'optimal': mean_val, 'correlation': np.nan,
            'p_value': np.nan, 'method': 'insufficient_data', 'n_samples': n_samples,
            'observed_range': (float(np.min(x)), float(np.max(x))) if n_samples > 0 else (np.nan, np.nan)
        }
    
    unique_x = np.unique(x)
    n_unique = len(unique_x)
    x_range = float(np.max(x) - np.min(x))
    
    if n_unique == 1 or x_range < 1e-10:
        return {
            'factor': factor, 'optimal': float(np.mean(x)), 'correlation': np.nan,
            'p_value': np.nan, 'method': 'constant_input', 'n_samples': n_samples,
            'n_unique': n_unique, 'observed_range': (float(np.min(x)), float(np.max(x)))
        }
    
    if n_unique < 3:
        return {
            'factor': factor, 'optimal': float(np.mean(x)), 'correlation': np.nan,
            'p_value': np.nan, 'method': 'insufficient_variation', 'n_samples': n_samples,
            'n_unique': n_unique, 'observed_range': (float(np.min(x)), float(np.max(x)))
        }
    
    unique_y = np.unique(y)
    if len(unique_y) == 1:
        return {
            'factor': factor, 'optimal': float(np.mean(x)), 'correlation': np.nan,
            'p_value': np.nan, 'method': 'constant_response', 'n_samples': n_samples,
            'n_unique': n_unique, 'observed_range': (float(np.min(x)), float(np.max(x)))
        }
    
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        try:
            r, p_value = scipy_stats.pearsonr(x, y)
            if np.isnan(r):
                raise ValueError("nan correlation")
        except Exception:
            return {
                'factor': factor, 'optimal': float(np.mean(x)), 'correlation': np.nan,
                'p_value': np.nan, 'method': 'correlation_failed', 'n_samples': n_samples,
                'n_unique': n_unique, 'observed_range': (float(np.min(x)), float(np.max(x)))
            }
        
        try:
            coeffs = np.polyfit(x, y, 2)
            a, b, c = coeffs
            if np.any(np.isnan(coeffs)) or np.any(np.isinf(coeffs)):
                raise ValueError("invalid coeffs")
            
            if a < -1e-10:
                optimal_x = float(np.clip(-b / (2 * a), np.min(x), np.max(x)))
                method = 'quadratic_max'
            else:
                optimal_x = float(np.max(x)) if r > 0 else float(np.min(x))
                method = 'linear_trend'
        except Exception:
            optimal_x = float(np.max(x)) if r > 0 else float(np.min(x))
            method = 'linear_trend'
    
    return {
        'factor': factor, 'optimal': optimal_x, 'correlation': float(r),
        'p_value': float(p_value), 'method': method, 'n_samples': n_samples,
        'n_unique': n_unique, 'observed_range': (float(np.min(x)), float(np.max(x)))
    }


# =============================================================================
# greedy optimization
# =============================================================================

def optimize_greedy(df: pd.DataFrame, response: str = 'test_f1',
                    categorical_factors: List[str] = None,
                    continuous_factors: List[str] = None,
                    min_samples_per_level: int = 5,
                    min_samples_after_filter: int = 10,
                    significance_threshold: float = 0.10,
                    verbose: bool = True) -> Dict:
    """
    greedy optimization: filter by categorical, then regress continuous.
    
    phase 1: iteratively filter by best level of each categorical factor
    phase 2: estimate optimal continuous values via regression
    """
    if categorical_factors is None:
        categorical_factors = ['segment_length', 'model', 'batch_size', 'diff_lr', 'cnn_frozen']
    if continuous_factors is None:
        continuous_factors = ['log_lr', 'log_rnn_lr', 'log_wd']
    
    # filter to rows with valid response values
    df_valid = df[df[response].notna()].copy()
    if len(df_valid) == 0:
        if verbose:
            print("\n[greedy] no valid data (all response values are NaN)")
        return {
            'method': 'greedy',
            'path': [],
            'continuous_results': {},
            'final_config': {},
            'predicted_f1': np.nan,
            'std_f1': np.nan,
            'n_samples': 0,
            'filtered_df': df_valid
        }
    
    categorical_factors = [f for f in categorical_factors if f in df_valid.columns and df_valid[f].nunique() > 1]
    continuous_factors = [f for f in continuous_factors if f in df_valid.columns and df_valid[f].notna().sum() > 0]
    
    df_current = df_valid.copy()
    optimization_path = []
    remaining = categorical_factors.copy()
    
    if verbose:
        print("\n[greedy] phase 1: categorical factors")
        print("-" * 60)
    
    while remaining and len(df_current) >= min_samples_after_filter:
        effects = []
        for factor in remaining:
            if df_current[factor].nunique() <= 1:
                continue
            
            result = compute_anova_oneway(df_current, factor, response)
            if np.isnan(result['p_value']) or result['p_value'] > significance_threshold:
                continue
            
            level_counts = df_current.groupby(factor)[response].agg(['mean', 'count'])
            best_level = level_counts['mean'].idxmax()
            best_count = int(level_counts.loc[best_level, 'count'])
            
            if best_count < min_samples_per_level or best_count < min_samples_after_filter:
                continue
            
            effects.append({
                'factor': factor, 'eta_sq': result['eta_sq'], 'p_value': result['p_value'],
                'best_level': best_level, 'best_count': best_count
            })
        
        if not effects:
            break
        
        effects_df = pd.DataFrame(effects).sort_values('eta_sq', ascending=False)
        best = effects_df.iloc[0]
        
        level_stats = df_current.groupby(best['factor'])[response].agg(['mean', 'std', 'count'])
        best_mean = level_stats.loc[best['best_level'], 'mean']
        best_std = level_stats.loc[best['best_level'], 'std']
        if pd.isna(best_std):
            best_std = 0.0
        
        step = {
            'step': len(optimization_path) + 1,
            'factor': best['factor'],
            'type': 'categorical',
            'eta_sq': best['eta_sq'],
            'p_value': best['p_value'],
            'best_level': best['best_level'],
            'mean_f1': best_mean,
            'std_f1': best_std,
            'n_runs': int(best['best_count']),
            'n_before': len(df_current)
        }
        optimization_path.append(step)
        
        if verbose:
            print(f"  {step['factor']}: {step['best_level']} (eta²={step['eta_sq']:.4f}, n={step['n_runs']})")
        
        df_current = df_current[df_current[best['factor']] == best['best_level']].copy()
        remaining.remove(best['factor'])
    
    if verbose:
        print(f"\n[greedy] phase 2: continuous factors (n={len(df_current)})")
        print("-" * 60)
    
    continuous_results = {}
    valid_methods = ['quadratic_max', 'linear_trend']
    
    for factor in continuous_factors:
        if factor not in df_current.columns:
            continue
        
        # check if there's valid data for this factor AND response
        factor_valid = df_current[factor].notna()
        response_valid = df_current[response].notna()
        both_valid = factor_valid & response_valid
        
        if both_valid.sum() < 5:
            if verbose:
                orig = factor.replace('log_', '')
                print(f"  {orig}: insufficient_data (n={both_valid.sum()})")
            continuous_results[factor] = {
                'factor': factor, 'optimal': np.nan, 'correlation': np.nan,
                'p_value': np.nan, 'method': 'insufficient_data', 
                'n_samples': int(both_valid.sum()), 'observed_range': (np.nan, np.nan)
            }
            continue
        
        result = estimate_optimal_continuous(df_current, factor, response)
        continuous_results[factor] = result
        
        if verbose:
            orig = factor.replace('log_', '')
            if result['method'] in valid_methods:
                opt = 10 ** result['optimal'] if factor.startswith('log_') else result['optimal']
                print(f"  {orig}: {opt:.2e} (r={result['correlation']:.3f}, {result['method']})")
            else:
                print(f"  {orig}: {result['method']} (n={result['n_samples']})")
    
    # build config
    final_config = {s['factor']: s['best_level'] for s in optimization_path}
    for factor, result in continuous_results.items():
        if result['method'] in valid_methods:
            orig = factor.replace('log_', '')
            val = 10 ** result['optimal'] if factor.startswith('log_') else result['optimal']
            final_config[orig] = val
    
    final_mean = df_current[response].mean() if len(df_current) > 0 else np.nan
    final_std = df_current[response].std() if len(df_current) > 0 else 0.0
    if pd.isna(final_std):
        final_std = 0.0
    
    return {
        'method': 'greedy',
        'path': optimization_path,
        'continuous_results': continuous_results,
        'final_config': final_config,
        'predicted_f1': final_mean,
        'std_f1': final_std,
        'n_samples': len(df_current),
        'filtered_df': df_current
    }


# =============================================================================
# random forest surrogate
# =============================================================================

def optimize_rf(df: pd.DataFrame, response: str = 'test_f1',
                categorical_cols: List[str] = None,
                continuous_cols: List[str] = None,
                n_candidates: int = 10000,
                verbose: bool = True) -> Dict:
    """random forest surrogate model optimization."""
    try:
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.model_selection import cross_val_score
    except ImportError:
        if verbose:
            print("sklearn not available, falling back to greedy")
        return optimize_greedy(df, response, categorical_cols, continuous_cols, verbose=verbose)
    
    if categorical_cols is None:
        categorical_cols = ['model', 'segment_length', 'batch_size', 'diff_lr', 'cnn_frozen']
    if continuous_cols is None:
        continuous_cols = ['log_lr', 'log_rnn_lr', 'log_wd']
    
    categorical_cols = [c for c in categorical_cols if c in df.columns]
    continuous_cols = [c for c in continuous_cols if c in df.columns]
    
    # filter rows with NaN response
    df_clean = df[df[response].notna()].copy()
    if len(df_clean) < 10:
        if verbose:
            print(f"  insufficient data after filtering NaN (n={len(df_clean)})")
        return optimize_greedy(df, response, categorical_cols, continuous_cols, verbose=verbose)
    
    X, encoder = encode_features(df_clean, categorical_cols, continuous_cols)
    y = df_clean[response].values
    
    if verbose:
        print(f"\n[rf] training random forest (n={len(df_clean)}, features={X.shape[1]})")
        print("-" * 60)
    
    # check for NaN in X or y
    if np.any(np.isnan(X)) or np.any(np.isnan(y)):
        if verbose:
            print("  warning: NaN values detected, imputing...")
        X = np.nan_to_num(X, nan=0.0)
        y = np.nan_to_num(y, nan=np.nanmean(y) if np.any(~np.isnan(y)) else 0.5)
    
    model = RandomForestRegressor(
        n_estimators=100, max_depth=10, min_samples_leaf=3,
        random_state=42, n_jobs=-1
    )
    model.fit(X, y)
    
    # cross-validation with error handling
    try:
        cv_scores = cross_val_score(model, X, y, cv=min(5, len(df_clean)), scoring='r2')
        cv_mean, cv_std = float(np.nanmean(cv_scores)), float(np.nanstd(cv_scores))
    except Exception as e:
        if verbose:
            print(f"  cv failed: {e}")
        cv_mean, cv_std = 0.0, 0.0
    
    if verbose:
        print(f"  cv R²: {cv_mean:.4f} (+/- {cv_std:.4f})")
    
    # feature importance
    importance = model.feature_importances_
    feature_names = get_feature_names(encoder)
    importance_df = pd.DataFrame({
        'feature': feature_names[:len(importance)],
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    if verbose:
        print("\n  top features:")
        for _, row in importance_df.head(5).iterrows():
            print(f"    {row['feature']}: {row['importance']:.4f}")
    
    # find optimal
    if verbose:
        print(f"\n  searching {n_candidates:,} candidates...")
    
    candidates = generate_candidates(encoder, n_candidates)
    predictions = model.predict(candidates)
    
    best_idx = np.argmax(predictions)
    best_config = decode_candidate(candidates[best_idx], encoder)
    best_pred = float(predictions[best_idx])
    
    top_indices = np.argsort(predictions)[-10:][::-1]
    top_configs = []
    for idx in top_indices:
        cfg = decode_candidate(candidates[idx], encoder)
        cfg['predicted_f1'] = float(predictions[idx])
        top_configs.append(cfg)
    
    if verbose:
        print(f"  best predicted: {best_pred:.4f}")
    
    return {
        'method': 'rf',
        'model': model,
        'encoder': encoder,
        'cv_r2': cv_mean,
        'cv_r2_std': cv_std,
        'feature_importance': importance_df,
        'final_config': best_config,
        'predicted_f1': best_pred,
        'std_f1': 0.0,  # RF doesn't provide std, set to 0
        'top_configs': top_configs,
        'n_samples': len(df_clean)
    }


# =============================================================================
# neural network surrogate
# =============================================================================

def optimize_nn(df: pd.DataFrame, response: str = 'test_f1',
                categorical_cols: List[str] = None,
                continuous_cols: List[str] = None,
                n_candidates: int = 10000,
                hidden_sizes: List[int] = None,
                epochs: int = 200,
                verbose: bool = True) -> Dict:
    """neural network surrogate model optimization."""
    if hidden_sizes is None:
        hidden_sizes = [64, 32]
    
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
    except ImportError:
        if verbose:
            print("pytorch not available, falling back to rf")
        return optimize_rf(df, response, categorical_cols, continuous_cols, n_candidates, verbose)
    
    if categorical_cols is None:
        categorical_cols = ['model', 'segment_length', 'batch_size', 'diff_lr', 'cnn_frozen']
    if continuous_cols is None:
        continuous_cols = ['log_lr', 'log_rnn_lr', 'log_wd']
    
    categorical_cols = [c for c in categorical_cols if c in df.columns]
    continuous_cols = [c for c in continuous_cols if c in df.columns]
    
    # filter rows with NaN response
    df_clean = df[df[response].notna()].copy()
    if len(df_clean) < 10:
        if verbose:
            print(f"  insufficient data after filtering NaN (n={len(df_clean)})")
        return optimize_greedy(df, response, categorical_cols, continuous_cols, verbose=verbose)
    
    X, encoder = encode_features(df_clean, categorical_cols, continuous_cols)
    y = df_clean[response].values
    
    # handle any remaining NaN
    X = np.nan_to_num(X, nan=0.0)
    y = np.nan_to_num(y, nan=np.nanmean(y) if np.any(~np.isnan(y)) else 0.5)
    
    if verbose:
        print(f"\n[nn] training neural network (n={len(df_clean)}, features={X.shape[1]})")
        print("-" * 60)
        if len(df_clean) < 200:
            print("  ⚠ WARNING: small dataset, high overfitting risk")
    
    torch.manual_seed(42)
    np.random.seed(42)
    
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.FloatTensor(y.reshape(-1, 1))
    
    n_train = int(0.8 * len(X))
    indices = np.random.permutation(len(X))
    train_idx, val_idx = indices[:n_train], indices[n_train:]
    
    X_train, X_val = X_tensor[train_idx], X_tensor[val_idx]
    y_train, y_val = y_tensor[train_idx], y_tensor[val_idx]
    
    layers = []
    in_size = X.shape[1]
    for h in hidden_sizes:
        layers.extend([nn.Linear(in_size, h), nn.ReLU(), nn.Dropout(0.2)])
        in_size = h
    layers.append(nn.Linear(in_size, 1))
    model = nn.Sequential(*layers)
    
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    best_state = None
    patience, patience_ctr = 20, 0
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        loss = criterion(model(X_train), y_train)
        loss.backward()
        optimizer.step()
        
        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_val), y_val).item()
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                break
    
    if best_state:
        model.load_state_dict(best_state)
    
    model.eval()
    with torch.no_grad():
        val_pred = model(X_val).numpy().flatten()
        y_val_np = y_val.numpy().flatten()
        ss_res = np.sum((y_val_np - val_pred) ** 2)
        ss_tot = np.sum((y_val_np - y_val_np.mean()) ** 2)
        val_r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    
    if verbose:
        print(f"  validation R²: {val_r2:.4f}")
        print(f"  epochs: {epoch + 1}")
    
    if verbose:
        print(f"\n  searching {n_candidates:,} candidates...")
    
    candidates = generate_candidates(encoder, n_candidates)
    
    model.eval()
    with torch.no_grad():
        predictions = model(torch.FloatTensor(candidates)).numpy().flatten()
    
    best_idx = np.argmax(predictions)
    best_config = decode_candidate(candidates[best_idx], encoder)
    best_pred = float(predictions[best_idx])
    
    top_indices = np.argsort(predictions)[-10:][::-1]
    top_configs = []
    for idx in top_indices:
        cfg = decode_candidate(candidates[idx], encoder)
        cfg['predicted_f1'] = float(predictions[idx])
        top_configs.append(cfg)
    
    if verbose:
        print(f"  best predicted: {best_pred:.4f}")
    
    return {
        'method': 'nn',
        'model': model,
        'encoder': encoder,
        'val_r2': float(val_r2),
        'final_config': best_config,
        'predicted_f1': best_pred,
        'std_f1': 0.0,  # NN doesn't provide std, set to 0
        'top_configs': top_configs,
        'n_samples': len(df_clean)
    }


# =============================================================================
# unified interface
# =============================================================================

def optimize(df: pd.DataFrame, method: str = 'greedy',
             response: str = 'test_f1',
             categorical_factors: List[str] = None,
             continuous_factors: List[str] = None,
             verbose: bool = True,
             **kwargs) -> Dict:
    """
    unified optimization interface.
    
    methods: 'greedy', 'rf', 'nn', 'all'
    """
    if categorical_factors is None:
        categorical_factors = ['segment_length', 'model', 'batch_size', 'diff_lr', 'cnn_frozen']
    if continuous_factors is None:
        continuous_factors = ['log_lr', 'log_rnn_lr', 'log_wd']
    
    # separate kwargs by method
    greedy_keys = {'min_samples_per_level', 'min_samples_after_filter', 'significance_threshold'}
    surrogate_keys = {'n_candidates', 'hidden_sizes', 'epochs'}
    
    greedy_kwargs = {k: v for k, v in kwargs.items() if k in greedy_keys}
    surrogate_kwargs = {k: v for k, v in kwargs.items() if k in surrogate_keys}
    
    if method == 'greedy':
        return optimize_greedy(
            df, response, categorical_factors, continuous_factors,
            verbose=verbose, **greedy_kwargs
        )
    elif method == 'rf':
        return optimize_rf(
            df, response, categorical_factors, continuous_factors,
            verbose=verbose, **surrogate_kwargs
        )
    elif method == 'nn':
        return optimize_nn(
            df, response, categorical_factors, continuous_factors,
            verbose=verbose, **surrogate_kwargs
        )
    elif method == 'all':
        results = {}
        for m in ['greedy', 'rf', 'nn']:
            if verbose:
                print(f"\n{'='*60}")
                print(f"method: {m.upper()}")
                print('='*60)
            method_kwargs = greedy_kwargs if m == 'greedy' else surrogate_kwargs
            results[m] = optimize(
                df, m, response, categorical_factors, continuous_factors, 
                verbose, **method_kwargs
            )
        return results
    else:
        raise ValueError(f"unknown method: {method}")


def compare_methods(results: Dict) -> None:
    """print comparison of optimization methods."""
    print("\n" + "="*70)
    print("method comparison")
    print("="*70)
    
    print(f"\n{'method':<10} {'predicted_f1':<15} {'model_fit':<20} {'n_samples':<10}")
    print("-"*60)
    
    for method, res in results.items():
        pred = res.get('predicted_f1', np.nan)
        n = res.get('n_samples', 0)
        
        if method == 'greedy':
            fit = f"n={n} after filter"
        elif method == 'rf':
            r2 = res.get('cv_r2', np.nan)
            fit = f"cv_R²={r2:.3f}" if not np.isnan(r2) else "N/A"
        elif method == 'nn':
            r2 = res.get('val_r2', np.nan)
            fit = f"val_R²={r2:.3f}" if not np.isnan(r2) else "N/A"
        else:
            fit = "N/A"
        
        print(f"{method:<10} {pred:<15.4f} {fit:<20} {n:<10}")

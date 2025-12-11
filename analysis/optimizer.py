#!/usr/bin/env python3
"""
optimizer.py - hyperparameter optimization methods
===================================================

depends on: core.py, stats.py

methods:
- greedy: pure greedy filtering on discretized factors (simple, interpretable)
- linear: regression-based estimation for continuous variables
- rf: random forest surrogate model
- nn: neural network surrogate model
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
# helper: discretize continuous variables
# =============================================================================

def discretize_continuous(df: pd.DataFrame, factor: str, n_bins: int = 4) -> pd.Series:
    """
    discretize a continuous variable into bins for greedy optimization.
    
    uses quantile-based binning to ensure roughly equal samples per bin.
    returns a series with bin labels like 'low', 'med-low', 'med-high', 'high'.
    """
    valid_mask = df[factor].notna()
    if valid_mask.sum() < n_bins:
        # not enough data to bin meaningfully
        return pd.Series(['all'] * len(df), index=df.index)
    
    values = df.loc[valid_mask, factor]
    
    # use quantile binning
    try:
        if n_bins == 2:
            labels = ['low', 'high']
        elif n_bins == 3:
            labels = ['low', 'med', 'high']
        elif n_bins == 4:
            labels = ['low', 'med-low', 'med-high', 'high']
        else:
            labels = [f'bin_{i}' for i in range(n_bins)]
        
        binned = pd.qcut(values, q=n_bins, labels=labels, duplicates='drop')
        
        # create full series with NaN for missing values
        result = pd.Series(index=df.index, dtype=object)
        result[valid_mask] = binned.astype(str)
        result[~valid_mask] = np.nan
        return result
        
    except ValueError:
        # qcut fails if too few unique values
        return pd.Series(['all'] * len(df), index=df.index)


def get_bin_representative(df: pd.DataFrame, factor: str, bin_label: str, 
                           binned_col: str) -> float:
    """get the median value of a bin for reporting."""
    mask = df[binned_col] == bin_label
    if mask.sum() == 0:
        return np.nan
    return df.loc[mask, factor].median()


# =============================================================================
# greedy optimization (pure greedy, no regression)
# =============================================================================

def optimize_greedy(df: pd.DataFrame, response: str = 'test_f1',
                    factors: List[str] = None,
                    continuous_factors: List[str] = None,
                    n_bins: int = 4,
                    min_samples_per_level: int = 5,
                    min_samples_after_filter: int = 10,
                    significance_threshold: float = 0.10,
                    verbose: bool = True) -> Dict:
    """
    pure greedy optimization: iteratively filter by best level of each factor.
    
    continuous variables are discretized into bins and treated the same as
    categorical variables. this is simple, interpretable, and avoids the
    complexity of regression-based estimation.
    
    algorithm:
        1. discretize continuous factors into bins
        2. for each iteration:
           - compute effect size (eta²) for all remaining factors
           - filter to best level of the factor with largest effect
           - repeat until no significant factors remain
    
    args:
        df: dataframe with experiment results
        response: target metric column
        factors: categorical factors to optimize (default: model, segment_length, etc.)
        continuous_factors: factors to discretize (default: log_lr, log_rnn_lr, log_wd)
        n_bins: number of bins for continuous variables
        min_samples_per_level: minimum runs needed to consider a level
        min_samples_after_filter: minimum runs needed after filtering
        significance_threshold: p-value threshold for including a factor
    """
    if factors is None:
        factors = ['segment_length', 'model', 'batch_size', 'diff_lr', 'cnn_frozen']
    if continuous_factors is None:
        continuous_factors = ['log_lr', 'log_rnn_lr', 'log_wd']
    
    # filter to rows with valid response
    df_work = df[df[response].notna()].copy()
    if len(df_work) == 0:
        if verbose:
            print("\n[greedy] no valid data (all response values are NaN)")
        return {
            'method': 'greedy',
            'path': [],
            'final_config': {},
            'predicted_f1': np.nan,
            'std_f1': np.nan,
            'n_samples': 0,
            'filtered_df': df_work
        }
    
    # discretize continuous factors
    binned_map = {}  # maps binned column name -> original column name
    for factor in continuous_factors:
        if factor not in df_work.columns:
            continue
        if df_work[factor].notna().sum() < n_bins:
            continue
        
        binned_col = f'{factor}_bin'
        df_work[binned_col] = discretize_continuous(df_work, factor, n_bins)
        binned_map[binned_col] = factor
    
    # combine all factors (categorical + binned continuous)
    all_factors = []
    for f in factors:
        if f in df_work.columns and df_work[f].nunique() > 1:
            all_factors.append(f)
    for binned_col in binned_map:
        if df_work[binned_col].nunique() > 1:
            all_factors.append(binned_col)
    
    if verbose:
        print(f"\n[greedy] optimizing {len(all_factors)} factors")
        print("-" * 60)
    
    df_current = df_work.copy()
    optimization_path = []
    remaining = all_factors.copy()
    
    while remaining and len(df_current) >= min_samples_after_filter:
        effects = []
        
        for factor in remaining:
            if df_current[factor].nunique() <= 1:
                continue
            
            # compute effect size
            result = compute_anova_oneway(df_current, factor, response)
            if np.isnan(result['p_value']) or result['p_value'] > significance_threshold:
                continue
            
            # find best level
            level_stats = df_current.groupby(factor)[response].agg(['mean', 'count'])
            level_stats = level_stats[level_stats['count'] >= min_samples_per_level]
            
            if len(level_stats) == 0:
                continue
            
            best_level = level_stats['mean'].idxmax()
            best_count = int(level_stats.loc[best_level, 'count'])
            
            if best_count < min_samples_after_filter:
                continue
            
            effects.append({
                'factor': factor,
                'eta_sq': result['eta_sq'],
                'p_value': result['p_value'],
                'best_level': best_level,
                'best_count': best_count,
                'best_mean': level_stats.loc[best_level, 'mean']
            })
        
        if not effects:
            break
        
        # pick factor with largest effect size
        effects_df = pd.DataFrame(effects).sort_values('eta_sq', ascending=False)
        best = effects_df.iloc[0]
        
        # get stats for the best level
        level_stats = df_current.groupby(best['factor'])[response].agg(['mean', 'std', 'count'])
        best_mean = level_stats.loc[best['best_level'], 'mean']
        best_std = level_stats.loc[best['best_level'], 'std']
        if pd.isna(best_std):
            best_std = 0.0
        
        # determine if this is a binned continuous factor
        factor_name = best['factor']
        is_continuous = factor_name in binned_map
        original_factor = binned_map.get(factor_name, factor_name)
        
        # for binned factors, get the representative value
        if is_continuous:
            representative_val = get_bin_representative(
                df_current, original_factor, best['best_level'], factor_name
            )
        else:
            representative_val = None
        
        step = {
            'step': len(optimization_path) + 1,
            'factor': original_factor,
            'factor_binned': factor_name if is_continuous else None,
            'type': 'continuous' if is_continuous else 'categorical',
            'eta_sq': best['eta_sq'],
            'p_value': best['p_value'],
            'best_level': best['best_level'],
            'representative_value': representative_val,
            'mean_f1': best_mean,
            'std_f1': best_std,
            'n_runs': int(best['best_count']),
            'n_before': len(df_current)
        }
        optimization_path.append(step)
        
        if verbose:
            if is_continuous:
                orig = original_factor.replace('log_', '')
                if original_factor.startswith('log_') and representative_val is not None:
                    actual_val = 10 ** representative_val
                    print(f"  {orig}: {best['best_level']} (~{actual_val:.2e}) (η²={step['eta_sq']:.4f}, n={step['n_runs']})")
                else:
                    print(f"  {orig}: {best['best_level']} (η²={step['eta_sq']:.4f}, n={step['n_runs']})")
            else:
                print(f"  {factor_name}: {best['best_level']} (η²={step['eta_sq']:.4f}, n={step['n_runs']})")
        
        # filter to best level
        df_current = df_current[df_current[best['factor']] == best['best_level']].copy()
        remaining.remove(best['factor'])
    
    # build final config from the best run in the filtered subset
    final_config = {}
    if len(df_current) > 0:
        best_idx = df_current[response].idxmax()
        best_run = df_current.loc[best_idx]
        
        # extract config from best run
        config_cols = ['model', 'segment_length', 'batch_size', 'diff_lr', 'cnn_frozen']
        for col in config_cols:
            if col in df_current.columns and pd.notna(best_run.get(col)):
                final_config[col] = best_run[col]
        
        # extract continuous values (convert from log)
        log_cols = ['log_lr', 'log_rnn_lr', 'log_wd']
        for log_col in log_cols:
            if log_col in df_current.columns and pd.notna(best_run.get(log_col)):
                orig = log_col.replace('log_', '')
                final_config[orig] = 10 ** best_run[log_col]
    
    final_mean = df_current[response].mean() if len(df_current) > 0 else np.nan
    final_std = df_current[response].std() if len(df_current) > 0 else 0.0
    if pd.isna(final_std):
        final_std = 0.0
    
    if verbose:
        print(f"\n  final: n={len(df_current)}, mean={final_mean:.4f}, std={final_std:.4f}")
    
    return {
        'method': 'greedy',
        'path': optimization_path,
        'final_config': final_config,
        'predicted_f1': final_mean,
        'std_f1': final_std,
        'n_samples': len(df_current),
        'filtered_df': df_current
    }


# =============================================================================
# linear optimization (pure linear regression)
# =============================================================================

def optimize_linear(df: pd.DataFrame, response: str = 'test_f1',
                    categorical_factors: List[str] = None,
                    continuous_factors: List[str] = None,
                    verbose: bool = True) -> Dict:
    """
    pure linear regression optimization.
    
    fits a single linear model: y = β₀ + Σβᵢxᵢ
    - categorical factors: one-hot encoded, pick level with highest coefficient
    - continuous factors: use coefficient sign to pick min or max observed value
    
    simple, interpretable, assumes linear response surface.
    """
    if categorical_factors is None:
        categorical_factors = ['segment_length', 'model', 'batch_size', 'diff_lr', 'cnn_frozen']
    if continuous_factors is None:
        continuous_factors = ['log_lr', 'log_rnn_lr', 'log_wd']
    
    # filter to rows with valid response
    df_valid = df[df[response].notna()].copy()
    if len(df_valid) < 10:
        if verbose:
            print(f"\n[linear] insufficient data (n={len(df_valid)})")
        return {
            'method': 'linear',
            'coefficients': {},
            'final_config': {},
            'predicted_f1': np.nan,
            'std_f1': np.nan,
            'r_squared': np.nan,
            'n_samples': len(df_valid)
        }
    
    categorical_factors = [f for f in categorical_factors if f in df_valid.columns and df_valid[f].nunique() > 1]
    continuous_factors = [f for f in continuous_factors if f in df_valid.columns and df_valid[f].notna().sum() > 0]
    
    if verbose:
        print(f"\n[linear] fitting linear regression (n={len(df_valid)})")
        print("-" * 60)
    
    # build design matrix
    X_parts = []
    feature_names = []
    feature_info = []  # track (factor, level_or_none) for each feature
    
    # one-hot encode categorical factors (drop first level as reference)
    for factor in categorical_factors:
        levels = sorted(df_valid[factor].dropna().unique())
        if len(levels) < 2:
            continue
        reference = levels[0]
        for level in levels[1:]:
            col = (df_valid[factor] == level).astype(float).values
            X_parts.append(col.reshape(-1, 1))
            feature_names.append(f"{factor}={level}")
            feature_info.append((factor, level, reference))
    
    # add continuous factors (standardized)
    continuous_stats = {}
    for factor in continuous_factors:
        vals = pd.to_numeric(df_valid[factor], errors='coerce').values
        valid_mask = np.isfinite(vals)
        if valid_mask.sum() < 5:
            continue
        
        mean_val = np.nanmean(vals)
        std_val = np.nanstd(vals)
        if std_val < 1e-10:
            continue
        
        standardized = (vals - mean_val) / std_val
        standardized = np.nan_to_num(standardized, nan=0.0)
        X_parts.append(standardized.reshape(-1, 1))
        feature_names.append(factor)
        feature_info.append((factor, None, None))
        continuous_stats[factor] = {
            'mean': mean_val, 'std': std_val,
            'min': float(np.nanmin(vals)), 'max': float(np.nanmax(vals))
        }
    
    if not X_parts:
        if verbose:
            print("  no valid features")
        return {
            'method': 'linear',
            'coefficients': {},
            'final_config': {},
            'predicted_f1': np.nan,
            'std_f1': np.nan,
            'r_squared': np.nan,
            'n_samples': len(df_valid)
        }
    
    X = np.hstack(X_parts)
    y = df_valid[response].values
    
    # add intercept
    X = np.hstack([np.ones((X.shape[0], 1)), X])
    feature_names = ['intercept'] + feature_names
    feature_info = [(None, None, None)] + feature_info
    
    # fit linear regression via least squares
    try:
        coeffs, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)
    except Exception as e:
        if verbose:
            print(f"  regression failed: {e}")
        return {
            'method': 'linear',
            'coefficients': {},
            'final_config': {},
            'predicted_f1': np.nan,
            'std_f1': np.nan,
            'r_squared': np.nan,
            'n_samples': len(df_valid)
        }
    
    # compute R²
    y_pred = X @ coeffs
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    
    if verbose:
        print(f"  R² = {r_squared:.4f}")
        print(f"\n  coefficients:")
    
    # store coefficients
    coef_dict = {}
    for name, coef in zip(feature_names, coeffs):
        coef_dict[name] = float(coef)
        if verbose and name != 'intercept':
            print(f"    {name}: {coef:+.4f}")
    
    # build optimal config from coefficients
    final_config = {}
    
    # for categorical: pick level with highest coefficient (or reference if all negative)
    categorical_processed = set()
    for i, (factor, level, reference) in enumerate(feature_info):
        if factor is None or level is None:
            continue
        if factor in categorical_processed:
            continue
        
        # gather all coefficients for this factor
        factor_coeffs = {reference: 0.0}  # reference level has implicit coef of 0
        for j, (f, l, r) in enumerate(feature_info):
            if f == factor and l is not None:
                factor_coeffs[l] = coeffs[j]
        
        best_level = max(factor_coeffs, key=factor_coeffs.get)
        final_config[factor] = best_level
        categorical_processed.add(factor)
    
    # for continuous: use coefficient sign to pick min or max
    for factor, stats in continuous_stats.items():
        idx = feature_names.index(factor)
        coef = coeffs[idx]
        
        # positive coef means higher values are better
        if coef > 0:
            optimal_val = stats['max']
        else:
            optimal_val = stats['min']
        
        # convert from log if needed
        orig = factor.replace('log_', '')
        if factor.startswith('log_'):
            final_config[orig] = 10 ** optimal_val
        else:
            final_config[orig] = optimal_val
    
    # predict F1 for optimal config
    x_opt = np.ones(len(feature_names))
    for i, (factor, level, reference) in enumerate(feature_info):
        if factor is None:
            continue
        if level is not None:
            # categorical: 1 if this is the chosen level, 0 otherwise
            x_opt[i] = 1.0 if final_config.get(factor) == level else 0.0
        else:
            # continuous: use optimal value (standardized)
            stats = continuous_stats[factor]
            orig = factor.replace('log_', '')
            opt_val = np.log10(final_config[orig]) if factor.startswith('log_') else final_config[orig]
            x_opt[i] = (opt_val - stats['mean']) / stats['std']
    
    predicted_f1 = float(x_opt @ coeffs)
    
    if verbose:
        print(f"\n  best predicted: {predicted_f1:.4f}")
    
    return {
        'method': 'linear',
        'coefficients': coef_dict,
        'final_config': final_config,
        'predicted_f1': predicted_f1,
        'std_f1': 0.0,
        'r_squared': r_squared,
        'n_samples': len(df_valid),
        'continuous_stats': continuous_stats
    }


def _estimate_continuous_optimal(df: pd.DataFrame, factor: str, 
                                  response: str = 'test_f1') -> Dict:
    """
    estimate optimal value for continuous hp via regression.
    internal helper for optimize_linear.
    """
    valid_mask = df[factor].notna() & df[response].notna()
    if valid_mask.sum() == 0:
        return {
            'factor': factor, 'optimal': np.nan, 'correlation': np.nan,
            'p_value': np.nan, 'method': 'insufficient_data', 'n_samples': 0,
            'observed_range': (np.nan, np.nan)
        }
    
    try:
        x = np.array([float(v) if v is not None else np.nan for v in df.loc[valid_mask, factor].values])
        y = np.array([float(v) if v is not None else np.nan for v in df.loc[valid_mask, response].values])
    except (TypeError, ValueError):
        return {
            'factor': factor, 'optimal': np.nan, 'correlation': np.nan,
            'p_value': np.nan, 'method': 'conversion_failed', 'n_samples': 0,
            'observed_range': (np.nan, np.nan)
        }
    
    finite_mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite_mask], y[finite_mask]
    n_samples = len(x)
    
    if n_samples < 5:
        mean_val = float(np.mean(x)) if n_samples > 0 else np.nan
        return {
            'factor': factor, 'optimal': mean_val, 'correlation': np.nan,
            'p_value': np.nan, 'method': 'insufficient_data', 'n_samples': n_samples,
            'observed_range': (float(np.min(x)), float(np.max(x))) if n_samples > 0 else (np.nan, np.nan)
        }
    
    n_unique = len(np.unique(x))
    x_range = float(np.max(x) - np.min(x))
    
    if n_unique < 3 or x_range < 1e-10:
        return {
            'factor': factor, 'optimal': float(np.mean(x)), 'correlation': np.nan,
            'p_value': np.nan, 'method': 'insufficient_variation', 'n_samples': n_samples,
            'n_unique': n_unique, 'observed_range': (float(np.min(x)), float(np.max(x)))
        }
    
    if len(np.unique(y)) == 1:
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
        return optimize_greedy(df, response, verbose=verbose)
    
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
        return optimize_greedy(df, response, verbose=verbose)
    
    X, encoder = encode_features(df_clean, categorical_cols, continuous_cols)
    y = df_clean[response].values
    
    if verbose:
        print(f"\n[rf] training random forest (n={len(df_clean)}, features={X.shape[1]})")
        print("-" * 60)
    
    # handle any remaining NaN
    X = np.nan_to_num(X, nan=0.0)
    y = np.nan_to_num(y, nan=np.nanmean(y) if np.any(~np.isnan(y)) else 0.5)
    
    model = RandomForestRegressor(
        n_estimators=100, max_depth=10, min_samples_leaf=3,
        random_state=42, n_jobs=-1
    )
    model.fit(X, y)
    
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
        'std_f1': 0.0,
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
    
    df_clean = df[df[response].notna()].copy()
    if len(df_clean) < 10:
        if verbose:
            print(f"  insufficient data after filtering NaN (n={len(df_clean)})")
        return optimize_greedy(df, response, verbose=verbose)
    
    X, encoder = encode_features(df_clean, categorical_cols, continuous_cols)
    y = df_clean[response].values
    
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
        'std_f1': 0.0,
        'top_configs': top_configs,
        'n_samples': len(df_clean)
    }


# =============================================================================
# unified interface
# =============================================================================

def optimize(df: pd.DataFrame, method: str = 'greedy',
             response: str = 'test_f1', verbose: bool = True,
             **kwargs) -> Dict:
    """
    unified optimization interface.
    
    methods:
        greedy: pure greedy filtering (discretizes continuous variables)
        linear: pure linear regression on all factors
        rf: random forest surrogate model
        nn: neural network surrogate model
        all: run all methods and compare
    
    args:
        df: dataframe with experiment results (should already be filtered for near-duplicates)
        method: optimization method
        response: target metric column
        verbose: print progress
    """
    # separate kwargs by method
    greedy_keys = {'factors', 'continuous_factors', 'n_bins', 
                   'min_samples_per_level', 'min_samples_after_filter', 'significance_threshold'}
    linear_keys = {'categorical_factors', 'continuous_factors'}
    surrogate_keys = {'categorical_cols', 'continuous_cols', 'n_candidates', 'hidden_sizes', 'epochs'}
    
    greedy_kwargs = {k: v for k, v in kwargs.items() if k in greedy_keys}
    linear_kwargs = {k: v for k, v in kwargs.items() if k in linear_keys}
    surrogate_kwargs = {k: v for k, v in kwargs.items() if k in surrogate_keys}
    
    if method == 'greedy':
        return optimize_greedy(df, response, verbose=verbose, **greedy_kwargs)
    elif method == 'linear':
        return optimize_linear(df, response, verbose=verbose, **linear_kwargs)
    elif method == 'rf':
        return optimize_rf(df, response, verbose=verbose, **surrogate_kwargs)
    elif method == 'nn':
        return optimize_nn(df, response, verbose=verbose, **surrogate_kwargs)
    elif method == 'all':
        results = {}
        for m in ['greedy', 'linear', 'rf', 'nn']:
            if verbose:
                print(f"\n{'='*60}")
                print(f"  {m.upper()} OPTIMIZATION")
                print('='*60)
            try:
                if m == 'greedy':
                    results[m] = optimize_greedy(df, response, verbose=verbose, **greedy_kwargs)
                elif m == 'linear':
                    results[m] = optimize_linear(df, response, verbose=verbose, **linear_kwargs)
                elif m == 'rf':
                    results[m] = optimize_rf(df, response, verbose=verbose, **surrogate_kwargs)
                elif m == 'nn':
                    results[m] = optimize_nn(df, response, verbose=verbose, **surrogate_kwargs)
            except Exception as e:
                if verbose:
                    print(f"  {m} failed: {e}")
                results[m] = {'method': m, 'error': str(e)}
        
        if verbose:
            compare_methods(results)
        
        # return the best result
        best_method = None
        best_f1 = -np.inf
        for m, r in results.items():
            if 'predicted_f1' in r and not np.isnan(r['predicted_f1']):
                if r['predicted_f1'] > best_f1:
                    best_f1 = r['predicted_f1']
                    best_method = m
        
        if best_method:
            result = results[best_method].copy()
            result['all_results'] = results
            return result
        else:
            return {'method': 'all', 'all_results': results, 'predicted_f1': np.nan}
    else:
        raise ValueError(f"unknown method: {method}. choose from: greedy, linear, rf, nn, all")


def compare_methods(results: Dict) -> None:
    """print comparison table and optimal configs from all methods."""
    print("\n" + "="*60)
    print("METHOD COMPARISON")
    print("="*60)
    print(f"{'Method':<10} {'Predicted F1':<15} {'N Samples':<12} {'Status'}")
    print("-"*60)
    
    for method, result in results.items():
        if 'error' in result:
            print(f"{method:<10} {'N/A':<15} {'N/A':<12} ERROR: {result['error'][:30]}")
        else:
            pred = result.get('predicted_f1', np.nan)
            n = result.get('n_samples', 0)
            pred_str = f"{pred:.4f}" if not np.isnan(pred) else "N/A"
            print(f"{method:<10} {pred_str:<15} {n:<12} OK")
    
    print("-"*60)
    
    # show optimal config from each method
    print("\nOPTIMAL CONFIGURATIONS BY METHOD")
    print("="*60)
    
    for method, result in results.items():
        if 'error' in result:
            continue
        
        config = result.get('final_config', {})
        pred = result.get('predicted_f1', np.nan)
        
        if not config:
            continue
        
        print(f"\n{method.upper()} (predicted F1: {pred:.4f}):")
        for key, val in config.items():
            if isinstance(val, float):
                print(f"  {key}: {val:.2e}")
            else:
                print(f"  {key}: {val}")
    
    # identify overall best
    best_method = None
    best_f1 = -np.inf
    for m, r in results.items():
        if 'predicted_f1' in r and not np.isnan(r['predicted_f1']):
            if r['predicted_f1'] > best_f1:
                best_f1 = r['predicted_f1']
                best_method = m
    
    if best_method:
        print("\n" + "="*60)
        print(f"BEST METHOD: {best_method.upper()} (predicted F1: {best_f1:.4f})")
        print("="*60)

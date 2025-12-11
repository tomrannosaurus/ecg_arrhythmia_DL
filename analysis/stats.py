#!/usr/bin/env python3
"""
stats.py - statistical analysis functions
=========================================

depends only on: core.py (and standard libs)

contents:
- effect size calculations (ANOVA, correlation)
- statistical tests (t-tests, interaction tests)
- variance decomposition
- model feature analysis
"""

import warnings
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats


# =============================================================================
# basic effect size calculations
# =============================================================================

def compute_anova_oneway(df: pd.DataFrame, factor: str, 
                          response: str = 'test_f1') -> Dict:
    """
    one-way anova for categorical factor effect.
    
    returns:
        dict with f_stat, p_value, eta_squared, n_groups
    """
    groups = [group[response].dropna().values for name, group in df.groupby(factor)]
    groups = [g for g in groups if len(g) > 0]
    
    if len(groups) < 2:
        return {'f_stat': np.nan, 'p_value': np.nan, 'eta_sq': np.nan, 'n_groups': len(groups)}
    
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        try:
            f_stat, p_value = scipy_stats.f_oneway(*groups)
        except:
            f_stat, p_value = np.nan, np.nan
    
    # eta-squared (effect size)
    grand_mean = df[response].mean()
    ss_between = sum(len(g) * (g.mean() - grand_mean)**2 for g in groups)
    ss_total = sum((df[response] - grand_mean)**2)
    eta_sq = ss_between / ss_total if ss_total > 0 else 0
    
    return {
        'f_stat': float(f_stat) if not np.isnan(f_stat) else np.nan,
        'p_value': float(p_value) if not np.isnan(p_value) else np.nan,
        'eta_sq': float(eta_sq),
        'n_groups': len(groups)
    }


def compute_correlation(df: pd.DataFrame, factor: str, 
                        response: str = 'test_f1') -> Dict:
    """
    pearson correlation for continuous factor.
    
    returns:
        dict with r, p_value, r_squared, n
    """
    x = df[factor].dropna()
    y = df.loc[x.index, response]
    
    if len(x) < 3:
        return {'r': np.nan, 'p_value': np.nan, 'r_sq': np.nan, 'n': len(x)}
    
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        try:
            r, p_value = scipy_stats.pearsonr(x, y)
        except:
            r, p_value = np.nan, np.nan
    
    return {
        'r': float(r) if not np.isnan(r) else np.nan,
        'p_value': float(p_value) if not np.isnan(p_value) else np.nan,
        'r_sq': float(r**2) if not np.isnan(r) else np.nan,
        'n': len(x)
    }


def compute_welch_ttest(group1: np.ndarray, group2: np.ndarray) -> Dict:
    """
    welch's t-test for comparing two groups with unequal variances.
    """
    if len(group1) < 2 or len(group2) < 2:
        return {'t_stat': np.nan, 'p_value': np.nan, 'effect_size': np.nan}
    
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        try:
            t_stat, p_value = scipy_stats.ttest_ind(group1, group2, equal_var=False)
        except:
            t_stat, p_value = np.nan, np.nan
    
    # cohen's d effect size
    pooled_std = np.sqrt((group1.std()**2 + group2.std()**2) / 2)
    if pooled_std > 0:
        effect_size = (group1.mean() - group2.mean()) / pooled_std
    else:
        effect_size = np.nan
    
    return {
        't_stat': float(t_stat) if not np.isnan(t_stat) else np.nan,
        'p_value': float(p_value) if not np.isnan(p_value) else np.nan,
        'effect_size': float(effect_size) if not np.isnan(effect_size) else np.nan
    }


# =============================================================================
# main effects analysis
# =============================================================================

def analyze_main_effects(df: pd.DataFrame, response: str = 'test_f1',
                         categorical_factors: List[str] = None,
                         continuous_factors: List[str] = None) -> pd.DataFrame:
    """
    analyze main effects of all factors on response.
    
    returns dataframe with effect sizes and p-values for each factor.
    """
    if categorical_factors is None:
        categorical_factors = ['model', 'segment_length', 'batch_size', 'diff_lr', 'cnn_frozen']
    if continuous_factors is None:
        continuous_factors = ['log_lr', 'log_rnn_lr', 'log_wd']
    
    results = []
    
    # categorical factors (ANOVA)
    for factor in categorical_factors:
        if factor not in df.columns or df[factor].nunique() <= 1:
            continue
        
        result = compute_anova_oneway(df, factor, response)
        results.append({
            'factor': factor,
            'type': 'categorical',
            'effect_size': result['eta_sq'],
            'p_value': result['p_value'],
            'n_levels': result['n_groups']
        })
    
    # continuous factors (correlation)
    for factor in continuous_factors:
        if factor not in df.columns:
            continue
        
        result = compute_correlation(df, factor, response)
        results.append({
            'factor': factor,
            'type': 'continuous',
            'effect_size': result['r_sq'],
            'p_value': result['p_value'],
            'n_levels': result['n']
        })
    
    effects_df = pd.DataFrame(results)
    if len(effects_df) > 0:
        effects_df = effects_df.sort_values('effect_size', ascending=False)
    
    return effects_df


# =============================================================================
# interaction analysis
# =============================================================================

def compute_interaction_means(df: pd.DataFrame, factor1: str, factor2: str,
                              response: str = 'test_f1') -> pd.DataFrame:
    """compute cell means for two-factor interaction."""
    return df.groupby([factor1, factor2])[response].agg(['mean', 'std', 'count']).reset_index()


def test_interaction(df: pd.DataFrame, factor1: str, factor2: str,
                     response: str = 'test_f1') -> Dict:
    """
    test for interaction between two factors.
    
    uses a simple approach: compare variance explained by
    main effects vs main effects + interaction.
    """
    # compute cell means
    cell_means = df.groupby([factor1, factor2])[response].mean()
    grand_mean = df[response].mean()
    
    # main effect means
    f1_means = df.groupby(factor1)[response].mean()
    f2_means = df.groupby(factor2)[response].mean()
    
    # compute sum of squares
    ss_total = ((df[response] - grand_mean)**2).sum()
    
    # ss for main effects (additive model)
    ss_f1 = sum(df.groupby(factor1)[response].count() * (f1_means - grand_mean)**2)
    ss_f2 = sum(df.groupby(factor2)[response].count() * (f2_means - grand_mean)**2)
    
    # ss for full model (with interaction)
    ss_cells = 0
    for (l1, l2), mean in cell_means.items():
        n_cell = len(df[(df[factor1] == l1) & (df[factor2] == l2)])
        ss_cells += n_cell * (mean - grand_mean)**2
    
    # interaction ss = full model - additive model
    ss_interaction = ss_cells - ss_f1 - ss_f2
    
    # eta-squared for interaction
    eta_sq_interaction = ss_interaction / ss_total if ss_total > 0 else 0
    
    # classify strength
    if eta_sq_interaction > 0.06:
        strength = 'strong'
    elif eta_sq_interaction > 0.01:
        strength = 'medium'
    else:
        strength = 'weak'
    
    return {
        'factor1': factor1,
        'factor2': factor2,
        'eta_sq': float(max(0, eta_sq_interaction)),
        'strength': strength,
        'ss_interaction': float(ss_interaction),
        'ss_total': float(ss_total)
    }


def analyze_interactions(df: pd.DataFrame, response: str = 'test_f1',
                         factors: List[str] = None) -> pd.DataFrame:
    """analyze all pairwise interactions between factors."""
    if factors is None:
        factors = ['model', 'segment_length', 'batch_size', 'diff_lr', 'cnn_frozen']
    
    factors = [f for f in factors if f in df.columns and df[f].nunique() > 1]
    
    results = []
    for i, f1 in enumerate(factors):
        for f2 in factors[i+1:]:
            result = test_interaction(df, f1, f2, response)
            results.append(result)
    
    interactions_df = pd.DataFrame(results)
    if len(interactions_df) > 0:
        interactions_df = interactions_df.sort_values('eta_sq', ascending=False)
    
    return interactions_df


# =============================================================================
# variance decomposition
# =============================================================================

def compute_variance_components(df: pd.DataFrame, response: str = 'test_f1') -> Dict:
    """
    decompose total variance into components.
    
    components:
    - model architecture
    - segment length
    - residual (seed + hyperparameters)
    """
    total_var = df[response].var()
    
    if total_var == 0 or np.isnan(total_var):
        return {
            'total_variance': 0,
            'model_pct': 0,
            'segment_pct': 0,
            'residual_pct': 100
        }
    
    # model effect
    model_means = df.groupby('model')[response].transform('mean')
    model_var = model_means.var()
    
    # segment length effect
    segment_means = df.groupby('segment_length')[response].transform('mean')
    segment_var = segment_means.var()
    
    # residual
    residual_var = total_var - model_var - segment_var
    residual_var = max(0, residual_var)
    
    return {
        'total_variance': float(total_var),
        'model_pct': float(100 * model_var / total_var),
        'segment_pct': float(100 * segment_var / total_var),
        'residual_pct': float(100 * residual_var / total_var)
    }


def analyze_variance_by_seed(df: pd.DataFrame, response: str = 'test_f1') -> pd.DataFrame:
    """
    analyze within-configuration variance (due to random seed).
    
    groups by (model, segment_length, batch_size, lr, diff_lr)
    and computes variance across seeds.
    """
    config_cols = ['model', 'segment_length', 'batch_size', 'diff_lr']
    config_cols = [c for c in config_cols if c in df.columns]
    
    if not config_cols:
        return pd.DataFrame()
    
    variance_by_config = df.groupby(config_cols)[response].agg(['mean', 'std', 'count'])
    variance_by_config = variance_by_config[variance_by_config['count'] >= 2]
    variance_by_config = variance_by_config.sort_values('std', ascending=False)
    
    return variance_by_config.reset_index()


# =============================================================================
# model feature analysis
# =============================================================================

def analyze_model_features(df: pd.DataFrame, response: str = 'test_f1') -> pd.DataFrame:
    """
    analyze effect of individual architectural features.
    
    requires model features to be decomposed (has_cnn, has_rnn, etc.)
    """
    feature_cols = ['has_cnn', 'has_rnn', 'rnn_type', 'is_bidirectional', 
                    'has_attention', 'has_residual', 'has_ln_after_rnn', 'pooling']
    
    results = []
    for feature in feature_cols:
        if feature not in df.columns:
            continue
        
        if df[feature].nunique() <= 1:
            continue
        
        result = compute_anova_oneway(df, feature, response)
        
        # find best level
        level_means = df.groupby(feature)[response].mean()
        best_level = level_means.idxmax()
        best_mean = level_means.max()
        
        results.append({
            'feature': feature,
            'eta_sq': result['eta_sq'],
            'p_value': result['p_value'],
            'best_level': best_level,
            'best_mean': best_mean,
            'n_levels': result['n_groups']
        })
    
    features_df = pd.DataFrame(results)
    if len(features_df) > 0:
        features_df = features_df.sort_values('eta_sq', ascending=False)
    
    return features_df


# =============================================================================
# configuration analysis
# =============================================================================

def find_top_configs(df: pd.DataFrame, n: int = 10, 
                     response: str = 'test_f1') -> pd.DataFrame:
    """find top n configurations by response metric."""
    cols = ['model', 'segment_length', 'lr', 'rnn_lr', 'weight_decay', 
            'batch_size', 'diff_lr', 'cnn_frozen', 'seed',
            'test_f1', 'test_auroc', 'test_acc', 'run_id']
    cols = [c for c in cols if c in df.columns]
    
    return df.nlargest(n, response)[cols]


def find_robust_configs(df: pd.DataFrame, min_seeds: int = 2,
                        response: str = 'test_f1') -> pd.DataFrame:
    """
    find configurations that are stable across seeds.
    
    groups by (model, segment_length) and computes stats.
    high mean - low std = robust configuration.
    """
    config_cols = ['model', 'segment_length']
    config_cols = [c for c in config_cols if c in df.columns]
    
    if not config_cols:
        return pd.DataFrame()
    
    stats = df.groupby(config_cols).agg({
        response: ['mean', 'std', 'count', 'min', 'max'],
        'test_auroc': 'mean'
    }).reset_index()
    
    # flatten column names
    stats.columns = config_cols + ['mean_f1', 'std_f1', 'count', 'min_f1', 'max_f1', 'mean_auroc']
    
    # filter by minimum seeds
    stats = stats[stats['count'] >= min_seeds]
    
    # fill NaN std with 0 (single observation)
    stats['std_f1'] = stats['std_f1'].fillna(0)
    
    # conservative score: mean - std
    stats['conservative_score'] = stats['mean_f1'] - stats['std_f1']
    stats = stats.sort_values('conservative_score', ascending=False)
    
    return stats


def compute_marginal_means(df: pd.DataFrame, factor: str,
                           response: str = 'test_f1') -> pd.DataFrame:
    """compute marginal means and confidence intervals for a factor."""
    results = []
    
    for level in df[factor].unique():
        subset = df[df[factor] == level][response].dropna()
        n = len(subset)
        mean = subset.mean()
        std = subset.std()
        
        if n > 1 and not np.isnan(std):
            se = std / np.sqrt(n)
            ci = scipy_stats.t.interval(0.95, n-1, loc=mean, scale=se)
        else:
            ci = (mean, mean)
        
        results.append({
            'level': level,
            'mean': mean,
            'std': std if not np.isnan(std) else 0,
            'n': n,
            'ci_lower': ci[0],
            'ci_upper': ci[1]
        })
    
    return pd.DataFrame(results).sort_values('mean', ascending=False)

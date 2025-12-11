#!/usr/bin/env python3
"""
outputs.py - visualization and report generation
=================================================

depends on: core.py, stats.py, optimizer.py

contents:
- printing utilities (summaries, tables)
- visualization (matplotlib plots)
- report generation (text reports)
- interpretation (answering DOE questions)
"""

from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

# imports from project modules
from core import summarize_data, decompose_model_features
from stats import (
    analyze_main_effects, analyze_interactions, analyze_model_features,
    compute_variance_components, find_top_configs, find_robust_configs,
    compute_marginal_means, compute_welch_ttest
)
from optimizer import optimize, optimize_greedy


# =============================================================================
# printing utilities
# =============================================================================

def print_data_summary(df: pd.DataFrame) -> None:
    """print basic data summary."""
    summary = summarize_data(df)
    
    print("\n" + "="*70)
    print("data summary")
    print("="*70)
    
    print(f"\ntotal runs: {summary['n_runs']}")
    print(f"unique models: {summary['n_models']}")
    print(f"unique seeds: {summary['n_seeds']}")
    
    print("\nruns by model:")
    for model, count in sorted(summary['models'].items(), key=lambda x: -x[1]):
        print(f"  {model}: {count}")
    
    print("\nruns by segment length:")
    for seg, count in sorted(summary['segment_lengths'].items()):
        print(f"  {seg}s: {count}")
    
    if 'test_f1_range' in summary:
        lo, hi = summary['test_f1_range']
        print(f"\ntest_f1 range: [{lo:.4f}, {hi:.4f}]")
        print(f"test_f1 mean: {summary['test_f1_mean']:.4f}")


def print_main_effects(effects_df: pd.DataFrame) -> None:
    """print main effects analysis results."""
    print("\n" + "="*70)
    print("main effects analysis")
    print("="*70)
    
    print("\nfactors ranked by effect size:")
    print("-"*70)
    
    for _, row in effects_df.iterrows():
        sig = ''
        if row['p_value'] < 0.001:
            sig = '***'
        elif row['p_value'] < 0.01:
            sig = '**'
        elif row['p_value'] < 0.05:
            sig = '*'
        
        print(f"  {row['factor']:20s} effect={row['effect_size']:.4f}  p={row['p_value']:.4f} {sig}")
    
    print("-"*70)
    print("significance: *** p<0.001, ** p<0.01, * p<0.05")


def print_model_leaderboard(df: pd.DataFrame, response: str = 'test_f1') -> None:
    """print model performance leaderboard."""
    print("\n" + "="*70)
    print("model leaderboard")
    print("="*70)
    
    stats = df.groupby('model').agg({
        response: ['mean', 'std', 'count', 'max'],
        'test_auroc': 'mean'
    }).round(4)
    
    stats.columns = ['mean_f1', 'std_f1', 'n_runs', 'best_f1', 'mean_auroc']
    stats = stats.sort_values('mean_f1', ascending=False)
    
    print("\n" + stats.to_string())
    print("-"*70)


def print_top_configs(df: pd.DataFrame, n: int = 10, response: str = 'test_f1') -> None:
    """print top n configurations."""
    print("\n" + "="*70)
    print(f"top {n} configurations by {response}")
    print("="*70)
    
    top = find_top_configs(df, n, response)
    print("\n" + top.to_string(index=False))
    print("-"*70)


def print_optimization_result(result: Dict, response: str = 'test_f1') -> None:
    """print optimization result."""
    print("\n" + "-"*60)
    print("final optimized configuration:")
    
    for k, v in result['final_config'].items():
        if isinstance(v, float):
            if np.isnan(v):
                print(f"  {k}: NaN")
            elif v < 0.01:
                print(f"  {k}: {v:.2e}")
            else:
                print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")
    
    pred_f1 = result.get('predicted_f1', np.nan)
    std_f1 = result.get('std_f1', 0.0)
    n_samples = result.get('n_samples', 0)
    
    if np.isnan(pred_f1):
        print(f"\nexpected {response}: NaN (optimization failed)")
    elif std_f1 > 0:
        print(f"\nexpected {response}: {pred_f1:.4f} +/- {std_f1:.4f}")
    else:
        print(f"\nexpected {response}: {pred_f1:.4f}")
    
    print(f"based on n={n_samples} runs")


# =============================================================================
# visualization
# =============================================================================

def setup_matplotlib():
    """setup matplotlib with available style."""
    import matplotlib.pyplot as plt
    
    for style in ['seaborn-v0_8-whitegrid', 'seaborn-whitegrid', 'ggplot']:
        try:
            plt.style.use(style)
            break
        except:
            continue
    
    return plt


def plot_main_effects(effects_df: pd.DataFrame, output_path: str = None) -> None:
    """bar chart of main effects."""
    plt = setup_matplotlib()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['#2ecc71' if p < 0.05 else '#e74c3c' for p in effects_df['p_value']]
    
    ax.barh(effects_df['factor'], effects_df['effect_size'], color=colors)
    ax.set_xlabel('Effect Size (η² or r²)')
    ax.set_title('Main Effects on Test F1')
    ax.invert_yaxis()
    
    # legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2ecc71', label='Significant (p<0.05)'),
        Patch(facecolor='#e74c3c', label='Not Significant')
    ]
    ax.legend(handles=legend_elements, loc='lower right')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"saved: {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_model_boxplot(df: pd.DataFrame, response: str = 'test_f1',
                       output_path: str = None) -> None:
    """boxplot of model performance."""
    plt = setup_matplotlib()
    
    # order by median
    model_order = df.groupby('model')[response].median().sort_values(ascending=False).index
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    data = [df[df['model'] == m][response].values for m in model_order]
    bp = ax.boxplot(data, labels=model_order, vert=True, patch_artist=True)
    
    for patch in bp['boxes']:
        patch.set_facecolor('#3498db')
        patch.set_alpha(0.7)
    
    ax.set_ylabel(response)
    ax.set_title('Model Performance Distribution')
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"saved: {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_interaction_heatmap(df: pd.DataFrame, factor1: str, factor2: str,
                             response: str = 'test_f1', output_path: str = None) -> None:
    """heatmap of two-factor interaction."""
    plt = setup_matplotlib()
    
    pivot = df.pivot_table(values=response, index=factor1, columns=factor2, aggfunc='mean')
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    im = ax.imshow(pivot.values, cmap='RdYlGn', aspect='auto')
    
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha='right')
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    
    ax.set_xlabel(factor2)
    ax.set_ylabel(factor1)
    ax.set_title(f'{factor1} × {factor2} Interaction')
    
    plt.colorbar(im, label=response)
    
    # annotate cells
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f'{val:.3f}', ha='center', va='center', fontsize=8)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"saved: {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_lr_response(df: pd.DataFrame, response: str = 'test_f1',
                     output_path: str = None) -> None:
    """scatter plot of learning rate vs response."""
    plt = setup_matplotlib()
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for ax, seg in zip(axes, sorted(df['segment_length'].unique())):
        subset = df[df['segment_length'] == seg]
        
        ax.scatter(subset['log_lr'], subset[response], alpha=0.6, s=50)
        ax.set_xlabel('log10(learning rate)')
        ax.set_ylabel(response)
        ax.set_title(f'Segment Length = {seg}s')
        
        # trend line
        if len(subset) > 2:
            z = np.polyfit(subset['log_lr'].dropna(), 
                          subset.loc[subset['log_lr'].notna(), response], 1)
            p = np.poly1d(z)
            x_range = np.linspace(subset['log_lr'].min(), subset['log_lr'].max(), 100)
            ax.plot(x_range, p(x_range), 'r--', alpha=0.5)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"saved: {output_path}")
    else:
        plt.show()
    
    plt.close()


def generate_all_figures(df: pd.DataFrame, output_dir: str = 'figures',
                         response: str = 'test_f1') -> None:
    """generate all standard figures."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    print("\ngenerating figures...")
    
    # main effects
    effects_df = analyze_main_effects(df, response)
    plot_main_effects(effects_df, output_path / 'main_effects.png')
    
    # model boxplot
    plot_model_boxplot(df, response, output_path / 'model_boxplot.png')
    
    # interactions
    if 'segment_length' in df.columns and 'model' in df.columns:
        plot_interaction_heatmap(df, 'model', 'segment_length', response,
                                 output_path / 'interaction_model_segment.png')
    
    # lr response
    if 'log_lr' in df.columns:
        plot_lr_response(df, response, output_path / 'lr_response.png')
    
    print(f"\nall figures saved to {output_dir}/")


# =============================================================================
# report generation
# =============================================================================

def generate_text_report(df: pd.DataFrame, response: str = 'test_f1',
                         output_path: str = None) -> str:
    """generate comprehensive text report."""
    lines = []
    
    lines.append("="*70)
    lines.append("ECG ARRHYTHMIA CLASSIFICATION - DOE ANALYSIS REPORT")
    lines.append("="*70)
    
    # data summary
    summary = summarize_data(df)
    lines.append("\n1. DATA SUMMARY")
    lines.append("-"*40)
    lines.append(f"total runs: {summary['n_runs']}")
    lines.append(f"unique models: {summary['n_models']}")
    lines.append(f"unique seeds: {summary['n_seeds']}")
    
    if 'test_f1_range' in summary:
        lo, hi = summary['test_f1_range']
        lines.append(f"{response} range: [{lo:.4f}, {hi:.4f}]")
    
    # main effects
    lines.append("\n2. MAIN EFFECTS")
    lines.append("-"*40)
    effects_df = analyze_main_effects(df, response)
    for _, row in effects_df.iterrows():
        sig = '***' if row['p_value'] < 0.001 else '**' if row['p_value'] < 0.01 else '*' if row['p_value'] < 0.05 else ''
        lines.append(f"  {row['factor']:20s} η²={row['effect_size']:.4f} p={row['p_value']:.4f} {sig}")
    
    # model leaderboard
    lines.append("\n3. MODEL LEADERBOARD")
    lines.append("-"*40)
    model_stats = df.groupby('model')[response].agg(['mean', 'std', 'count']).round(4)
    model_stats = model_stats.sort_values('mean', ascending=False)
    for model, row in model_stats.iterrows():
        lines.append(f"  {model:25s} mean={row['mean']:.4f} std={row['std']:.4f} n={int(row['count'])}")
    
    # variance decomposition
    lines.append("\n4. VARIANCE DECOMPOSITION")
    lines.append("-"*40)
    var_comp = compute_variance_components(df, response)
    lines.append(f"  model architecture: {var_comp['model_pct']:.1f}%")
    lines.append(f"  segment length:     {var_comp['segment_pct']:.1f}%")
    lines.append(f"  residual (seed+hp): {var_comp['residual_pct']:.1f}%")
    
    # optimization
    lines.append("\n5. GREEDY OPTIMIZATION")
    lines.append("-"*40)
    greedy = optimize_greedy(df, response, verbose=False)
    
    lines.append("optimization path:")
    for step in greedy['path']:
        lines.append(f"  step {step['step']}: {step['factor']} = {step['best_level']} (η²={step['eta_sq']:.4f})")
    
    lines.append("\nfinal configuration:")
    for k, v in greedy['final_config'].items():
        if isinstance(v, float):
            lines.append(f"  {k}: {v:.2e}" if v < 0.01 else f"  {k}: {v:.4f}")
        else:
            lines.append(f"  {k}: {v}")
    
    lines.append(f"\nexpected {response}: {greedy['predicted_f1']:.4f} +/- {greedy['std_f1']:.4f}")
    
    # top configs
    lines.append("\n6. TOP CONFIGURATIONS")
    lines.append("-"*40)
    top = find_top_configs(df, 5, response)
    for i, (_, row) in enumerate(top.iterrows(), 1):
        lines.append(f"  {i}. {row['model']} seg={row['segment_length']} {response}={row[response]:.4f}")
    
    # robust configs
    lines.append("\n7. ROBUST CONFIGURATIONS")
    lines.append("-"*40)
    robust = find_robust_configs(df, min_seeds=2, response=response)
    for _, row in robust.head(5).iterrows():
        lines.append(f"  {row['model']:20s} seg={row['segment_length']:2d} mean={row['mean_f1']:.4f} std={row['std_f1']:.4f}")
    
    lines.append("\n" + "="*70)
    lines.append("END OF REPORT")
    lines.append("="*70)
    
    report = '\n'.join(lines)
    
    if output_path:
        with open(output_path, 'w') as f:
            f.write(report)
        print(f"report saved to: {output_path}")
    
    return report


# =============================================================================
# interpretation (answering DOE questions)
# =============================================================================

def interpret_results(df: pd.DataFrame, response: str = 'test_f1') -> Dict:
    """
    answer key DOE questions.
    
    returns dict with answers to:
    - best architecture
    - optimal segment length
    - differential lr recommendation
    - lr range recommendation
    - cnn frozen recommendation
    - seed variance
    - notable interactions
    """
    results = {}
    
    # Q1: best architecture
    model_stats = df.groupby('model')[response].agg(['mean', 'std', 'count'])
    best_model = model_stats['mean'].idxmax()
    second_model = model_stats['mean'].nlargest(2).index[1]
    
    best_data = df[df['model'] == best_model][response]
    second_data = df[df['model'] == second_model][response]
    ttest = compute_welch_ttest(best_data.values, second_data.values)
    
    results['best_architecture'] = {
        'best': best_model,
        'mean': model_stats.loc[best_model, 'mean'],
        'second': second_model,
        'significant': ttest['p_value'] < 0.05 if not np.isnan(ttest['p_value']) else False
    }
    
    # Q2: optimal segment length
    seg_stats = df.groupby('segment_length')[response].agg(['mean', 'std', 'count'])
    best_seg = seg_stats['mean'].idxmax()
    results['optimal_segment'] = {
        'best': best_seg,
        'mean': seg_stats.loc[best_seg, 'mean'],
        'stats': seg_stats.to_dict()
    }
    
    # Q3: differential lr
    if 'diff_lr' in df.columns and df['diff_lr'].nunique() > 1:
        diff_true = df[df['diff_lr'] == True][response]
        diff_false = df[df['diff_lr'] == False][response]
        ttest = compute_welch_ttest(diff_true.values, diff_false.values)
        
        results['diff_lr'] = {
            'better': True if diff_true.mean() > diff_false.mean() else False,
            'diff_true_mean': diff_true.mean(),
            'diff_false_mean': diff_false.mean(),
            'significant': ttest['p_value'] < 0.05 if not np.isnan(ttest['p_value']) else False
        }
    
    # Q4: lr range
    if 'log_lr' in df.columns:
        top_10_pct = df.nlargest(int(len(df) * 0.1), response)
        lr_range = (10**top_10_pct['log_lr'].min(), 10**top_10_pct['log_lr'].max())
        results['lr_range'] = {
            'top_10_pct_range': lr_range,
            'recommended': lr_range
        }
    
    # Q5: cnn frozen
    if 'cnn_frozen' in df.columns and df['cnn_frozen'].nunique() > 1:
        frozen_true = df[df['cnn_frozen'] == True][response]
        frozen_false = df[df['cnn_frozen'] == False][response]
        results['cnn_frozen'] = {
            'better': True if frozen_true.mean() > frozen_false.mean() else False,
            'frozen_mean': frozen_true.mean(),
            'unfrozen_mean': frozen_false.mean()
        }
    
    # Q6: seed variance
    var_comp = compute_variance_components(df, response)
    results['variance'] = var_comp
    
    # Q7: interactions
    interactions = analyze_interactions(df, response)
    results['interactions'] = interactions.head(3).to_dict('records') if len(interactions) > 0 else []
    
    return results


def print_interpretation(results: Dict) -> None:
    """print interpretation of DOE results."""
    print("\n" + "="*70)
    print("DOE INTERPRETATION")
    print("="*70)
    
    # Q1
    arch = results.get('best_architecture', {})
    print(f"\nQ1: Best architecture?")
    print(f"    {arch.get('best', 'unknown')} (mean={arch.get('mean', 0):.4f})")
    if arch.get('significant'):
        print(f"    Significantly better than {arch.get('second')}")
    else:
        print(f"    Not significantly different from {arch.get('second')}")
    
    # Q2
    seg = results.get('optimal_segment', {})
    print(f"\nQ2: Optimal segment length?")
    print(f"    {seg.get('best', 'unknown')}s (mean={seg.get('mean', 0):.4f})")
    
    # Q3
    diff = results.get('diff_lr', {})
    if diff:
        print(f"\nQ3: Use differential learning rate?")
        print(f"    {'Yes' if diff.get('better') else 'No'} (diff_lr={diff.get('diff_true_mean', 0):.4f} vs {diff.get('diff_false_mean', 0):.4f})")
    
    # Q4
    lr = results.get('lr_range', {})
    if lr:
        print(f"\nQ4: Recommended LR range?")
        lo, hi = lr.get('recommended', (1e-4, 1e-3))
        print(f"    [{lo:.2e}, {hi:.2e}]")
    
    # Q6
    var = results.get('variance', {})
    print(f"\nQ6: Variance decomposition?")
    print(f"    Model: {var.get('model_pct', 0):.1f}%")
    print(f"    Segment: {var.get('segment_pct', 0):.1f}%")
    print(f"    Residual: {var.get('residual_pct', 0):.1f}%")

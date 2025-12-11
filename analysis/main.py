#!/usr/bin/env python3
"""
main.py - command line interface
================================

unified entry point for all analysis tasks.

usage:
    python main.py analyze --exclude-mps
    python main.py optimize --method rf --exclude-mps
    python main.py visualize --exclude-mps --output-dir figures
    python main.py report --exclude-mps --output report.txt
    python main.py interpret --exclude-mps
"""

import argparse
import sys

# import from project modules
from core import load_all_runs, filter_mps_runs, add_derived_features, remove_near_duplicates
from stats import (
    analyze_main_effects, analyze_interactions, analyze_model_features,
    compute_variance_components, find_top_configs, find_robust_configs
)
from optimizer import optimize
from outputs import (
    print_data_summary, print_main_effects, print_model_leaderboard,
    print_top_configs, print_optimization_result,
    generate_all_figures, generate_text_report,
    interpret_results, print_interpretation
)


def load_data(args) -> 'pd.DataFrame':
    """load and preprocess data based on args."""
    df = load_all_runs(args.checkpoint_dir)
    if df is None:
        sys.exit(1)
    
    if args.exclude_mps:
        df = filter_mps_runs(df)
    
    df = add_derived_features(df)
    
    # remove near-duplicate rows to prevent multicollinearity
    df = remove_near_duplicates(df, verbose=True)
    
    print(f"loaded {len(df)} runs")
    
    return df


def cmd_analyze(args):
    """run full analysis."""
    df = load_data(args)
    
    print_data_summary(df)
    
    # main effects
    effects_df = analyze_main_effects(df, args.response)
    print_main_effects(effects_df)
    
    # model feature analysis
    if args.features:
        print("\n" + "="*70)
        print("model architecture feature analysis")
        print("="*70)
        features_df = analyze_model_features(df, args.response)
        print(features_df.to_string(index=False))
    
    # model leaderboard
    print_model_leaderboard(df, args.response)
    
    # interactions
    print("\n" + "="*70)
    print("interaction analysis")
    print("="*70)
    interactions_df = analyze_interactions(df, args.response)
    for _, row in interactions_df.iterrows():
        print(f"  {row['factor1']:15s} × {row['factor2']:15s} η²={row['eta_sq']:.4f} ({row['strength']})")
    
    # variance decomposition
    print("\n" + "="*70)
    print("variance decomposition")
    print("="*70)
    var_comp = compute_variance_components(df, args.response)
    print(f"  model architecture: {var_comp['model_pct']:.1f}%")
    print(f"  segment length:     {var_comp['segment_pct']:.1f}%")
    print(f"  residual (seed+hp): {var_comp['residual_pct']:.1f}%")
    
    # greedy optimization
    print("\n" + "="*70)
    print("greedy optimization")
    print("="*70)
    result = optimize(df, method='greedy', response=args.response, verbose=True)
    print_optimization_result(result, args.response)
    
    # top configs
    print_top_configs(df, 10, args.response)
    
    # robust configs
    print("\n" + "="*70)
    print("robust configurations")
    print("="*70)
    robust = find_robust_configs(df, min_seeds=2, response=args.response)
    print(robust.head(10).to_string(index=False))
    
    # save csv if requested
    if args.save_csv:
        output_path = args.checkpoint_dir + '/experiment_data.csv'
        df.to_csv(output_path, index=False)
        print(f"\nsaved data to: {output_path}")


def cmd_optimize(args):
    """run optimization."""
    df = load_data(args)
    
    result = optimize(
        df, 
        method=args.method, 
        response=args.response,
        verbose=True,
        n_candidates=args.n_candidates
    )
    
    # for single methods, print the result summary
    if args.method != 'all':
        print_optimization_result(result, args.response)


def cmd_visualize(args):
    """generate visualizations."""
    df = load_data(args)
    generate_all_figures(df, args.output_dir, args.response)


def cmd_report(args):
    """generate text report."""
    df = load_data(args)
    report = generate_text_report(df, args.response, args.output)
    
    if not args.output:
        print(report)


def cmd_interpret(args):
    """interpret DOE results."""
    df = load_data(args)
    results = interpret_results(df, args.response)
    print_interpretation(results)
    
    # also run optimization
    print("\n" + "="*70)
    print("optimization recommendation")
    print("="*70)
    opt_result = optimize(df, method='greedy', response=args.response, verbose=True)
    print_optimization_result(opt_result, args.response)


def main():
    # parent parser with common arguments
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument('--checkpoint-dir', default='checkpoints',
                               help='directory with results json files')
    parent_parser.add_argument('--exclude-mps', action='store_true',
                               help='filter out mps backend runs')
    parent_parser.add_argument('--response', default='test_f1',
                               help='target metric')
    
    # main parser
    parser = argparse.ArgumentParser(
        description='ECG DOE Analysis Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
    python main.py analyze --exclude-mps
    python main.py optimize --method rf --exclude-mps
    python main.py visualize --exclude-mps --output-dir figures
    python main.py report --exclude-mps --output report.txt
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='command')
    
    # analyze command
    analyze_parser = subparsers.add_parser('analyze', help='run full analysis',
                                           parents=[parent_parser])
    analyze_parser.add_argument('--features', action='store_true',
                                help='include model feature analysis')
    analyze_parser.add_argument('--save-csv', action='store_true',
                                help='save processed data to csv')
    
    # optimize command
    optimize_parser = subparsers.add_parser('optimize', help='run optimization',
                                            parents=[parent_parser])
    optimize_parser.add_argument('--method', default='greedy',
                                 choices=['greedy', 'linear', 'rf', 'nn', 'all'],
                                 help='optimization method')
    optimize_parser.add_argument('--n-candidates', type=int, default=10000,
                                 help='candidates for surrogate methods')
    
    # visualize command
    viz_parser = subparsers.add_parser('visualize', help='generate figures',
                                       parents=[parent_parser])
    viz_parser.add_argument('--output-dir', default='figures',
                            help='output directory for figures')
    
    # report command
    report_parser = subparsers.add_parser('report', help='generate text report',
                                          parents=[parent_parser])
    report_parser.add_argument('--output', '-o', default=None,
                               help='output file (default: print to stdout)')
    
    # interpret command
    interpret_parser = subparsers.add_parser('interpret', help='interpret DOE results',
                                             parents=[parent_parser])
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return 1
    
    # dispatch to command handler
    commands = {
        'analyze': cmd_analyze,
        'optimize': cmd_optimize,
        'visualize': cmd_visualize,
        'report': cmd_report,
        'interpret': cmd_interpret,
    }
    
    return commands[args.command](args)


if __name__ == '__main__':
    sys.exit(main() or 0)

"""
Training Visualization Utility
===============================

Plots training curves from saved training history JSON files.

Usage:
    # Plot a single run
    python plot_training.py checkpoints/cnn_only_seed42_history.json
    
    # Compare multiple runs
    python plot_training.py checkpoints/cnn_only_seed42_history.json \
                           checkpoints/cnn_lstm_seed42_history.json \
                           --labels "CNN-only" "CNN-LSTM"
    
    # Save figures instead of displaying
    python plot_training.py checkpoints/*_history.json --save figures/
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse


def load_history(json_path):
    """Load training history from JSON file."""
    with open(json_path, 'r') as f:
        return json.load(f)


def plot_single_run(history, title=None, save_path=None):
    """
    Plot training curves for a single run.
    
    Creates a 2x2 grid:
    - Top left: Loss curves (train vs val)
    - Top right: Accuracy curves (train vs val)  
    - Bottom left: F1 and AUROC over time
    - Bottom right: Per-class metrics (last epoch)
    """
    epochs = range(1, len(history['train_loss']) + 1)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(title or "Training History", fontsize=16, fontweight='bold')
    
    # Loss curves
    ax = axes[0, 0]
    ax.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    ax.plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Loss Curves', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Mark best epoch
    if 'best_epoch' in history:
        best_epoch = history['best_epoch']
        ax.axvline(x=best_epoch, color='g', linestyle='--', alpha=0.5, label=f'Best (Epoch {best_epoch})')
        ax.legend(fontsize=10)
    
    # Accuracy curves
    ax = axes[0, 1]
    ax.plot(epochs, history['train_acc'], 'b-', label='Train Acc', linewidth=2)
    ax.plot(epochs, history['val_acc'], 'r-', label='Val Acc', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Accuracy Curves', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    if 'best_epoch' in history:
        ax.axvline(x=best_epoch, color='g', linestyle='--', alpha=0.5)
    
    # F1 and AUROC
    ax = axes[1, 0]
    ax.plot(epochs, history['val_f1'], 'purple', label='Val F1', linewidth=2)
    ax.plot(epochs, history['val_auroc'], 'orange', label='Val AUROC', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Validation Metrics', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])
    
    if 'best_epoch' in history:
        ax.axvline(x=best_epoch, color='g', linestyle='--', alpha=0.5)
    
    # Per-class metrics (last epoch)
    ax = axes[1, 1]
    class_names = ['Normal', 'AF', 'Other', 'Noisy']
    x = np.arange(len(class_names))
    width = 0.35
    
    # Get last epoch's metrics
    last_sensitivity = history['val_sensitivity'][-1]
    last_specificity = history['val_specificity'][-1]
    
    bars1 = ax.bar(x - width/2, last_sensitivity, width, label='Sensitivity', color='skyblue')
    bars2 = ax.bar(x + width/2, last_specificity, width, label='Specificity', color='lightcoral')
    
    ax.set_xlabel('Class', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Per-Class Metrics (Final Epoch)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, fontsize=10)
    ax.legend(fontsize=10)
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f" Saved figure to: {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_comparison(histories, labels, save_path=None):
    """
    Compare training curves from multiple runs.
    
    Creates a 2x2 grid comparing key metrics across runs.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Model Comparison", fontsize=16, fontweight='bold')
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(histories)))
    
    for idx, (history, label, color) in enumerate(zip(histories, labels, colors)):
        epochs = range(1, len(history['train_loss']) + 1)
        
        # Loss comparison
        ax = axes[0, 0]
        ax.plot(epochs, history['val_loss'], color=color, label=label, linewidth=2)
        
        # Accuracy comparison
        ax = axes[0, 1]
        ax.plot(epochs, history['val_acc'], color=color, label=label, linewidth=2)
        
        # F1 comparison
        ax = axes[1, 0]
        ax.plot(epochs, history['val_f1'], color=color, label=label, linewidth=2)
        
        # AUROC comparison
        ax = axes[1, 1]
        ax.plot(epochs, history['val_auroc'], color=color, label=label, linewidth=2)
    
    # Configure subplots
    titles = ['Validation Loss', 'Validation Accuracy', 'Validation F1', 'Validation AUROC']
    ylabels = ['Loss', 'Accuracy', 'F1 Score', 'AUROC']
    
    for ax, title, ylabel in zip(axes.flat, titles, ylabels):
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    
    # y-axis limits for metrics
    axes[0, 1].set_ylim([0, 1])  # Accuracy
    axes[1, 0].set_ylim([0, 1])  # F1
    axes[1, 1].set_ylim([0, 1])  # AUROC
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f" Saved comparison to: {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_learning_rate(history, title=None, save_path=None):
    """Plot learning rate schedule over epochs."""
    epochs = range(1, len(history['learning_rate']) + 1)
    
    plt.figure(figsize=(10, 4))
    plt.plot(epochs, history['learning_rate'], 'b-', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Learning Rate', fontsize=12)
    plt.title(title or 'Learning Rate Schedule', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f" Saved LR schedule to: {save_path}")
    else:
        plt.show()
    
    plt.close()


def print_summary(history, name="Model"):
    """Print summary statistics from training history."""
    print("\n" + "="*70)
    print(f"{name} - Training Summary")
    print("="*70)
    
    print("\nTraining completed:")
    print(f"  Total epochs: {history['epochs_completed']}")
    print(f"  Best epoch: {history['best_epoch']}")
    print(f"  Early stopping: {'Yes' if history['stopped_early'] else 'No'}")
    
    print(f"\nBest validation metrics (Epoch {history['best_epoch']}):")
    best_idx = history['best_epoch'] - 1
    print(f"  F1:       {history['val_f1'][best_idx]:.4f}")
    print(f"  AUROC:    {history['val_auroc'][best_idx]:.4f}")
    print(f"  Accuracy: {history['val_acc'][best_idx]:.4f}")
    print(f"  Loss:     {history['val_loss'][best_idx]:.4f}")
    
    print(f"\nFinal metrics (Epoch {history['epochs_completed']}):")
    print(f"  F1:       {history['val_f1'][-1]:.4f}")
    print(f"  AUROC:    {history['val_auroc'][-1]:.4f}")
    print(f"  Accuracy: {history['val_acc'][-1]:.4f}")
    print(f"  Loss:     {history['val_loss'][-1]:.4f}")
    
    print("\nPer-class sensitivity (final):")
    class_names = ['Normal', 'AF', 'Other', 'Noisy']
    for name, sens in zip(class_names, history['val_sensitivity'][-1]):
        print(f"  {name}: {sens:.4f}")
    
    print("\nPer-class specificity (final):")
    for name, spec in zip(class_names, history['val_specificity'][-1]):
        print(f"  {name}: {spec:.4f}")


def main():
    parser = argparse.ArgumentParser(
        description='Plot training curves from saved history files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Plot single run
  python plot_training.py checkpoints/cnn_only_seed42_history.json
  
  # Compare multiple runs
  python plot_training.py checkpoints/cnn_only_seed42_history.json \\
                         checkpoints/cnn_lstm_seed42_history.json \\
                         --labels "CNN-only" "CNN-LSTM"
  
  # Save figures
  python plot_training.py checkpoints/*_history.json --save figures/
        """
    )
    parser.add_argument('history_files', nargs='+', help='Training history JSON file(s)')
    parser.add_argument('--labels', nargs='+', help='Labels for comparison plots')
    parser.add_argument('--save', type=str, help='Directory to save figures (default: show plots)')
    parser.add_argument('--compare', action='store_true', help='Create comparison plots')
    parser.add_argument('--lr-schedule', action='store_true', help='Plot learning rate schedule')
    parser.add_argument('--summary', action='store_true', help='Print summary statistics')
    
    args = parser.parse_args()
    
    # load histories
    histories = []
    for hist_file in args.history_files:
        hist_path = Path(hist_file)
        if not hist_path.exists():
            print(f"Warning: {hist_file} not found, skipping...")
            continue
        histories.append(load_history(hist_path))
    
    if not histories:
        print("Error: No valid history files found!")
        return
    
    # Generate labels if not provided
    if args.labels:
        labels = args.labels
    else:
        labels = [Path(f).stem.replace('_history', '') for f in args.history_files]
    
    # Ensure we have enough labels
    if len(labels) < len(histories):
        labels += [f"Model {i+1}" for i in range(len(labels), len(histories))]
    
    # Setup save directory
    save_dir = Path(args.save) if args.save else None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
    
    # Single run plots
    if len(histories) == 1 or not args.compare:
        for history, label, hist_file in zip(histories, labels, args.history_files):
            # Print summary
            if args.summary:
                print_summary(history, name=label)
            
            # Plot training curves
            save_path = save_dir / f"{label}_curves.png" if save_dir else None
            plot_single_run(history, title=f"{label} - Training History", save_path=save_path)
            
            # Plot learning rate schedule
            if args.lr_schedule:
                lr_save_path = save_dir / f"{label}_lr_schedule.png" if save_dir else None
                plot_learning_rate(history, title=f"{label} - Learning Rate", save_path=lr_save_path)
    
    # Comparison plots
    if len(histories) > 1 and (args.compare or len(args.history_files) > 1):
        save_path = save_dir / "model_comparison.png" if save_dir else None
        plot_comparison(histories, labels, save_path=save_path)
        
        # Print comparison summary
        if args.summary:
            print("\n" + "="*70)
            print("COMPARISON SUMMARY")
            print("="*70)
            print(f"\n{'Model':<20} {'Best F1':<10} {'Best AUROC':<10} {'Final F1':<10} {'Final AUROC':<10}")
            print("-"*70)
            for history, label in zip(histories, labels):
                best_idx = history['best_epoch'] - 1
                print(f"{label:<20} {history['val_f1'][best_idx]:<10.4f} "
                      f"{history['val_auroc'][best_idx]:<10.4f} "
                      f"{history['val_f1'][-1]:<10.4f} "
                      f"{history['val_auroc'][-1]:<10.4f}")
    
    if not save_dir:
        print("\nNote: Use --save <directory> to save figures instead of displaying them")


if __name__ == "__main__":
    main()
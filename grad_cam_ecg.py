#!/usr/bin/env python3
"""
Grad-CAM Visualization for ECG Arrhythmia Classification
=========================================================

Visualizes important regions in ECG signals using Gradient-weighted Class
Activation Mapping (Grad-CAM) for the CNN-based models in this project.

Grad-CAM produces a heatmap showing which temporal regions of the ECG signal
contributed most to the model's classification decision. This is particularly
valuable in medical AI for understanding whether the model focuses on
clinically meaningful features (P-waves, QRS complexes, RR intervals, etc.).

References:
    - Selvaraju et al. (2017) "Grad-CAM: Visual Explanations from Deep Networks"

Usage:
    # visualize class comparison with default settings
    python grad_cam_ecg.py --checkpoint checkpoints/cnn_only_seed42_*.pt
    python grad_cam_ecg.py --checkpoint checkpoints\cnn_lstm_seed1_20251208_011346.pt --output "figures\gradcam_cnn_lstm"
    python grad_cam_ecg.py --checkpoint checkpoints\cnn_lstm_ln_seed42_20251208_034724.pt --output "figures\gradcam_cnn_lstm_ln"
    
Authors: CS541 Deep Learning Team
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import json
from typing import Optional, Tuple, List, Dict

# from codebase
from dataset import load_splits
from train import get_model, MODEL_REGISTRY


# class labels for the PhysioNet 2017 dataset
CLASS_NAMES = ['Normal', 'AF', 'Other', 'Noisy']
CLASS_COLORS = ['#2ecc71', '#e74c3c', '#f39c12', '#9b59b6']


def inspect_checkpoint(checkpoint_path: str) -> dict:
    """
    Inspect a checkpoint file and return information about its format.
    
    Useful for debugging and understanding checkpoint structure.
    
    Args:
        checkpoint_path: Path to .pt checkpoint file
    
    Returns:
        Dict with checkpoint information
    """
    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    info = {
        'path': str(checkpoint_path),
        'type': type(checkpoint).__name__,
        'format': 'unknown'
    }
    
    if isinstance(checkpoint, dict):
        info['keys'] = list(checkpoint.keys())
        
        # the only model that is a CNN without an RNN component is 'cnn_only'
        # for all others, if the name contains CNN, it is a hybrid CNN-RNN model
        if 'model_state_dict' in checkpoint:
            info['format'] = 'new_format_with_metadata'
            info['has_config'] = 'config' in checkpoint
            info['has_optimizer'] = 'optimizer_state_dict' in checkpoint
            info['has_scheduler'] = 'scheduler_state_dict' in checkpoint
            if 'config' in checkpoint:
                config = checkpoint['config']
                info['model_name'] = config.get('model_name', 'unknown')
                info['model_class'] = config.get('model_class', 'unknown')
        elif any(k.startswith(('cnn.', 'features.', 'lstm.', 'fc.', 'classifier.')) 
                 for k in checkpoint.keys()):
            info['format'] = 'state_dict_only'
            info['num_parameters'] = len(checkpoint)
            # try to infer model type from layer names
            if any('lstm' in k.lower() for k in checkpoint.keys()):
                info['likely_model_type'] = 'CNN-LSTM variant'
            elif any('gru' in k.lower() for k in checkpoint.keys()):
                info['likely_model_type'] = 'CNN-GRU variant'
            else:
                info['likely_model_type'] = 'CNN-only'
        else:
            info['format'] = 'unknown_dict'
    else:
        info['format'] = 'non_dict'
    
    return info


class GradCAM1D:
    """
    Grad-CAM implementation for 1D CNNs (ECG signals).
    
    Computes gradient-weighted class activation maps for 1D convolutional
    neural networks. The implementation hooks into the last convolutional
    layer to capture activations and gradients during forward/backward passes.
    
    Attributes:
        model: PyTorch model (must have CNN layers)
        target_layer: The convolutional layer to use for Grad-CAM
        activations: Stored activations from forward pass
        gradients: Stored gradients from backward pass
    """
    
    def __init__(self, model: nn.Module, target_layer: Optional[nn.Module] = None):
        """
        Initialize Grad-CAM.
        
        Args:
            model: Trained PyTorch model with CNN layers
            target_layer: Specific layer to use. If None, automatically finds
                         the last Conv1d layer in the model.
        """
        self.model = model
        self.model.eval()
        
        # find target layer if not specified
        if target_layer is None:
            self.target_layer = self._find_last_conv_layer(model)
        else:
            self.target_layer = target_layer
        
        if self.target_layer is None:
            raise ValueError("Could not find a Conv1d layer in the model. "
                           "Please specify target_layer manually.")
        
        # storage for activations and gradients
        self.activations = None
        self.gradients = None
        
        # register hooks
        self._register_hooks()
    
    def _find_last_conv_layer(self, model: nn.Module) -> Optional[nn.Module]:
        """
        Find the last Conv1d layer in the model.
        
        Searches through all modules recursively to find the deepest
        Conv1d layer, which typically captures the most abstract features.
        """
        last_conv = None
        
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv1d):
                last_conv = module
        
        return last_conv
    
    def _register_hooks(self):
        """Register forward and backward hooks on the target layer."""
        
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)
    
    def __call__(self, x: torch.Tensor, target_class: Optional[int] = None) -> Tuple[np.ndarray, int, float]:
        """
        Compute Grad-CAM heatmap for input.
        
        Args:
            x: Input tensor of shape (1, seq_len) or (seq_len,)
            target_class: Class to compute Grad-CAM for. If None, uses
                         the predicted class.
        
        Returns:
            heatmap: Grad-CAM heatmap of shape (seq_len,), values in [0, 1]
            predicted_class: Model's predicted class
            confidence: Model's confidence for the target class
        """
        # ensure proper input shape
        if x.dim() == 1:
            x = x.unsqueeze(0)  # add batch dimension
        
        x = x.to(next(self.model.parameters()).device)
        x.requires_grad_(True)
        
        # temporarily switch to train mode for backward pass BUG FIX
        # this is required because cudnn RNN backward only works in training mode
        # note: this means dropout will be active, but for visualization purposes
        # this is acceptable since we're looking at attention patterns, not exact outputs
        was_training = self.model.training
        self.model.train()
        
        # forward pass
        output = self.model(x)
        probs = F.softmax(output, dim=1)
        predicted_class = output.argmax(dim=1).item()
        
        # use predicted class if target not specified
        if target_class is None:
            target_class = predicted_class
        
        confidence = probs[0, target_class].item()
        
        # backward pass for target class
        self.model.zero_grad()
        target_score = output[0, target_class]
        target_score.backward()
        
        # restore original mode
        if not was_training:
            self.model.eval()
        
        # compute grad-cam
        # global average pooling of gradients -> weights
        weights = self.gradients.mean(dim=2, keepdim=True)  # (1, C, 1)
        
        # weighted combination of activation maps
        cam = (weights * self.activations).sum(dim=1)  # (1, L)
        
        # relu to keep only positive contributions
        cam = F.relu(cam)
        
        # normalize to [0, 1]
        cam = cam.squeeze().cpu().numpy()
        if cam.max() > 0:
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        
        # upsample to match input sequence length
        seq_len = x.shape[-1]
        heatmap = np.interp(
            np.linspace(0, len(cam) - 1, seq_len),
            np.arange(len(cam)),
            cam
        )
        
        return heatmap, predicted_class, confidence


def find_target_layer_by_model(model: nn.Module, model_name: str) -> nn.Module:
    """
    Find the appropriate target layer for Grad-CAM based on model architecture.
    
    Different model architectures in the project have different structures,
    so we need to handle each appropriately.
    
    Args:
        model: The loaded model
        model_name: Name from MODEL_REGISTRY (e.g., 'cnn_only', 'cnn_lstm')
    
    Returns:
        The target layer for Grad-CAM
    """
    # try common patterns in the codebase
    
    # pattern 1: model.features (SimpleCNN, model_cnn_only.py)
    if hasattr(model, 'features'):
        for module in reversed(list(model.features.modules())):
            if isinstance(module, nn.Conv1d):
                return module
    
    # pattern 2: model.cnn (CNNLSTM variants)
    if hasattr(model, 'cnn'):
        for module in reversed(list(model.cnn.modules())):
            if isinstance(module, nn.Conv1d):
                return module
    
    # pattern 3: search all modules
    last_conv = None
    for module in model.modules():
        if isinstance(module, nn.Conv1d):
            last_conv = module
    
    if last_conv is not None:
        return last_conv
    
    raise ValueError(f"Could not find Conv1d layer in model {model_name}")


def load_model_from_checkpoint(checkpoint_path: str, model_name: Optional[str] = None) -> Tuple[nn.Module, str, dict]:
    """
    Load a model from a checkpoint file.
    
    Supports two checkpoint formats:
    1. Old format: Just the model state_dict
    2. New format: Dictionary with 'model_state_dict', 'config', 'optimizer_state_dict', etc.
    
    Attempts to infer the model name from the checkpoint filename or from
    the checkpoint's embedded config if not provided.
    
    Args:
        checkpoint_path: Path to .pt checkpoint file
        model_name: Model name (from MODEL_REGISTRY). If None, inferred from 
                   checkpoint config or filename.
    
    Returns:
        model: Loaded PyTorch model
        model_name: Name of the model
        metadata: Associated metadata (config, results, etc.)
    """
    checkpoint_path = Path(checkpoint_path)
    
    # load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # detect checkpoint format and extract state_dict and metadata
    metadata = {}
    state_dict = None
    
    if isinstance(checkpoint, dict):
        # new format: dictionary with model_state_dict and config
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            
            # extract config/metadata from checkpoint
            if 'config' in checkpoint:
                metadata['config'] = checkpoint['config']
                # try to get model name from config
                if model_name is None and 'model_name' in checkpoint['config']:
                    model_name = checkpoint['config']['model_name']
            
            # store other checkpoint info
            for key in ['epoch', 'best_val_f1', 'optimizer_state_dict', 'scheduler_state_dict']:
                if key in checkpoint:
                    metadata[key] = checkpoint[key]
        
        # could also be old format where state_dict is the dict itself
        # (contains layer names like 'cnn.0.weight', etc.)
        elif any(k.startswith(('cnn.', 'features.', 'lstm.', 'fc.', 'classifier.')) 
                 for k in checkpoint.keys()):
            state_dict = checkpoint
        else:
            # unknown dict format - assume it's a state_dict
            state_dict = checkpoint
    else:
        # very old format or unexpected - try to use as-is
        state_dict = checkpoint
    
    # try to infer model name from filename if still not determined
    if model_name is None:
        # filename format: {model_name}_seed{seed}_{timestamp}.pt
        name_parts = checkpoint_path.stem.split('_seed')
        if len(name_parts) >= 1:
            model_name = name_parts[0]
        else:
            raise ValueError(f"Cannot infer model name from {checkpoint_path}. "
                           f"Please specify --model explicitly.")
    
    # validate model name
    if model_name not in MODEL_REGISTRY:
        available = ', '.join(MODEL_REGISTRY.keys())
        raise ValueError(f"Unknown model: {model_name}. Available: {available}")
    
    # create and load model
    model = get_model(model_name)
    model.load_state_dict(state_dict)
    model.eval()
    
    # try to load associated metadata from _results.json file
    results_path = checkpoint_path.parent / f"{checkpoint_path.stem}_results.json"
    if results_path.exists():
        with open(results_path) as f:
            results_metadata = json.load(f)
            # merge with checkpoint metadata (file metadata takes precedence for conflicts)
            metadata.update(results_metadata)
    
    return model, model_name, metadata


def plot_class_comparison(
    class_samples: Dict[int, List[Dict]],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (18, 10),
    fs: int = 300
) -> plt.Figure:
    """
    Create a comparison plot showing Grad-CAM for each class.
    
    Shows one representative sample per class to compare where the model
    focuses for different arrhythmia types.
    
    Args:
        class_samples: Dict mapping class_id -> list of sample dicts
        save_path: Path to save figure
        figsize: Figure size
        fs: Sampling frequency
    
    Returns:
        matplotlib Figure object
    """
    n_classes = len(class_samples)
    
    fig, axes = plt.subplots(n_classes, 1, figsize=figsize, sharex=True)
    fig.suptitle('Grad-CAM Class Comparison: Where Does the Model Focus?', 
                fontsize=14, fontweight='bold')
    
    if n_classes == 1:
        axes = [axes]
    
    for class_id, samples in class_samples.items():
        ax = axes[class_id]
        sample = samples[0]  # take first sample
        
        signal = sample['signal']
        heatmap = sample['heatmap']
        time = np.arange(len(signal)) / fs
        
        # plot ecg
        ax.plot(time, signal, 'k-', linewidth=0.8, alpha=0.9, zorder=2)
        
        # highlight important regions
        threshold = 0.5
        important = heatmap > threshold
        if np.any(important):
            ax.fill_between(time, signal.min() - 0.2, signal.max() + 0.2,
                           where=important, alpha=0.3, 
                           color=CLASS_COLORS[class_id], zorder=1)
        
        # add heatmap as background
        extent = [time[0], time[-1], signal.min() - 0.2, signal.max() + 0.2]
        ax.imshow(heatmap.reshape(1, -1), aspect='auto', extent=extent,
                 cmap='Reds', alpha=0.4, vmin=0, vmax=1, zorder=0)
        
        # formatting
        pred_correct = sample['true_label'] == sample['predicted_label']
        status = '(Correct)' if pred_correct else '(Misclassified)'
        
        ax.set_ylabel(f'{CLASS_NAMES[class_id]}\n{status}', 
                     fontsize=11, color=CLASS_COLORS[class_id], fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, time[-1]])
    
    axes[-1].set_xlabel('Time (seconds)', fontsize=11)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"Saved: {save_path}")
    
    return fig


def analyze_attention_statistics(
    samples: List[Dict],
    fs: int = 300
) -> Dict:
    """
    Compute statistics about where the model focuses attention.
    
    Useful for understanding model behavior across many samples.
    
    Args:
        samples: List of sample dicts with heatmaps
        fs: Sampling frequency
    
    Returns:
        Dict with attention statistics
    """
    stats = {
        'mean_peak_position': [],
        'attention_spread': [],
        'max_attention': [],
        'attention_above_threshold': []
    }
    
    threshold = 0.5
    
    for sample in samples:
        heatmap = sample['heatmap']
        time = np.arange(len(heatmap)) / fs
        
        # peak position (time of maximum attention)
        peak_idx = np.argmax(heatmap)
        stats['mean_peak_position'].append(time[peak_idx])
        
        # spread (std of attention distribution)
        attention_weighted_time = np.average(time, weights=heatmap + 1e-8)
        attention_weighted_var = np.average((time - attention_weighted_time)**2, 
                                            weights=heatmap + 1e-8)
        stats['attention_spread'].append(np.sqrt(attention_weighted_var))
        
        # max attention
        stats['max_attention'].append(np.max(heatmap))
        
        # fraction above threshold
        stats['attention_above_threshold'].append(np.mean(heatmap > threshold))
    
    # compute summary statistics
    summary = {}
    for key, values in stats.items():
        summary[key] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values)
        }
    
    return summary


def run_gradcam_analysis(
    checkpoint_path: str,
    model_name: Optional[str] = None,
    split_dir: str = 'data/splits',
    num_samples: int = 5,
    sample_indices: Optional[List[int]] = None,
    target_class: Optional[int] = None,
    output_dir: str = 'figures/gradcam',
    dataset_split: str = 'test'
) -> Dict:
    """
    Run complete Grad-CAM analysis on a model.
    
    Main entry point for Grad-CAM visualization. Loads model, processes
    samples, and generates the class comparison visualization.
    
    Args:
        checkpoint_path: Path to model checkpoint
        model_name: Model name (inferred from checkpoint if None)
        split_dir: Directory with train/val/test splits
        num_samples: Number of samples to analyze per class
        sample_indices: Specific sample indices to analyze (overrides num_samples)
        target_class: Specific class to compute Grad-CAM for
        output_dir: Directory to save visualizations
        dataset_split: Which split to use ('train', 'val', 'test')
    
    Returns:
        Dict with analysis results and statistics
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # load model
    print(f"\n{'='*60}")
    print("Grad-CAM Analysis for ECG Arrhythmia Classification")
    print(f"{'='*60}")
    
    print(f"\nLoading model from: {checkpoint_path}")
    model, model_name, metadata = load_model_from_checkpoint(checkpoint_path, model_name)
    
    # get device
    device = torch.device('cuda' if torch.cuda.is_available() else 
                         'mps' if torch.backends.mps.is_available() else 'cpu')
    model = model.to(device)
    print(f"Model: {model_name}")
    print(f"Device: {device}")
    
    # load data
    print(f"\nLoading data from: {split_dir}")
    X_train, X_val, X_test, y_train, y_val, y_test, _ = load_splits(split_dir)
    
    # select appropriate split
    split_data = {
        'train': (X_train, y_train),
        'val': (X_val, y_val),
        'test': (X_test, y_test)
    }
    X, y = split_data[dataset_split]
    print(f"Using {dataset_split} split: {len(X)} samples")
    
    # find target layer
    target_layer = find_target_layer_by_model(model, model_name)
    print(f"Target layer: {target_layer}")
    
    # initialize grad-cam
    gradcam = GradCAM1D(model, target_layer)
    print(f"Method: Grad-CAM")
    
    # select samples
    if sample_indices is not None:
        indices = sample_indices
    else:
        # get samples from each class
        indices = []
        for class_id in range(len(CLASS_NAMES)):
            class_indices = np.where(y == class_id)[0]
            if len(class_indices) > 0:
                selected = np.random.choice(class_indices, 
                                           min(num_samples, len(class_indices)),
                                           replace=False)
                indices.extend(selected)
    
    print(f"\nAnalyzing {len(indices)} samples...")
    
    # process samples
    all_samples = []
    class_samples = {i: [] for i in range(len(CLASS_NAMES))}
    
    for idx in indices:
        signal = X[idx]
        true_label = y[idx]
        
        # compute grad-cam
        x_tensor = torch.FloatTensor(signal)
        heatmap, predicted, confidence = gradcam(x_tensor, target_class)
        
        sample_dict = {
            'signal': signal,
            'heatmap': heatmap,
            'true_label': int(true_label),
            'predicted_label': predicted,
            'confidence': confidence,
            'sample_idx': idx
        }
        
        all_samples.append(sample_dict)
        class_samples[int(true_label)].append(sample_dict)
    
    # generate class comparison visualization
    print("\nGenerating class comparison visualization...")
    
    if any(len(v) > 0 for v in class_samples.values()):
        # filter to classes with samples
        class_samples_filtered = {k: v for k, v in class_samples.items() if len(v) > 0}
        comparison_path = output_dir / "gradcam_class_comparison.png"
        plot_class_comparison(class_samples_filtered, save_path=str(comparison_path))
        plt.close()
    
    # compute statistics
    stats = analyze_attention_statistics(all_samples)
    
    # save statistics
    stats_path = output_dir / "gradcam_statistics.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"\nStatistics saved to: {stats_path}")
    
    # print summary
    print(f"\n{'='*60}")
    print("Analysis Summary")
    print(f"{'='*60}")
    print(f"Model: {model_name}")
    print(f"Method: Grad-CAM")
    print(f"Samples analyzed: {len(all_samples)}")
    
    correct = sum(1 for s in all_samples if s['true_label'] == s['predicted_label'])
    print(f"Correct predictions: {correct}/{len(all_samples)} ({100*correct/len(all_samples):.1f}%)")
    
    print(f"\nAttention Statistics:")
    print(f"  Mean peak position: {stats['mean_peak_position']['mean']:.2f}s "
          f"(±{stats['mean_peak_position']['std']:.2f}s)")
    print(f"  Mean attention spread: {stats['attention_spread']['mean']:.2f}s")
    print(f"  Mean fraction above 50%: {stats['attention_above_threshold']['mean']:.1%}")
    
    print(f"\nVisualizations saved to: {output_dir}")
    
    return {
        'model_name': model_name,
        'method': 'Grad-CAM',
        'samples': all_samples,
        'statistics': stats,
        'output_dir': str(output_dir)
    }


def main():
    """Main entry point for command-line usage."""
    parser = argparse.ArgumentParser(
        description='Grad-CAM Visualization for ECG Classification',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # basic usage with automatic model detection
    python grad_cam_ecg.py --checkpoint checkpoints/cnn_only_seed42_*.pt
    
    # specify model explicitly
    python grad_cam_ecg.py --checkpoint model.pt --model cnn_only
    
    # analyze specific samples
    python grad_cam_ecg.py --checkpoint model.pt --sample_indices 0 10 20 30
    
    # compute grad-cam for specific class
    python grad_cam_ecg.py --checkpoint model.pt --target_class 1  # AF class
    
    # inspect checkpoint format (useful for debugging)
    python grad_cam_ecg.py --checkpoint model.pt --inspect
        """
    )
    
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint (.pt file)')
    parser.add_argument('--model', type=str, default=None,
                       help='Model name from registry (auto-detected if not specified)')
    parser.add_argument('--split_dir', type=str, default='data/splits',
                       help='Directory containing train/val/test splits')
    parser.add_argument('--num_samples', type=int, default=5,
                       help='Number of samples to analyze per class')
    parser.add_argument('--sample_indices', type=int, nargs='+', default=None,
                       help='Specific sample indices to analyze')
    parser.add_argument('--target_class', type=int, default=None,
                       choices=[0, 1, 2, 3],
                       help='Target class for Grad-CAM (0=Normal, 1=AF, 2=Other, 3=Noisy)')
    parser.add_argument('--output', type=str, default='figures/gradcam',
                       help='Output directory for visualizations')
    parser.add_argument('--split', type=str, default='test',
                       choices=['train', 'val', 'test'],
                       help='Which data split to use')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for sample selection')
    parser.add_argument('--inspect', action='store_true',
                       help='Inspect checkpoint format and exit (useful for debugging)')
    
    args = parser.parse_args()
    
    # handle inspect mode
    if args.inspect:
        print("\n" + "="*60)
        print("Checkpoint Inspection")
        print("="*60)
        info = inspect_checkpoint(args.checkpoint)
        for key, value in info.items():
            print(f"  {key}: {value}")
        print("="*60 + "\n")
        return
    
    # set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # run analysis
    run_gradcam_analysis(
        checkpoint_path=args.checkpoint,
        model_name=args.model,
        split_dir=args.split_dir,
        num_samples=args.num_samples,
        sample_indices=args.sample_indices,
        target_class=args.target_class,
        output_dir=args.output,
        dataset_split=args.split
    )


if __name__ == '__main__':
    main()
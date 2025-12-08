import torch
import torch.nn as nn
import numpy as np
import random
import json
import configparser
import getpass
import platform
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix
from pathlib import Path
from datetime import datetime

from dataset import get_dataloaders


def load_model_registry(path=None):
    config = configparser.ConfigParser()
    file = Path(__file__).with_name("models.ini") if path is None else Path(path)
    config.read(file)

    registry = {}
    for section in config.sections():
        registry[section] = (
            config[section]["module"],
            config[section]["class"]
        )
    return registry

MODEL_REGISTRY = load_model_registry()


def get_model(model_name):
    """Dynamically import and instantiate a model by name."""
    if model_name not in MODEL_REGISTRY:
        available = ', '.join(MODEL_REGISTRY.keys())
        raise ValueError(f"Unknown model: {model_name}. Available: {available}")
    
    module_name, class_name = MODEL_REGISTRY[model_name]
    module = __import__(module_name)
    model_class = getattr(module, class_name)
    return model_class()


def set_seed(seed):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_best_device():
    """Get best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def calculate_metrics(y_true, y_pred, y_prob):
    """Calculate F1, AUROC, sensitivity, specificity."""
    # Overall F1
    f1 = f1_score(y_true, y_pred, average='weighted')
    
    # AUROC (one-vs-rest)
    try:
        auroc = roc_auc_score(y_true, y_prob, multi_class='ovr', average='weighted')
    except:
        auroc = 0.0
    
    # Per-class sensitivity and specificity
    cm = confusion_matrix(y_true, y_pred)
    sensitivity = np.diag(cm) / cm.sum(axis=1)
    specificity = []
    for i in range(len(cm)):
        tn = cm.sum() - (cm[i, :].sum() + cm[:, i].sum() - cm[i, i])
        fp = cm[:, i].sum() - cm[i, i]
        specificity.append(tn / (tn + fp) if (tn + fp) > 0 else 0.0)
    
    return {
        'f1': f1,
        'auroc': auroc,
        'sensitivity': sensitivity,
        'specificity': specificity
    }


def calculate_accuracy(y_true, y_pred):
    """Calculate overall accuracy."""
    return np.mean(np.array(y_true) == np.array(y_pred))


def train_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch and return loss + accuracy."""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients (DOES NOT HELP WITH VANISHING GRADIENTS)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        
        # track preds for accuracy
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.cpu().numpy())
    
    avg_loss = total_loss / len(loader)
    accuracy = calculate_accuracy(all_labels, all_preds)
    
    return avg_loss, accuracy


def evaluate(model, loader, criterion, device):
    """Evaluate model and return loss + metrics."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            
            outputs = model(X)
            loss = criterion(outputs, y)
            
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)
            
            total_loss += loss.item()
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    
    avg_loss = total_loss / len(loader)
    metrics = calculate_metrics(all_labels, all_preds, all_probs)
    metrics['accuracy'] = calculate_accuracy(all_labels, all_preds)
    
    return avg_loss, metrics


def train_model(model, train_loader, val_loader, criterion, optimizer, 
                device, num_epochs=50, patience=10, save_path="checkpoints/best_model.pt",
                log_path=None, config=None):
    """Train model with early stopping and logging.
    
    Args:
        model: PyTorch model
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        num_epochs: Max epochs
        patience: Early stopping patience
        save_path: Path for best model weights
        log_path: Path for training log (auto if None)
        config: Dict of all hyperparams/settings (auto-saved)
    
    Returns:
        model: Trained model
        history: Training history dict
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Auto log path if not specified
    if log_path is None:
        log_path = save_path.parent / f"{save_path.stem}_history.json"
    else:
        log_path = Path(log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # LR scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )
    
    # Training history w/ complete config
    history = {
        # Config (all hyperparams for searching optimal settings)
        'config': config or {},
        
        # Per-epoch metrics
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'val_f1': [],
        'val_auroc': [],
        'val_sensitivity': [],
        'val_specificity': [],
        'learning_rate': [],
        'learning_rate_dict': [], 
        
        # Training metadata
        'epochs_completed': 0,
        'best_epoch': 0,
        'best_val_f1': 0.0,
        'stopped_early': False,
        'start_time': datetime.now().isoformat(),
        'end_time': None
    }
    
    best_val_f1 = 0
    epochs_no_improve = 0
    
    print(f"\nTraining with logging to: {log_path}")
    
    for epoch in range(num_epochs):
        # Training
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validation
        val_loss, val_metrics = evaluate(model, val_loader, criterion, device)
        
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        
        # Log history
        history['train_loss'].append(float(train_loss))
        history['train_acc'].append(float(train_acc))
        history['val_loss'].append(float(val_loss))
        history['val_acc'].append(float(val_metrics['accuracy']))
        history['val_f1'].append(float(val_metrics['f1']))
        history['val_auroc'].append(float(val_metrics['auroc']))
        history['val_sensitivity'].append([float(x) for x in val_metrics['sensitivity']])
        history['val_specificity'].append([float(x) for x in val_metrics['specificity']])
        history['learning_rate'].append(float(current_lr))
        history['epochs_completed'] = epoch + 1
        
        # Log all LRs if multiple param groups
        if len(optimizer.param_groups) > 1:
            lr_dict = {}
            for i, group in enumerate(optimizer.param_groups):
                name = group.get('name', f'group_{i}')
                lr_dict[name] = float(group['lr'])
            history['learning_rate_dict'].append(lr_dict)
        else:
            history['learning_rate_dict'].append(None)
        
        # Print progress
        print(f"Epoch {epoch+1}/{num_epochs} - "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} - "
              f"Val Loss: {val_loss:.4f}, Val F1: {val_metrics['f1']:.4f}, "
              f"Val AUROC: {val_metrics['auroc']:.4f}")
        
        # Update scheduler
        scheduler.step(val_metrics['f1'])
        
        # Early stopping and checkpoint saving
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            history['best_val_f1'] = float(best_val_f1)
            history['best_epoch'] = epoch + 1
            epochs_no_improve = 0
            torch.save(model.state_dict(), save_path)
            print(f"  Best model saved (F1: {best_val_f1:.4f})")
        else:
            epochs_no_improve += 1
        
        # Step scheduler again
        scheduler.step(val_metrics['f1'])
        
        with open(log_path, 'w') as f:
            json.dump(history, f, indent=2)
        
        if epochs_no_improve >= patience:
            print(f"Early stopping after {epoch+1} epochs: no improvement for {patience} epochs")
            history['stopped_early'] = True
            break
    
    # Record end time
    history['end_time'] = datetime.now().isoformat()
    
    # Save training history
    with open(log_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"\n Training history saved to: {log_path}")
    
    # Load best model
    model.load_state_dict(torch.load(save_path))
    
    return model, history


def main(model_name='cnn_lstm', seed=42, split_dir='data/splits', save_dir=None,
         num_epochs=50, patience=10, lr=1e-4, lstm_lr=None, weight_decay=1e-4, batch_size=64,
         freeze_cnn=False):  # NEW: Add freeze_cnn parameter
    """Train model w/ comprehensive logging of all params.
    
    Automatically generates unique timestamp-based filenames and logs ALL
    hyperparameters for later searching/comparing optimal configurations.
    
    Args:
        model_name: Model arch (cnn_lstm, cnn_only, bilstm, etc.)
        seed: Random seed
        split_dir: Dir w/ train/val/test splits
        save_dir: Dir for checkpoints (default: checkpoints/)
        num_epochs: Max training epochs
        patience: Early stopping patience
        lr: Learning rate
        lstm_lr: If specified, LSTM components use this LR instead of main LR.
            Typically 10-100x smaller than main LR (e.g., lr=1e-4, lstm_lr=1e-5)
        weight_decay: Weight decay (L2 reg)
        batch_size: Training batch size
        freeze_cnn: If True, freeze CNN weights (only train LSTM)
    
    Returns:
        dict: Test metrics + history
    """
    # Generate unique timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Set seeds
    set_seed(seed)
    
    # Setup save directory
    if save_dir is None:
        save_dir = Path("checkpoints")
    else:
        save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # filenames w/ timestamps
    run_id = f"{model_name}_seed{seed}_{timestamp}"
    model_path = save_dir / f"{run_id}.pt"
    log_path = save_dir / f"{run_id}_history.json"
    results_path = save_dir / f"{run_id}_results.json"
    
    # Build comprehensive config (all hyperparams for searching)
    config = {
        # Run identification
        'run_id': run_id,
        'timestamp': timestamp,
        'datetime': datetime.now().isoformat(),
        
        # Model config
        'model_name': model_name,
        'model_file': MODEL_REGISTRY[model_name][0] + '.py',
        'model_class': MODEL_REGISTRY[model_name][1],
        
        # Data config
        'split_dir': str(split_dir),
        'batch_size': batch_size,
        
        # Training config
        'num_epochs': num_epochs,
        'patience': patience,
        'seed': seed,
        
        # Optimizer config
        'optimizer': 'Adam',
        'learning_rate': lr,
        'lstm_learning_rate': lstm_lr,
        'differential_lr': lstm_lr is not None,
        'weight_decay': weight_decay,
        'freeze_cnn': freeze_cnn,
                
        # Scheduler config
        'scheduler': 'ReduceLROnPlateau',
        'scheduler_mode': 'max',
        'scheduler_factor': 0.5,
        'scheduler_patience': 5,
        
        # Regularization
        'gradient_clip_norm': 1.0,
        
        # Paths
        'save_dir': str(save_dir),
        'model_path': str(model_path),
        'log_path': str(log_path),
        'results_path': str(results_path)
    }
    
    # Setup
    device = get_best_device()
    
    # Track device/user for debugging MPS issues
    config['device'] = str(device)
    config['user'] = getpass.getuser()
    config['platform'] = platform.system()
    config['torch_version'] = torch.__version__

    # Print config summary
    print("="*70)
    print(f"RUN: {run_id}")
    print("="*70)
    print(f"Device: {device}")
    print(f"Model: {model_name} ({config['model_file']})")
    print(f"Split: {split_dir}")
    if lstm_lr is not None:
        print(f"LR: CNN={lr}, LSTM={lstm_lr} (differential)")
    else:
        print(f"LR: {lr} (uniform)")
    if freeze_cnn:  # NEW: Print freeze status
        print("CNN: FROZEN (only LSTM trains)")
    print(f"WD: {weight_decay}, BS: {batch_size}")
    print(f"Epochs: {num_epochs}, Patience: {patience}, Seed: {seed}")
    print(f"Saving to: {save_dir}")
    
    # Load data
    train_loader, val_loader, test_loader, class_weights = get_dataloaders(
        batch_size=batch_size, split_dir=split_dir
    )
    print(f"Batches: train={len(train_loader)}, val={len(val_loader)}, test={len(test_loader)}")
    
    # Model
    model = get_model(model_name).to(device)
    
    # NEW: Freeze CNN if requested
    if freeze_cnn and hasattr(model, 'freeze_cnn'):
        model.freeze_cnn()
        print("✓ CNN frozen - only LSTM will train")
    elif freeze_cnn:
        print(f"WARNING: Model {model_name} doesn't support freeze_cnn()")
    
    n_params = sum(p.numel() for p in model.parameters())
    config['model_parameters'] = n_params
    print(f"Parameters: {n_params:,}")
    
    # Loss + optimizer
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    config['criterion'] = 'CrossEntropyLoss'
    config['class_weights'] = class_weights.tolist()
    
    # MODIFIED: Create optimizer with differential LR if specified
    if lstm_lr is not None and hasattr(model, 'get_param_groups'):
        # Model supports differential learning rates
        param_groups = model.get_param_groups(cnn_lr=lr, lstm_lr=lstm_lr)
        optimizer = torch.optim.Adam(param_groups, weight_decay=weight_decay)
        print("\nUsing differential learning rates:")
        for group in param_groups:
            n = sum(p.numel() for p in group['params'])
            print(f"  {group['name']}: LR={group['lr']}, {n:,} params")
    else:
        # Standard single learning rate
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        if lstm_lr is not None:
            print(f"\nWARNING: Model {model_name} doesn't support differential LR, using uniform LR={lr}")
    
    # Train w/ config logging
    print("\nTraining...")
    model, history = train_model(
        model, train_loader, val_loader, criterion, optimizer, 
        device, num_epochs=num_epochs, patience=patience, 
        save_path=model_path, log_path=log_path, config=config
    )
    
    # Test eval
    print("\nTest set evaluation:")
    test_loss, test_metrics = evaluate(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Acc:  {test_metrics['accuracy']:.4f}")
    print(f"Test F1:   {test_metrics['f1']:.4f}")
    print(f"Test AUROC: {test_metrics['auroc']:.4f}")
    
    # Complete results w/ config
    results = {
        'config': config,
        'history': history,
        'test': {
            'loss': test_loss,
            'accuracy': test_metrics['accuracy'],
            'f1': test_metrics['f1'],
            'auroc': test_metrics['auroc'],
            'sensitivity': test_metrics['sensitivity'].tolist(),
            'specificity': test_metrics['specificity']
        }
    }
    
    # Save complete results
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved: {results_path}")
    print(f"Model saved: {model_path}")
    print(f"History saved: {log_path}")
    
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description='Train model w/ comprehensive hyperparameter logging'
    )
    parser.add_argument('--model', type=str, default='test',
                       choices=list(MODEL_REGISTRY.keys()),
                       help='Model architecture')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--split_dir', type=str, default='data/splits', help='Split dir')
    parser.add_argument('--save_dir', type=str, default=None, help='Checkpoint dir (default: checkpoints/)')
    parser.add_argument('--num_epochs', type=int, default=50, help='Max epochs')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate (CNN/main)')
    parser.add_argument('--lstm_lr', type=float, default=None, 
                       help='LSTM learning rate (if None, uses --lr for all components)')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--freeze_cnn', action='store_true',  # NEW: Add freeze_cnn argument
                       help='Freeze CNN weights (only train LSTM)')
    args = parser.parse_args()
    
    results = main(
        model_name=args.model,
        seed=args.seed,
        split_dir=args.split_dir,
        save_dir=args.save_dir,
        num_epochs=args.num_epochs,
        patience=args.patience,
        lr=args.lr,
        lstm_lr=args.lstm_lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        freeze_cnn=args.freeze_cnn  # NEW: Pass freeze_cnn parameter
    )
import torch
import torch.nn as nn
import numpy as np
import random
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix
from pathlib import Path
from tqdm import tqdm

from dataset import get_dataloaders


# Model registry - maps names to (module, class) pairs
MODEL_REGISTRY = {
    'cnn_lstm': ('model', 'CNNLSTM'),
    'cnn_only': ('model_cnn_only', 'SimpleCNN'),
    'bilstm': ('model_bilstm', 'CNNBiLSTM'),
    'simple_lstm': ('model_simple_lstm', 'CNNSimpleLSTM'),
    'lstm_only': ('model_lstm_only', 'LSTMOnly'),
    'residual': ('model_residual', 'CNNLSTMResidual'),
    'gru': ('model_gru', 'CNNGRU'),
    'attention': ('model_attention', 'CNNLSTMAttention'),
}


def get_model(model_name):
    """Dynamically import and instantiate a model by name."""
    if model_name not in MODEL_REGISTRY:
        available = ', '.join(MODEL_REGISTRY.keys())
        raise ValueError(f"Unknown model: {model_name}. Available: {available}")
    
    module_name, class_name = MODEL_REGISTRY[model_name]
    module = __import__(module_name)
    model_class = getattr(module, class_name)
    return model_class()


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
        specificity.append(tn / (tn + fp))
    
    return {
        'f1': f1,
        'auroc': auroc,
        'sensitivity': sensitivity,
        'specificity': specificity
    }


def train_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(loader)


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
    
    return avg_loss, metrics


def train_model(model, train_loader, val_loader, criterion, optimizer, 
                device, num_epochs=50, patience=10, save_path="checkpoints/best_model.pt"):
    """Train model with early stopping."""
    save_path = Path(save_path)
    save_path.parent.mkdir(exist_ok=True)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )
    
    best_val_f1 = 0
    epochs_no_improve = 0
    
    for epoch in range(num_epochs):
        # Training with gradient clipping
        model.train()
        total_loss = 0
        
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            total_loss += loss.item()
        
        train_loss = total_loss / len(train_loader)
        
        # Validation
        val_loss, val_metrics = evaluate(model, val_loader, criterion, device)
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, F1: {val_metrics['f1']:.4f}, AUROC: {val_metrics['auroc']:.4f}")
        
        # Update scheduler
        scheduler.step(val_metrics['f1'])
        
        # Early stopping
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            torch.save(model.state_dict(), save_path)
            print(f"  Saved (F1: {best_val_f1:.4f})")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping after {epoch+1} epochs")
                break
    
    # Load best model
    model.load_state_dict(torch.load(save_path))
    return model


def set_seed(seed):
    """Set all random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_best_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main(model_name='cnn_lstm', seed=42, split_dir="data/splits", save_path=None, 
         num_epochs=50, patience=10, lr=1e-4, weight_decay=1e-4, batch_size=64):
    """
    Train a model.
    
    Args:
        model_name: Model architecture to use (default: 'cnn_lstm')
            Options: cnn_lstm, cnn_only, bilstm, simple_lstm, lstm_only, residual, gru, attention
        seed: Random seed for reproducibility (default: 42)
        split_dir: Directory containing train/val/test splits (default: "data/splits")
        save_path: Model checkpoint path (default: "checkpoints/{model_name}_seed{seed}.pt")
        num_epochs: Maximum training epochs (default: 50)
        patience: Early stopping patience (default: 10)
        lr: Learning rate (default: 1e-4)
        weight_decay: Weight decay for optimizer (default: 1e-4)
        batch_size: Training batch size (default: 64)
    
    Returns:
        dict: Test metrics (loss, f1, auroc, sensitivity, specificity)
    """
    # Set seeds
    set_seed(seed)
    
    # Default save path
    if save_path is None:
        save_path = f"checkpoints/{model_name}_seed{seed}.pt"
    
    # Setup
    device = get_best_device()
    print(f"Using device: {device}")
    print(f"Model: {model_name}")
    print(f"Random seed: {seed}")
    print(f"Split directory: {split_dir}")
    print(f"Learning rate: {lr}, Weight decay: {weight_decay}")
    
    # Load data
    train_loader, val_loader, test_loader, class_weights = get_dataloaders(
        batch_size=batch_size, split_dir=split_dir
    )
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    # Model
    model = get_model(model_name).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Train
    print("\nTraining...")
    model = train_model(model, train_loader, val_loader, criterion, optimizer, 
                       device, num_epochs=num_epochs, patience=patience, save_path=save_path)
    
    # Test evaluation
    print("\nTest set evaluation:")
    test_loss, test_metrics = evaluate(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test F1: {test_metrics['f1']:.4f}")
    print(f"Test AUROC: {test_metrics['auroc']:.4f}")
    print(f"Sensitivity per class: {test_metrics['sensitivity']}")
    print(f"Specificity per class: {test_metrics['specificity']}")
    
    # Return results
    return {
        'test_loss': test_loss,
        'test_f1': test_metrics['f1'],
        'test_auroc': test_metrics['auroc'],
        'test_sensitivity': test_metrics['sensitivity'].tolist(),
        'test_specificity': test_metrics['specificity']
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('--model', type=str, default='cnn_lstm',
                       choices=list(MODEL_REGISTRY.keys()),
                       help='Model architecture to train')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--split_dir', type=str, default='data/splits', help='Split directory')
    parser.add_argument('--save_path', type=str, default=None, help='Model save path')
    parser.add_argument('--num_epochs', type=int, default=50, help='Max epochs')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    args = parser.parse_args()
    
    results = main(
        model_name=args.model,
        seed=args.seed,
        split_dir=args.split_dir,
        save_path=args.save_path,
        num_epochs=args.num_epochs,
        patience=args.patience,
        lr=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size
    )
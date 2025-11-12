import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix
from pathlib import Path
from tqdm import tqdm

from model import CNNLSTM
from dataset import get_dataloaders


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
    
    best_val_f1 = 0
    epochs_no_improve = 0
    
    for epoch in range(num_epochs):
        # Training
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validation
        val_loss, val_metrics = evaluate(model, val_loader, criterion, device)
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, F1: {val_metrics['f1']:.4f}, AUROC: {val_metrics['auroc']:.4f}")
        
        # Early stopping
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            torch.save(model.state_dict(), save_path)
            print(f"  Saved best model (F1: {best_val_f1:.4f})")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping after {epoch+1} epochs")
                break
    
    # Load best model
    model.load_state_dict(torch.load(save_path))
    return model

def get_best_device():
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def main():
    # Setup
    device = get_best_device()
    print(f"Using device: {device}")
    
    # Load data
    train_loader, val_loader, test_loader, class_weights = get_dataloaders(batch_size=32)
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    # Model
    model = CNNLSTM().to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Train
    print("\nTraining...")
    model = train_model(model, train_loader, val_loader, criterion, optimizer, 
                       device, num_epochs=50, patience=10)
    
    # Test evaluation
    print("\nTest set evaluation:")
    test_loss, test_metrics = evaluate(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test F1: {test_metrics['f1']:.4f}")
    print(f"Test AUROC: {test_metrics['auroc']:.4f}")
    print(f"Sensitivity per class: {test_metrics['sensitivity']}")
    print(f"Specificity per class: {test_metrics['specificity']}")


if __name__ == "__main__":
    main()

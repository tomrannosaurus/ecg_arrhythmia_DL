import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

from model import CNNLSTM
from dataset import get_dataloaders


class SimpleCNN(nn.Module):
    """Simplified CNN-only model for debugging."""
    
    def __init__(self, num_classes=4):
        super(SimpleCNN, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Dropout(0.3),
            
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Dropout(0.3),
            
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dim
        x = self.features(x)
        x = self.classifier(x)
        return x


def check_class_distribution(split_dir="data/splits"):
    """Check actual class distribution in splits."""
    y_train = np.load(Path(split_dir) / "y_train.npy")
    y_val = np.load(Path(split_dir) / "y_val.npy")
    
    print("\nClass distribution:")
    classes = ["Normal", "AF", "Other", "Noisy"]
    for i, name in enumerate(classes):
        train_count = np.sum(y_train == i)
        train_pct = 100 * train_count / len(y_train)
        print(f"  {name}: {train_count} ({train_pct:.1f}%)")
    
    weights = np.load(Path(split_dir) / "class_weights.npy")
    print(f"\nClass weights: {weights}")
    print(f"Weight ratio (max/min): {weights.max()/weights.min():.2f}")


def train_simple(num_epochs=30):
    """Train simplified model using same method as compare_models."""
    from train import train_model, evaluate
    
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Check data distribution
    check_class_distribution()
    
    # Load data
    train_loader, val_loader, test_loader, class_weights = get_dataloaders(batch_size=64)
    print(f"\nTrain batches: {len(train_loader)}")
    
    # Simple CNN model
    model = SimpleCNN().to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    
    # Train using same function as compare_models
    model = train_model(model, train_loader, val_loader, criterion, optimizer,
                       device, num_epochs=num_epochs, patience=10,
                       save_path="checkpoints/simple_model.pt")
    
    # Test evaluation
    print("\nTest set evaluation:")
    test_loss, test_metrics = evaluate(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test F1: {test_metrics['f1']:.4f}")
    print(f"Test AUROC: {test_metrics['auroc']:.4f}")


if __name__ == "__main__":
    train_simple()
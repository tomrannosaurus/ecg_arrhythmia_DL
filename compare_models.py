import torch
import torch.nn as nn
from pathlib import Path

from model import CNNLSTM
from debug_train import SimpleCNN
from dataset import get_dataloaders
from train import train_model, evaluate


def compare_models():
    """Compare CNN-only vs CNN-LSTM performance."""
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}\n")
    
    # Load data
    train_loader, val_loader, test_loader, class_weights = get_dataloaders(batch_size=64)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    
    results = {}
    
    # Test 1: Simple CNN
    print("="*60)
    print("MODEL 1: Simple CNN (no temporal modeling)")
    print("="*60)
    model_cnn = SimpleCNN().to(device)
    print(f"Parameters: {sum(p.numel() for p in model_cnn.parameters()):,}")
    
    optimizer = torch.optim.Adam(model_cnn.parameters(), lr=1e-4, weight_decay=1e-4)
    model_cnn = train_model(model_cnn, train_loader, val_loader, criterion, optimizer,
                           device, num_epochs=50, patience=10, 
                           save_path="checkpoints/cnn_only.pt")
    
    _, metrics_cnn = evaluate(model_cnn, test_loader, criterion, device)
    results['CNN'] = metrics_cnn
    print(f"\nCNN Test Results:")
    print(f"  F1: {metrics_cnn['f1']:.4f}")
    print(f"  AUROC: {metrics_cnn['auroc']:.4f}\n")
    
    # Test 2: CNN-LSTM
    print("="*60)
    print("MODEL 2: CNN-LSTM (with temporal modeling)")
    print("="*60)
    model_lstm = CNNLSTM().to(device)
    print(f"Parameters: {sum(p.numel() for p in model_lstm.parameters()):,}")
    
    optimizer = torch.optim.Adam(model_lstm.parameters(), lr=1e-4, weight_decay=1e-4)
    model_lstm = train_model(model_lstm, train_loader, val_loader, criterion, optimizer,
                            device, num_epochs=50, patience=10,
                            save_path="checkpoints/cnn_lstm.pt")
    
    _, metrics_lstm = evaluate(model_lstm, test_loader, criterion, device)
    results['CNN-LSTM'] = metrics_lstm
    print(f"\nCNN-LSTM Test Results:")
    print(f"  F1: {metrics_lstm['f1']:.4f}")
    print(f"  AUROC: {metrics_lstm['auroc']:.4f}\n")
    
    # Summary
    print("="*60)
    print("COMPARISON SUMMARY")
    print("="*60)
    print(f"{'Model':<15} {'F1':<10} {'AUROC':<10} {'Parameters'}")
    print("-"*60)
    print(f"{'CNN':<15} {results['CNN']['f1']:<10.4f} {results['CNN']['auroc']:<10.4f} {sum(p.numel() for p in model_cnn.parameters()):,}")
    print(f"{'CNN-LSTM':<15} {results['CNN-LSTM']['f1']:<10.4f} {results['CNN-LSTM']['auroc']:<10.4f} {sum(p.numel() for p in model_lstm.parameters()):,}")
    
    winner = 'CNN-LSTM' if results['CNN-LSTM']['f1'] > results['CNN']['f1'] else 'CNN'
    print(f"\nBest model: {winner}")


if __name__ == "__main__":
    compare_models()

"""
Model: CNN-Only (No LSTM)
Diagnostic model to verify CNN can learn independently.

Usage:
    python train.py --model cnn_only --seed 42
"""

import torch
import torch.nn as nn

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

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = SimpleCNN()
    print(f"SimpleCNN parameters: {count_parameters(model):,}")
    # dummy forward pass to verify shapes
    x = torch.randn(8, 1500)  # (batch_size, sequence_length)
    y = model(x)
    print(f"Input: {x.shape} -> Output: {y.shape}")
"""
Model: CNN with GRU
Replaces LSTM with GRU (simpler gating, fewer parameters).

Usage:
    from model_gru import CNNGRU
    model = CNNGRU()
"""

import torch
import torch.nn as nn


class CNNGRU(nn.Module):
    """CNN with GRU for ECG classification."""
    
    def __init__(self, input_size=1500, num_classes=4,
                 cnn_channels=[32, 64, 128],
                 gru_hidden=128, gru_layers=2, dropout=0.5):
        super(CNNGRU, self).__init__()
        
        # CNN feature extractor (same as original)
        self.cnn = nn.Sequential(
            nn.Conv1d(1, cnn_channels[0], kernel_size=7, padding=3),
            nn.BatchNorm1d(cnn_channels[0]),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),
            
            nn.Conv1d(cnn_channels[0], cnn_channels[1], kernel_size=5, padding=2),
            nn.BatchNorm1d(cnn_channels[1]),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),
            
            nn.Conv1d(cnn_channels[1], cnn_channels[2], kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_channels[2]),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),
        )
        
        # GRU instead of LSTM
        self.gru = nn.GRU(
            input_size=cnn_channels[2],
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
            dropout=dropout if gru_layers > 1 else 0
        )
        
        # Classifier
        self.fc = nn.Sequential(
            nn.Linear(gru_hidden, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        batch_size = x.size(0)
        
        x = x.unsqueeze(1)
        x = self.cnn(x)
        x = x.permute(0, 2, 1)
        
        # GRU only has hidden state (no cell state)
        gru_out, h_n = self.gru(x)
        features = h_n[-1]
        
        out = self.fc(features)
        return out


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = CNNGRU()
    print(f"CNNGRU parameters: {count_parameters(model):,}")
    
    x = torch.randn(8, 1500)
    y = model(x)
    print(f"Input: {x.shape} -> Output: {y.shape}")

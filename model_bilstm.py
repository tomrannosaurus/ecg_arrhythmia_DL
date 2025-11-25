"""
Model: CNN with Bidirectional LSTM
Captures both forward and backward temporal dependencies.

Usage:
    from model_bilstm import CNNBiLSTM
    model = CNNBiLSTM()
"""

import torch
import torch.nn as nn


class CNNBiLSTM(nn.Module):
    """CNN with Bidirectional LSTM for ECG classification."""
    
    def __init__(self, input_size=1500, num_classes=4,
                 cnn_channels=[32, 64, 128],
                 lstm_hidden=128, lstm_layers=2, dropout=0.5):
        super(CNNBiLSTM, self).__init__()
        
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
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=cnn_channels[2],
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0,
            bidirectional=True  # KEY CHANGE
        )
        
        # Classifier (input is 2x lstm_hidden due to bidirectional)
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        batch_size = x.size(0)
        
        x = x.unsqueeze(1)
        x = self.cnn(x)
        x = x.permute(0, 2, 1)
        
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Concatenate final forward and backward hidden states
        # h_n shape: (num_layers * 2, batch, hidden)
        h_forward = h_n[-2]  # Last layer forward
        h_backward = h_n[-1]  # Last layer backward
        features = torch.cat([h_forward, h_backward], dim=1)
        
        out = self.fc(features)
        return out


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = CNNBiLSTM()
    print(f"CNNBiLSTM parameters: {count_parameters(model):,}")
    
    x = torch.randn(8, 1500)
    y = model(x)
    print(f"Input: {x.shape} -> Output: {y.shape}")

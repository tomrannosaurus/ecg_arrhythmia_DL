"""
Model: CNN-LSTM Simple (Improved Architecture, Smaller Capacity)

Similar to seq16, but with reduced capacity:
- Smaller LSTM (32 units instead of 64)
- Shorter sequences (16 timesteps)
- Good for testing if model is too complex

Usage:
    python train.py --model simple_lstm --seed 42
"""

import torch
import torch.nn as nn


class CNNSimpleLSTM(nn.Module):
    """Simplified CNN-LSTM with reduced capacity."""
    
    def __init__(self, input_size=1500, num_classes=4,
                 cnn_channels=[32, 64, 128],
                 lstm_hidden=32,  # SMALLER than base model
                 dropout=0.2,
                 target_seq_len=16):  # SHORT sequences
        super(CNNSimpleLSTM, self).__init__()
        
        self.target_seq_len = target_seq_len
        
        # CNN feature extractor - NO BATCH NORM
        self.cnn = nn.Sequential(
            # Block 1
            nn.Conv1d(1, cnn_channels[0], kernel_size=7, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),
            
            # Block 2
            nn.Conv1d(cnn_channels[0], cnn_channels[1], kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),
            
            # Block 3
            nn.Conv1d(cnn_channels[1], cnn_channels[2], kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),
        )
        
        # LayerNorm
        self.layer_norm = nn.LayerNorm(cnn_channels[2])
        
        # Simple LSTM (smaller)
        self.lstm = nn.LSTM(
            input_size=cnn_channels[2],
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
            dropout=0
        )
        
        # Classifier (smaller)
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, num_classes)
        )
    
    def forward(self, x):
        """
        Args:
            x: (batch, 1500)
        Returns:
            (batch, num_classes)
        """
        
        # CNN
        x = x.unsqueeze(1)
        x = self.cnn(x)
        
        # Reduce sequence length
        x = torch.nn.functional.interpolate(
            x, size=self.target_seq_len, 
            mode='linear', align_corners=False
        )
        
        # Prepare for LSTM
        x = x.permute(0, 2, 1)
        x = self.layer_norm(x)
        
        # LSTM
        lstm_out, (h_n, c_n) = self.lstm(x)
        features = h_n[-1]
        
        # Classify
        out = self.fc(features)
        
        return out


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = CNNSimpleLSTM()
    print(f"CNNSimpleLSTM parameters: {count_parameters(model):,}")
    
    # Test forward pass
    x = torch.randn(8, 1500)
    y = model(x)
    print(f"Input: {x.shape} -> Output: {y.shape}")
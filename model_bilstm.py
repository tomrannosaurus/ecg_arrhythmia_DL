"""
Model: CNN with Bidirectional LSTM (Improved Architecture)

Based on improved architecture from cnn_bilstm_seq24:
- Short sequences (16 timesteps)
- No BatchNorm in CNN
- LayerNorm before LSTM
- Low dropout (0.1)
- Single BiLSTM layer

Usage:
    python train.py --model bilstm --seed 42
"""

import torch
import torch.nn as nn


class CNNBiLSTM(nn.Module):
    """CNN with Bidirectional LSTM for ECG classification."""
    
    def __init__(self, input_size=1500, num_classes=4,
                 cnn_channels=[32, 64, 128],
                 lstm_hidden=128, 
                 dropout=0.1,
                 target_seq_len=16):
        super(CNNBiLSTM, self).__init__()
        
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
        
        # LayerNorm for LSTM input
        self.layer_norm = nn.LayerNorm(cnn_channels[2])
        
        # Bidirectional LSTM - Single layer
        self.lstm = nn.LSTM(
            input_size=cnn_channels[2],
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
            dropout=0,
            bidirectional=True
        )
        
        # Classifier (input is 2x lstm_hidden due to bidirectional)
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        """
        Args:
            x: (batch, 1500)
        Returns:
            (batch, num_classes)
        """
        
        # CNN: (batch, 1500) -> (batch, 1, 1500) -> (batch, 128, 187)
        x = x.unsqueeze(1)
        x = self.cnn(x)
        
        # Reduce sequence length (batch, 128, 187) -> (batch, 128, 16)
        x = torch.nn.functional.interpolate(
            x, size=self.target_seq_len, 
            mode='linear', align_corners=False
        )
        
        # Prepare for LSTM: (batch, 128, 16) -> (batch, 16, 128)
        x = x.permute(0, 2, 1)
        
        # Normalize
        x = self.layer_norm(x)
        
        # BiLSTM: (batch, 16, 128) -> (batch, 16, 256)
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Concatenate final forward and backward hidden states
        # h_n shape: (2, batch, hidden) for bidirectional
        h_forward = h_n[-2]   # Forward direction
        h_backward = h_n[-1]  # Backward direction
        features = torch.cat([h_forward, h_backward], dim=1)  # (batch, 256)
        
        # Classify
        out = self.fc(features)
        return out


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = CNNBiLSTM()
    print(f"CNNBiLSTM parameters: {count_parameters(model):,}")
    
    # Test forward pass
    x = torch.randn(8, 1500)
    y = model(x)
    print(f"Input: {x.shape} -> Output: {y.shape}")
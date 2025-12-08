"""
Model: CNN-GRU (Improved Architecture)

Based on improved architecture from CNN-LSTM series, but using GRU instead of LSTM.
GRU is simpler than LSTM (fewer parameters, faster training).

Usage:
    python train.py --model gru --seed 42
"""

import torch
import torch.nn as nn


class CNNGRU(nn.Module):
    """CNN-GRU with improved architecture."""
    
    def __init__(self, input_size=1500, num_classes=4,
                 cnn_channels=[32, 64, 128],
                 gru_hidden=96,
                 dropout=0.15,
                 target_seq_len=20,
                 bidirectional=False):
        super(CNNGRU, self).__init__()
        
        self.target_seq_len = target_seq_len
        self.bidirectional = bidirectional
        
        # CNN feature extractor - WITH BATCH NORM
        self.cnn = nn.Sequential(
            # Block 1
            nn.Conv1d(1, cnn_channels[0], kernel_size=7, padding=3),
            nn.BatchNorm1d(cnn_channels[0]),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),
            
            # Block 2  
            nn.Conv1d(cnn_channels[0], cnn_channels[1], kernel_size=5, padding=2),
            nn.BatchNorm1d(cnn_channels[1]),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),
            
            # Block 3
            nn.Conv1d(cnn_channels[1], cnn_channels[2], kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_channels[2]),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),
        )
        
        # GRU (simpler than LSTM - no cell state)
        self.gru = nn.GRU(
            input_size=cnn_channels[2],
            hidden_size=gru_hidden,
            num_layers=1,
            batch_first=True,
            dropout=0,
            bidirectional=bidirectional
        )
        
        # Classifier
        gru_output_size = gru_hidden * 2 if bidirectional else gru_hidden
        self.fc = nn.Sequential(
            nn.Linear(gru_output_size, 48),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(48, num_classes)
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
        
        # Reduce sequence length
        x = torch.nn.functional.interpolate(
            x, size=self.target_seq_len, 
            mode='linear', align_corners=False
        )
        
        # Prepare for GRU: (batch, 128, seq) -> (batch, seq, 128)
        x = x.permute(0, 2, 1)
        
        # GRU: (batch, seq, 128) -> (batch, seq, gru_hidden)
        gru_out, h_n = self.gru(x)
        
        # Use last hidden state
        if self.bidirectional:
            # Concatenate forward and backward final states
            features = torch.cat([h_n[-2], h_n[-1]], dim=1)
        else:
            features = h_n[-1]
        
        # Classify
        out = self.fc(features)
        
        return out


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = CNNGRU()
    print(f"CNNGRU parameters: {count_parameters(model):,}")
    
    # Test forward pass
    x = torch.randn(8, 1500)
    y = model(x)
    print(f"Input: {x.shape} -> Output: {y.shape}")
"""
Model: CNN-LSTM with Residual Connection (Improved Architecture)

Adds skip connection from CNN output to post-LSTM features.
Based on improved architecture from ultra series.

Usage:
    from model_residual import CNNLSTMResidual
    model = CNNLSTMResidual()
"""

import torch
import torch.nn as nn


class CNNLSTMResidual(nn.Module):
    """CNN-LSTM with residual connection around LSTM.
    
    The residual connection allows gradient flow directly from CNN
    to classifier, potentially helping with vanishing gradients.
    """
    
    def __init__(self, input_size=1500, num_classes=4,
                 cnn_channels=[32, 64, 128],
                 lstm_hidden=96, 
                 dropout=0.15,
                 target_seq_len=20):
        super(CNNLSTMResidual, self).__init__()
        
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
        
        # LSTM
        self.lstm = nn.LSTM(
            input_size=cnn_channels[2],
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
            dropout=0
        )
        
        # Residual projection (CNN features -> same size as LSTM output)
        self.residual_proj = nn.Linear(cnn_channels[2], lstm_hidden)
        
        # Classifier (takes LSTM output + residual)
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden, 48),
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
        
        # CNN: (batch, 1500) -> (batch, 128, 187)
        x = x.unsqueeze(1)
        x = self.cnn(x)
        
        # Reduce sequence length
        x = torch.nn.functional.interpolate(
            x, size=self.target_seq_len, 
            mode='linear', align_corners=False
        )
        
        # Prepare for LSTM: (batch, 128, seq) -> (batch, seq, 128)
        x = x.permute(0, 2, 1)
        
        # Normalize
        x = self.layer_norm(x)
        
        # Save for residual (pool over sequence dimension)
        residual = x.mean(dim=1)  # (batch, 128)
        residual = self.residual_proj(residual)  # (batch, lstm_hidden)
        
        # LSTM
        lstm_out, (h_n, c_n) = self.lstm(x)
        features = h_n[-1]  # (batch, lstm_hidden)
        
        # Add residual connection
        features = features + residual  # Element-wise addition
        
        # Classify
        out = self.fc(features)
        
        return out


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = CNNLSTMResidual()
    print(f"CNNLSTMResidual parameters: {count_parameters(model):,}")
    
    # Test forward pass
    x = torch.randn(8, 1500)
    y = model(x)
    print(f"Input: {x.shape} -> Output: {y.shape}")
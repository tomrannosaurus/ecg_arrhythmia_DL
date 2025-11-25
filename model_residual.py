"""
Model: CNN-LSTM with Residual Connection
Adds skip connection from CNN output to post-LSTM features.

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
                 lstm_hidden=128, lstm_layers=2, dropout=0.5):
        super(CNNLSTMResidual, self).__init__()
        
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
        
        # LSTM
        self.lstm = nn.LSTM(
            input_size=cnn_channels[2],
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0
        )
        
        # Projection for residual (CNN pooled output to match LSTM hidden size)
        self.cnn_pool = nn.AdaptiveAvgPool1d(1)
        self.residual_proj = nn.Linear(cnn_channels[2], lstm_hidden)
        
        # Classifier (takes LSTM output + residual)
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden, 64),
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
        batch_size = x.size(0)
        
        # Add channel dim: (batch, 1500) -> (batch, 1, 1500)
        x = x.unsqueeze(1)
        
        # CNN: (batch, 1, 1500) -> (batch, 128, seq_len)
        cnn_out = self.cnn(x)
        
        # Residual path: pool CNN output and project
        cnn_pooled = self.cnn_pool(cnn_out).squeeze(-1)  # (batch, 128)
        residual = self.residual_proj(cnn_pooled)  # (batch, lstm_hidden)
        
        # LSTM path: (batch, 128, seq_len) -> (batch, seq_len, 128)
        lstm_in = cnn_out.permute(0, 2, 1)
        lstm_out, (h_n, c_n) = self.lstm(lstm_in)
        lstm_features = h_n[-1]  # (batch, lstm_hidden)
        
        # Combine LSTM output with residual
        combined = lstm_features + residual  # Element-wise addition
        
        # Classify
        out = self.fc(combined)
        
        return out


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = CNNLSTMResidual()
    print(f"CNNLSTMResidual parameters: {count_parameters(model):,}")
    
    x = torch.randn(8, 1500)
    y = model(x)
    print(f"Input: {x.shape} -> Output: {y.shape}")

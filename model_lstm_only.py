"""
Model: LSTM-Only (Diagnostic)

No CNN, just LSTM directly on ECG signal.
Useful for testing if CNN is actually helping.

Usage:
    from model_lstm_only import LSTMOnly
    model = LSTMOnly()
"""

import torch
import torch.nn as nn


class LSTMOnly(nn.Module):
    """LSTM-only model (no CNN)."""
    
    def __init__(self, input_size=1500, num_classes=4,
                 lstm_hidden=128, 
                 dropout=0.15):
        super(LSTMOnly, self).__init__()
        
        # Project raw input to lower dimension
        self.input_proj = nn.Linear(1, 16)
        
        # LayerNorm
        self.layer_norm = nn.LayerNorm(16)
        
        # LSTM - can use 2 layers since it's the only component
        self.lstm = nn.LSTM(
            input_size=16,
            hidden_size=lstm_hidden,
            num_layers=2,
            batch_first=True,
            dropout=dropout
        )
        
        # Classifier
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
        # Reshape: (batch, 1500) -> (batch, 1500, 1)
        x = x.unsqueeze(-1)
        
        # Project: (batch, 1500, 1) -> (batch, 1500, 16)
        x = self.input_proj(x)
        
        # Normalize
        x = self.layer_norm(x)
        
        # LSTM: (batch, 1500, 16) -> (batch, 1500, lstm_hidden)
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Use last hidden state
        features = h_n[-1]
        
        # Classify
        out = self.fc(features)
        
        return out


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = LSTMOnly()
    print(f"LSTMOnly parameters: {count_parameters(model):,}")
    
    # Test forward pass
    x = torch.randn(8, 1500)
    y = model(x)
    print(f"Input: {x.shape} -> Output: {y.shape}")
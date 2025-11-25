"""
Model: LSTM-Only (No CNN)
Diagnostic model to verify LSTM can learn independently.

Usage:
    from model_lstm_only import LSTMOnly
    model = LSTMOnly()
"""

import torch
import torch.nn as nn


class LSTMOnly(nn.Module):
    """LSTM-only model for ECG classification (no CNN).
    
    Diagnostic model to verify the LSTM component can learn
    when given raw ECG input directly.
    """
    
    def __init__(self, input_size=1500, num_classes=4,
                 lstm_hidden=128, lstm_layers=2, dropout=0.5):
        super(LSTMOnly, self).__init__()
        
        self.input_size = input_size
        
        # Reshape raw input into sequence
        # Split 1500 samples into chunks for LSTM
        self.chunk_size = 50  # 50 samples per timestep
        self.seq_len = input_size // self.chunk_size  # 30 timesteps
        
        # LSTM processes sequence of chunks
        self.lstm = nn.LSTM(
            input_size=self.chunk_size,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0
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
        batch_size = x.size(0)
        
        # Reshape to (batch, seq_len, chunk_size)
        x = x.view(batch_size, self.seq_len, self.chunk_size)
        
        # LSTM
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Use last hidden state
        features = h_n[-1]  # (batch, lstm_hidden)
        
        # Classify
        out = self.fc(features)
        
        return out


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = LSTMOnly()
    print(f"LSTMOnly parameters: {count_parameters(model):,}")
    
    x = torch.randn(8, 1500)
    y = model(x)
    print(f"Input: {x.shape} -> Output: {y.shape}")

"""
Model: CNN-LSTM with Attention
Adds attention mechanism over LSTM outputs.

Usage:
    from model_attention import CNNLSTMAttention
    model = CNNLSTMAttention()
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    """Simple attention mechanism over sequence."""
    
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.attention = nn.Linear(hidden_size, 1)
    
    def forward(self, lstm_out):
        """
        Args:
            lstm_out: (batch, seq_len, hidden_size)
        Returns:
            context: (batch, hidden_size)
            weights: (batch, seq_len, 1)
        """
        # Compute attention scores
        scores = self.attention(lstm_out)  # (batch, seq_len, 1)
        weights = F.softmax(scores, dim=1)
        
        # Weighted sum
        context = (lstm_out * weights).sum(dim=1)  # (batch, hidden_size)
        
        return context, weights


class CNNLSTMAttention(nn.Module):
    """CNN-LSTM with attention mechanism for ECG classification."""
    
    def __init__(self, input_size=1500, num_classes=4,
                 cnn_channels=[32, 64, 128],
                 lstm_hidden=128, lstm_layers=2, dropout=0.5):
        super(CNNLSTMAttention, self).__init__()
        
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
        
        # Attention mechanism
        self.attention = Attention(lstm_hidden)
        
        # Classifier
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )
        
        # Store attention weights for interpretability
        self.attention_weights = None
    
    def forward(self, x):
        batch_size = x.size(0)
        
        x = x.unsqueeze(1)
        x = self.cnn(x)
        x = x.permute(0, 2, 1)
        
        # LSTM outputs all timesteps
        lstm_out, (h_n, c_n) = self.lstm(x)  # (batch, seq_len, hidden)
        
        # Apply attention
        context, self.attention_weights = self.attention(lstm_out)
        
        out = self.fc(context)
        return out
    
    def get_attention_weights(self):
        """Return attention weights from last forward pass."""
        return self.attention_weights


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = CNNLSTMAttention()
    print(f"CNNLSTMAttention parameters: {count_parameters(model):,}")
    
    x = torch.randn(8, 1500)
    y = model(x)
    print(f"Input: {x.shape} -> Output: {y.shape}")
    print(f"Attention weights shape: {model.get_attention_weights().shape}")

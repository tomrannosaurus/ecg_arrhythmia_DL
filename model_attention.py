"""
Model: CNN-LSTM with Attention Mechanism (Improved Architecture)

Adds attention mechanism to weight LSTM outputs before classification.
Based on improved architecture from ultra series.

Usage:
    python train.py --model attention --seed 42
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionLayer(nn.Module):
    """Attention mechanism to weight sequence outputs."""
    
    def __init__(self, hidden_size):
        super(AttentionLayer, self).__init__()
        self.attention = nn.Linear(hidden_size, 1)
    
    def forward(self, lstm_output):
        """
        Args:
            lstm_output: (batch, seq_len, hidden_size)
        Returns:
            context: (batch, hidden_size)
            attention_weights: (batch, seq_len)
        """
        # Compute attention scores
        attn_scores = self.attention(lstm_output)  # (batch, seq_len, 1)
        attn_scores = attn_scores.squeeze(-1)       # (batch, seq_len)
        
        # Apply softmax to get weights
        attn_weights = F.softmax(attn_scores, dim=1)  # (batch, seq_len)
        
        # Compute weighted sum
        context = torch.bmm(
            attn_weights.unsqueeze(1),  # (batch, 1, seq_len)
            lstm_output                  # (batch, seq_len, hidden)
        ).squeeze(1)                     # (batch, hidden)
        
        return context, attn_weights


class CNNLSTMAttention(nn.Module):
    """CNN-LSTM with attention mechanism."""
    
    def __init__(self, input_size=1500, num_classes=4,
                 cnn_channels=[32, 64, 128],
                 lstm_hidden=96,
                 dropout=0.15,
                 target_seq_len=20):
        super(CNNLSTMAttention, self).__init__()
        
        self.target_seq_len = target_seq_len
        
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
        
        # LSTM
        self.lstm = nn.LSTM(
            input_size=cnn_channels[2],
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
            dropout=0
        )
        
        # Attention mechanism
        self.attention = AttentionLayer(lstm_hidden)
        
        # Classifier
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden, 48),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(48, num_classes)
        )
    
    def forward(self, x, return_attention=False):
        """
        Args:
            x: (batch, 1500)
            return_attention: Whether to return attention weights
        Returns:
            out: (batch, num_classes)
            attn_weights: (batch, seq_len) if return_attention=True
        """
        
        # CNN: (batch, 1500) -> (batch, 1, 1500) -> (batch, 128, 187)
        x = x.unsqueeze(1)
        x = self.cnn(x)
        
        # Reduce sequence length
        x = torch.nn.functional.interpolate(
            x, size=self.target_seq_len, 
            mode='linear', align_corners=False
        )
        
        # Prepare for LSTM: (batch, 128, seq) -> (batch, seq, 128)
        x = x.permute(0, 2, 1)
        
        # LSTM: (batch, seq, 128) -> (batch, seq, lstm_hidden)
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Apply attention
        context, attn_weights = self.attention(lstm_out)
        
        # Classify
        out = self.fc(context)
        
        if return_attention:
            return out, attn_weights
        return out

    def freeze_cnn(self):
        """Freeze CNN weights (only RNN trains)."""
        for param in self.cnn.parameters():
            param.requires_grad = False
    
    def unfreeze_cnn(self):
        """Unfreeze CNN for fine-tuning."""
        for param in self.cnn.parameters():
            param.requires_grad = True
    
    def get_param_groups(self, cnn_lr, rnn_lr, fc_lr=None):
        """Return parameter groups for differential learning rates."""
        if fc_lr is None:
            fc_lr = cnn_lr
        return [
            {'params': self.cnn.parameters(), 'lr': cnn_lr, 'name': 'cnn'},
            {'params': self.lstm.parameters(), 'lr': rnn_lr, 'name': 'rnn'},
            {'params': self.attention.parameters(), 'lr': fc_lr, 'name': 'attention'},
            {'params': self.fc.parameters(), 'lr': fc_lr, 'name': 'fc'}
        ]

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = CNNLSTMAttention()
    print(f"CNNLSTMAttention parameters: {count_parameters(model):,}")
    
    # Test forward pass
    x = torch.randn(8, 1500)
    y = model(x)
    print(f"Input: {x.shape} -> Output: {y.shape}")
    
    # Test with attention weights
    y, attn = model(x, return_attention=True)
    print(f"Attention weights shape: {attn.shape}")
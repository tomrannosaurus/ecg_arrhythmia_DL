"""
Model: CNN-LSTM with Critical Fixes
- Removes BatchNorm before LSTM (interferes with temporal processing)
- Adds LayerNorm after permute (proper normalization for LSTM input)
- Designed for differential learning rates (CNN vs LSTM components)

Usage:
    python train.py --model cnn_lstm_fixed --seed 42
    python train.py --model cnn_lstm_fixed --lr 1e-3 --lstm_lr 1e-4 --seed 42
    python train.py --model cnn_lstm_fixed --lr 1e-3 --lstm_lr 1e-5 --seed 42
"""

import torch
import torch.nn as nn


class CNNLSTMFixed(nn.Module):
    """CNN-LSTM with architecture fixes for proper training.
    
    Key changes from original:
    1. NO BatchNorm in CNN - BatchNorm statistics computed on channels
       don't translate properly when those channels become LSTM features
    2. LayerNorm applied AFTER permute - normalizes features per timestep
    3. Separate component groups for differential learning rates
    """
    
    def __init__(self, input_size=1500, num_classes=4,
                 cnn_channels=[32, 64, 128],
                 lstm_hidden=128, lstm_layers=2, dropout=0.5):
        super(CNNLSTMFixed, self).__init__()
        
        # CNN feature extractor - NO BATCH NORM
        self.cnn = nn.Sequential(
            # Conv block 1
            nn.Conv1d(1, cnn_channels[0], kernel_size=7, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),
            
            # Conv block 2
            nn.Conv1d(cnn_channels[0], cnn_channels[1], kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),
            
            # Conv block 3
            nn.Conv1d(cnn_channels[1], cnn_channels[2], kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),
        )
        
        # Calculate CNN output size
        self.cnn_output_size = input_size // 8  # Three MaxPool2d layers
        
        # LayerNorm applied AFTER permute (normalizes across feature dimension)
        # This is correct: normalizes the 128 features at each of the 187 timesteps
        self.layer_norm = nn.LayerNorm(cnn_channels[2])
        
        # LSTM temporal modeling
        self.lstm = nn.LSTM(
            input_size=cnn_channels[2],
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
            x: input tensor of shape (batch, 1500)
        
        Returns:
            output tensor of shape (batch, 4)
        """
        batch_size = x.size(0)
        
        # Add channel dimension for Conv1d: (batch, 1500) -> (batch, 1, 1500)
        x = x.unsqueeze(1)
        
        # CNN feature extraction: (batch, 1, 1500) -> (batch, 128, seq_len)
        x = self.cnn(x)
        
        # Reshape for LSTM: (batch, 128, seq_len) -> (batch, seq_len, 128)
        x = x.permute(0, 2, 1)
        
        # Apply LayerNorm NOW (after permute, before LSTM)
        # This normalizes the 128 features at each timestep
        x = self.layer_norm(x)
        
        # LSTM: (batch, seq_len, 128) -> (batch, seq_len, lstm_hidden)
        x, (h_n, c_n) = self.lstm(x)
        
        # Use last hidden state: (batch, lstm_hidden)
        x = h_n[-1]
        
        # Classifier: (batch, lstm_hidden) -> (batch, 4)
        x = self.fc(x)
        
        return x
    
    def get_param_groups(self, cnn_lr, lstm_lr, fc_lr=None):
        """Return parameter groups for differential learning rates.
        
        Args:
            cnn_lr: Learning rate for CNN
            lstm_lr: Learning rate for LSTM (typically 10-100x smaller)
            fc_lr: Learning rate for classifier (optional, defaults to cnn_lr) Not currently implemented in train.py
        
        Returns:
            List of parameter group dicts for optimizer
        """
        if fc_lr is None:
            fc_lr = cnn_lr
        
        return [
            {'params': self.cnn.parameters(), 'lr': cnn_lr, 'name': 'cnn'},
            {'params': self.layer_norm.parameters(), 'lr': lstm_lr, 'name': 'layer_norm'},
            {'params': self.lstm.parameters(), 'lr': lstm_lr, 'name': 'lstm'},
            {'params': self.fc.parameters(), 'lr': fc_lr, 'name': 'fc'}
        ]


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test model
    model = CNNLSTMFixed()
    print(f"Model parameters: {count_parameters(model):,}")
    
    # Test forward pass
    x = torch.randn(8, 1500)  # batch of 8 samples
    y = model(x)
    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    
    # Test parameter groups
    print("\nParameter groups for differential LR:")
    param_groups = model.get_param_groups(cnn_lr=1e-4, lstm_lr=1e-5)
    for group in param_groups:
        n_params = sum(p.numel() for p in group['params'])
        print(f"  {group['name']}: {n_params:,} params, LR={group['lr']}")

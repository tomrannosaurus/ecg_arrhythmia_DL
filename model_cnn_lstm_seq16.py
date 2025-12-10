"""
Model: CNN-LSTM Short Sequences

1. Reduce sequence to ~16 timesteps (vs 187) - CRITICAL
2. Lower dropout to 0.2 (vs 0.5)
3. Single LSTM layer (64 units)
4. Option to freeze CNN for two-stage training

Usage:
    python train.py --model cnn_lstm_seq16 --seed 42
    python train.py --model cnn_lstm_seq16 --lr 1e-4 --rnn_lr 1e-5 --seed 42
"""

import torch
import torch.nn as nn


class CNNLSTMSeq16(nn.Module):
    """Ultra-simple CNN-LSTM with short sequences.
    
    Addresses the #1 issue: Sequence length too long (187 → 16)
    Based on sensors2406306 paper that uses only 10 timesteps.
    """
    
    def __init__(self, input_size=1500, num_classes=4,
                 cnn_channels=[32, 64, 128],
                 lstm_hidden=64, dropout=0.2,  # Lower dropout!
                 target_seq_len=16):  # Target sequence length
        super(CNNLSTMSeq16, self).__init__()
        
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
        
        # CRITICAL: Target sequence length for pooling
        # This reduces 187 timesteps down to target_seq_len (default 16)
        self.target_seq_len = target_seq_len
        
        # Single-layer LSTM (simpler = more stable)
        self.lstm = nn.LSTM(
            input_size=cnn_channels[2],
            hidden_size=lstm_hidden,
            num_layers=1,  # Single layer
            batch_first=True,
            dropout=0  # No dropout in single-layer LSTM
        )
        
        # Classifier
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes)
        )
    
    def forward(self, x):        
        # CNN: (batch, 1500) -> (batch, 1, 1500) -> (batch, 128, 187)
        x = x.unsqueeze(1)
        x = self.cnn(x)
        
        # CRITICAL: Reduce sequence length (batch, 128, 187) -> (batch, 128, 16)
        # Using interpolate instead of AdaptiveAvgPool1d for MPS compatibility
        x = torch.nn.functional.interpolate(x, size=self.target_seq_len, mode='linear', align_corners=False)
        
        # Prepare for LSTM: (batch, 128, 16) -> (batch, 16, 128)
        x = x.permute(0, 2, 1)
        
        # LSTM: (batch, 16, 128) -> (batch, 16, 64)
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Use last hidden state
        features = h_n[-1]  # (batch, 64)
        
        # Classify
        out = self.fc(features)
        
        return out
    
    def freeze_cnn(self):
        """Freeze CNN weights for two-stage training."""
        for param in self.cnn.parameters():
            param.requires_grad = False
        print("CNN frozen - only LSTM will train")
    
    def unfreeze_cnn(self):
        """Unfreeze CNN for fine-tuning."""
        for param in self.cnn.parameters():
            param.requires_grad = True
        print("CNN unfrozen")
    
    def get_param_groups(self, cnn_lr, rnn_lr, fc_lr=None):
        """Return parameter groups for differential learning rates."""
        if fc_lr is None:
            fc_lr = cnn_lr
        
        return [
            {'params': self.cnn.parameters(), 'lr': cnn_lr, 'name': 'cnn'},
            {'params': self.lstm.parameters(), 'lr': rnn_lr, 'name': 'lstm'},
            {'params': self.fc.parameters(), 'lr': fc_lr, 'name': 'fc'}
        ]


def count_parameters(model):
    """Count trainable parameters."""
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    return total, frozen


if __name__ == "__main__":
    # Test model
    model = CNNLSTMSeq16(target_seq_len=16)
    total, frozen = count_parameters(model)
    print(f"Trainable parameters: {total:,}")
    print(f"Frozen parameters: {frozen:,}")
    
    # Test forward pass
    x = torch.randn(8, 1500)
    y = model(x)
    print(f"\nInput: {x.shape}")
    print(f"Output: {y.shape}")
    
    # Test freezing
    print("\n" + "="*50)
    model.freeze_cnn()
    total, frozen = count_parameters(model)
    print("After freezing CNN:")
    print(f"  Trainable: {total:,}")
    print(f"  Frozen: {frozen:,}")
    
    # Test parameter groups
    print("\n" + "="*50)
    model.unfreeze_cnn()
    print("\nParameter groups:")
    param_groups = model.get_param_groups(cnn_lr=1e-4, rnn_lr=1e-5)
    for group in param_groups:
        n_params = sum(p.numel() for p in group['params'])
        print(f"  {group['name']}: {n_params:,} params, LR={group['lr']}")
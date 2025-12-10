"""
Model: CNN-LSTM Bidirectional w/ Optimized Hyperparameters

- Longer sequences (24 timesteps)
- Low dropout (0.1)
- Single BiLSTM layer
- Differential learning rates supported

Usage:
    python train.py --model cnn_bilstm_seq24 --seed 42
    python train.py --model cnn_bilstm_seq24 --lr 1e-4 --rnn_lr 1e-5 --seed 42
"""

import torch
import torch.nn as nn


class CNNBiLSTMSeq24(nn.Module):
    """Optimized CNN-LSTM with hyperparameters tuned for better performance."""
    
    def __init__(self, input_size=1500, num_classes=4,
                 cnn_channels=[32, 64, 128],
                 lstm_hidden=128,  # Increased from 64
                 dropout=0.1,      # Decreased from 0.2
                 target_seq_len=24,  # Increased from 16
                 bidirectional=True):  # New: bidirectional
        super(CNNBiLSTMSeq24, self).__init__()
        
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
        
        # Target sequence length
        self.target_seq_len = target_seq_len
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=cnn_channels[2],
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
            dropout=0,
            bidirectional=bidirectional
        )
        
        # Classifier (note: lstm_hidden * 2 if bidirectional)
        lstm_output_size = lstm_hidden * 2 if bidirectional else lstm_hidden
        self.fc = nn.Sequential(
            nn.Linear(lstm_output_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):        
        # CNN: (batch, 1500) -> (batch, 1, 1500) -> (batch, 128, 187)
        x = x.unsqueeze(1)
        x = self.cnn(x)
        
        # Reduce sequence length (batch, 128, 187) -> (batch, 128, 24)
        x = torch.nn.functional.interpolate(x, size=self.target_seq_len, mode='linear', align_corners=False)
        
        # Prepare for LSTM: (batch, 128, 24) -> (batch, 24, 128)
        x = x.permute(0, 2, 1)
        
        # Bidirectional LSTM: (batch, 24, 128) -> (batch, 24, 256)
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # For bidirectional, concatenate final forward and backward states
        if self.bidirectional:
            # h_n shape: (2, batch, 128) for bidirectional
            features = torch.cat([h_n[0], h_n[1]], dim=1)  # (batch, 256)
        else:
            features = h_n[-1]  # (batch, 128)
        
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
    model = CNNBiLSTMSeq24()
    total, frozen = count_parameters(model)
    print("CNNBiLSTMSeq24 (Optimized)")
    print(f"  Trainable parameters: {total:,}")
    print(f"  Frozen parameters: {frozen:,}")
    
    # Test forward pass
    x = torch.randn(8, 1500)
    y = model(x)
    print(f"\nInput: {x.shape}")
    print(f"Output: {y.shape}")
    
    # Show parameter groups
    print("\nParameter groups for differential LR:")
    param_groups = model.get_param_groups(cnn_lr=1e-4, rnn_lr=1e-5)
    for group in param_groups:
        n_params = sum(p.numel() for p in group['params'])
        print(f"  {group['name']}: {n_params:,} params, LR={group['lr']}")
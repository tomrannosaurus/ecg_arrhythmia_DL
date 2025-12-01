"""
Model: CNN-LSTM Mean Pooling Variant

Different approach from V2:
1. Uses MEAN of all LSTM outputs (not just last state)
   - Captures information from entire sequence
   - More robust than single timestep
2. Moderate sequence length: 20 timesteps 
3. Moderate LSTM size: 96 units 
4. Unidirectional LSTM

Usage:
    python train.py --model cnn_lstm_meanpool --seed 42
    python train.py --model cnn_lstm_meanpool --lr 1e-4 --lstm_lr 1e-5 --seed 42
"""

import torch
import torch.nn as nn


class CNNLSTMMeanPool(nn.Module):
    """CNN-LSTM with mean pooling over LSTM outputs."""
    
    def __init__(self, input_size=1500, num_classes=4,
                 cnn_channels=[32, 64, 128],
                 lstm_hidden=96,  # Between V1 and V2
                 dropout=0.15,     # Between V1 (0.2) and V2 (0.1)
                 target_seq_len=20):  # Between V1 (16) and V2 (24)
        super(CNNLSTMMeanPool, self).__init__()
        
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
        
        # LayerNorm for LSTM input
        self.layer_norm = nn.LayerNorm(cnn_channels[2])
        
        # LSTM
        self.lstm = nn.LSTM(
            input_size=cnn_channels[2],
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
            dropout=0
        )
        
        # Classifier
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden, 48),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(48, num_classes)
        )
    
    def forward(self, x):
        batch_size = x.size(0)
        
        # CNN: (batch, 1500) -> (batch, 1, 1500) -> (batch, 128, 187)
        x = x.unsqueeze(1)
        x = self.cnn(x)
        
        # Reduce sequence length
        x = torch.nn.functional.interpolate(x, size=self.target_seq_len, mode='linear', align_corners=False)
        
        # Prepare for LSTM: (batch, 128, seq) -> (batch, seq, 128)
        x = x.permute(0, 2, 1)
        
        # Normalize
        x = self.layer_norm(x)
        
        # LSTM: (batch, seq, 128) -> (batch, seq, lstm_hidden)
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # KEY DIFFERENCE: Use MEAN of all outputs instead of just last
        # This captures information from entire sequence more robustly
        features = lstm_out.mean(dim=1)  # (batch, lstm_hidden)
        
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
    
    def get_param_groups(self, cnn_lr, lstm_lr, fc_lr=None):
        """Return parameter groups for differential learning rates."""
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
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    return total, frozen


if __name__ == "__main__":
    # Test model
    model = CNNLSTMMeanPool()
    total, frozen = count_parameters(model)
    print(f"CNNLSTMMeanPool (Mean Pooling)")
    print(f"  Trainable parameters: {total:,}")
    
    # Test forward pass
    x = torch.randn(8, 1500)
    y = model(x)
    print(f"\nInput: {x.shape}")
    print(f"Output: {y.shape}")
    
    # Show parameter groups
    print("\nParameter groups:")
    param_groups = model.get_param_groups(cnn_lr=1e-4, lstm_lr=1e-5)
    for group in param_groups:
        n_params = sum(p.numel() for p in group['params'])
        print(f"  {group['name']}: {n_params:,} params, LR={group['lr']}")
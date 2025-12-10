"""
Model: CNN-LSTM Simple (Improved Architecture, Smaller Capacity)

Similar to seq16, but with reduced capacity:
- Smaller LSTM (32 units instead of 64)
- Shorter sequences (16 timesteps)
- Good for testing if model is too complex

Usage:
    python train.py --model simple_lstm --seed 42
"""

import torch
import torch.nn as nn


class CNNSimpleLSTM(nn.Module):
    """Simplified CNN-LSTM with reduced capacity."""
    
    def __init__(self, input_size=1500, num_classes=4,
                 cnn_channels=[32, 64, 128],
                 lstm_hidden=32,  # SMALLER than base model
                 dropout=0.2,
                 target_seq_len=16):  # SHORT sequences
        super(CNNSimpleLSTM, self).__init__()
        
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
        
        # Simple LSTM (smaller)
        self.lstm = nn.LSTM(
            input_size=cnn_channels[2],
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
            dropout=0
        )
        
        # Classifier (smaller)
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, num_classes)
        )
    
    def forward(self, x):
        """
        Args:
            x: (batch, 1500)
        Returns:
            (batch, num_classes)
        """
        
        # CNN
        x = x.unsqueeze(1)
        x = self.cnn(x)
        
        # Reduce sequence length
        x = torch.nn.functional.interpolate(
            x, size=self.target_seq_len, 
            mode='linear', align_corners=False
        )
        
        # Prepare for LSTM
        x = x.permute(0, 2, 1)
        
        # LSTM
        lstm_out, (h_n, c_n) = self.lstm(x)
        features = h_n[-1]
        
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
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = CNNSimpleLSTM()
    print(f"CNNSimpleLSTM parameters: {count_parameters(model):,}")
    
    # Test forward pass
    x = torch.randn(8, 1500)
    y = model(x)
    print(f"Input: {x.shape} -> Output: {y.shape}")
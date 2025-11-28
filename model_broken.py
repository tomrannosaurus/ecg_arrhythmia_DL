import torch
import torch.nn as nn


class CNNLSTM(nn.Module):
    """Serial CNN-LSTM model for ECG classification.
    
    Architecture:
        Input (batch, 1500) -> CNN feature extraction -> 
        LSTM temporal modeling -> FC classifier -> Output (batch, 4)
    """
    
    def __init__(self, input_size=1500, num_classes=4, 
                 cnn_channels=[32, 64, 128], 
                 lstm_hidden=128, lstm_layers=2,
                 dropout=0.5):
        """
        Args:
            input_size: length of ECG segment (1500 for 5 sec @ 300 Hz)
            num_classes: number of output classes (4)
            cnn_channels: list of CNN channel sizes
            lstm_hidden: LSTM hidden size
            lstm_layers: number of LSTM layers
            dropout: dropout probability
        """
        super(CNNLSTM, self).__init__()
        
        # CNN feature extractor
        self.cnn = nn.Sequential(
            # Conv block 1
            nn.Conv1d(1, cnn_channels[0], kernel_size=7, padding=3),
            nn.BatchNorm1d(cnn_channels[0]),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),
            
            # Conv block 2
            nn.Conv1d(cnn_channels[0], cnn_channels[1], kernel_size=5, padding=2),
            nn.BatchNorm1d(cnn_channels[1]),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),
            
            # Conv block 3
            nn.Conv1d(cnn_channels[1], cnn_channels[2], kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_channels[2]),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),
        )
        
        # Calculate CNN output size
        self.cnn_output_size = input_size // 8  # Three MaxPool2d layers
        
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
        
        # LSTM: (batch, seq_len, 128) -> (batch, seq_len, lstm_hidden)
        x, (h_n, c_n) = self.lstm(x)
        
        # Use last hidden state: (batch, lstm_hidden)
        x = h_n[-1]
        
        # Classifier: (batch, lstm_hidden) -> (batch, 4)
        x = self.fc(x)
        
        return x


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test model
    model = CNNLSTM()
    print(f"Model parameters: {count_parameters(model):,}")
    
    # Test forward pass
    x = torch.randn(8, 1500)  # batch of 8 samples
    y = model(x)
    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {y.shape}")

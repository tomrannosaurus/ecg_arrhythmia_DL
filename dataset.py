import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path


class ECGDataset(Dataset):
    """PyTorch Dataset for ECG segments."""
    
    def __init__(self, X, y):
        """
        Args:
            X: numpy array of shape (n_samples, 1500)
            y: numpy array of shape (n_samples,)
        """
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def load_splits(split_dir="data/splits"):
    """Load train/val/test splits and class weights."""
    split_dir = Path(split_dir)
    
    X_train = np.load(split_dir / "X_train.npy")
    X_val = np.load(split_dir / "X_val.npy")
    X_test = np.load(split_dir / "X_test.npy")
    y_train = np.load(split_dir / "y_train.npy")
    y_val = np.load(split_dir / "y_val.npy")
    y_test = np.load(split_dir / "y_test.npy")
    class_weights = np.load(split_dir / "class_weights.npy")
    
    return X_train, X_val, X_test, y_train, y_val, y_test, class_weights


def get_dataloaders(batch_size=32, split_dir="data/splits"):
    """Create train/val/test DataLoaders."""
    X_train, X_val, X_test, y_train, y_val, y_test, class_weights = load_splits(split_dir)
    
    train_dataset = ECGDataset(X_train, y_train)
    val_dataset = ECGDataset(X_val, y_val)
    test_dataset = ECGDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    class_weights = torch.FloatTensor(class_weights)
    
    return train_loader, val_loader, test_loader, class_weights


if __name__ == "__main__":
    # Test the DataLoader
    train_loader, val_loader, test_loader, weights = get_dataloaders(batch_size=32)
    
    print("DataLoader test:")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    print(f"Class weights: {weights}")
    
    # Test one batch
    X_batch, y_batch = next(iter(train_loader))
    print(f"\nBatch shapes:")
    print(f"X: {X_batch.shape}")
    print(f"y: {y_batch.shape}")

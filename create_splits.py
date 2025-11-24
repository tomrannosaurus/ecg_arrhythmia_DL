import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split


def create_splits(X, y, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    """Create stratified train/val/test splits."""
    # First split: test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_ratio, stratify=y, random_state=seed
    )
    
    # Second split: train and val
    val_ratio_adj = val_ratio / (train_ratio + val_ratio)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio_adj, stratify=y_temp, random_state=seed
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def calculate_class_weights(y_train):
    """Calculate class weights for imbalanced data."""
    counts = np.bincount(y_train)
    weights = len(y_train) / (len(counts) * counts)
    return weights


def print_stats(y_train, y_val, y_test):
    """Print split statistics."""
    classes = ["Normal", "AF", "Other", "Noisy"]
    total = len(y_train) + len(y_val) + len(y_test)
    
    print(f"\nTotal: {total}")
    print(f"Train: {len(y_train)} ({100*len(y_train)/total:.1f}%)")
    print(f"Val:   {len(y_val)} ({100*len(y_val)/total:.1f}%)")
    print(f"Test:  {len(y_test)} ({100*len(y_test)/total:.1f}%)")
    
    print("\nClass distribution:")
    for i, name in enumerate(classes):
        train_n = np.sum(y_train == i)
        val_n = np.sum(y_val == i)
        test_n = np.sum(y_test == i)
        print(f"{name}: {train_n} / {val_n} / {test_n}")


def main(processed_dir="data/processed", split_dir="data/splits", seed=42):
    """
    Create train/val/test splits from processed data.
    
    Args:
        processed_dir: Directory containing X.npy and y.npy (default: "data/processed")
        split_dir: Output directory for splits (default: "data/splits")
        seed: Random seed for reproducibility (default: 42)
    """
    processed_dir = Path(processed_dir)
    split_dir = Path(split_dir)
    
    # Load preprocessed data
    X = np.load(processed_dir / "X.npy")
    y = np.load(processed_dir / "y.npy")
    
    print(f"Loaded: X {X.shape}, y {y.shape}")
    
    # Create splits
    X_train, X_val, X_test, y_train, y_val, y_test = create_splits(X, y, seed=seed)
    
    # Calculate class weights
    weights = calculate_class_weights(y_train)
    
    # Print stats
    print_stats(y_train, y_val, y_test)
    print(f"\nClass weights: {weights}")
    
    # Save splits
    split_dir.mkdir(parents=True, exist_ok=True)
    
    np.save(split_dir / "X_train.npy", X_train)
    np.save(split_dir / "X_val.npy", X_val)
    np.save(split_dir / "X_test.npy", X_test)
    np.save(split_dir / "y_train.npy", y_train)
    np.save(split_dir / "y_val.npy", y_val)
    np.save(split_dir / "y_test.npy", y_test)
    np.save(split_dir / "class_weights.npy", weights)
    
    print(f"\nSaved to {split_dir}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Create train/val/test splits')
    parser.add_argument('--processed_dir', type=str, default='data/processed', help='Input directory')
    parser.add_argument('--split_dir', type=str, default='data/splits', help='Output directory')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()
    
    main(processed_dir=args.processed_dir, split_dir=args.split_dir, seed=args.seed)

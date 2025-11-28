import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split


def create_splits(X, y, recording_ids, train_ratio=0.7, val_ratio=0.15, 
                                   test_ratio=0.15, seed=42):
    """Create stratified train/val/test splits at the recording level.
    
    BUGFIX: Splits recordings (not segments) to prevent data leakage.
    All segments from a recording go to the same split.
        
    Args:
        X: Segment features (n_segments, segment_length)
        y: Segment labels (n_segments,)
        recording_ids: Recording index for each segment (n_segments,)
        train_ratio: Proportion for training (default: 0.7)
        val_ratio: Proportion for validation (default: 0.15)
        test_ratio: Proportion for test (default: 0.15)
        seed: Random seed for reproducibility (default: 42)
    
    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test
    """
    # unique recordings and their labels
    unique_recordings = np.unique(recording_ids)
    n_recordings = len(unique_recordings)
    
    # for each recording, get its label (all segments have same label)
    recording_labels = np.array([y[recording_ids == rec_id][0] for rec_id in unique_recordings])
    
    print("\nSplitting at individual recording level:")
    print(f" Total recordings: {n_recordings}")
    print(f" Total segments: {len(X)}")
    
    # split recordings into train/val/test
    # first split: separate test set
    train_val_recs, test_recs = train_test_split(
        unique_recordings,
        test_size=test_ratio,
        stratify=recording_labels,
        random_state=seed
    )
    
    # second split: separate train and val
    train_val_labels = np.array([recording_labels[np.where(unique_recordings == r)[0][0]] 
                                  for r in train_val_recs])
    val_ratio_adj = val_ratio / (train_ratio + val_ratio)
    
    train_recs, val_recs = train_test_split(
        train_val_recs,
        test_size=val_ratio_adj,
        stratify=train_val_labels,
        random_state=seed
    )
    
    print("\nRecording splits:")
    print(f"  Train: {len(train_recs)} recordings ({100*len(train_recs)/n_recordings:.1f}%)")
    print(f"  Val:   {len(val_recs)} recordings ({100*len(val_recs)/n_recordings:.1f}%)")
    print(f"  Test:  {len(test_recs)} recordings ({100*len(test_recs)/n_recordings:.1f}%)")
    
    # convert to sets for fast lookup
    train_recs_set = set(train_recs)
    val_recs_set = set(val_recs)
    test_recs_set = set(test_recs)
    
    # map segments to splits based on their parent recording
    train_mask = np.isin(recording_ids, list(train_recs_set))
    val_mask = np.isin(recording_ids, list(val_recs_set))
    test_mask = np.isin(recording_ids, list(test_recs_set))
    
    X_train = X[train_mask]
    X_val = X[val_mask]
    X_test = X[test_mask]
    
    y_train = y[train_mask]
    y_val = y[val_mask]
    y_test = y[test_mask]
    
    print("\nSegment splits:")
    print(f"  Train: {len(X_train)} segments ({100*len(X_train)/len(X):.1f}%)")
    print(f"  Val:   {len(X_val)} segments ({100*len(X_val)/len(X):.1f}%)")
    print(f"  Test:  {len(X_test)} segments ({100*len(X_test)/len(X):.1f}%)")
    
    # verify no leakage
    assert len(train_recs_set & val_recs_set) == 0, "LEAKAGE! Train-Val overlap!"
    assert len(train_recs_set & test_recs_set) == 0, "LEAKAGE! Train-Test overlap!"
    assert len(val_recs_set & test_recs_set) == 0, "LEAKAGE! Val-Test overlap!"
    print("\n Verified: No data leakage (all recordings isolated to single split)")
    
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
    
    print(f"\nTotal segments: {total}")
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
    
    BUGFIX: Now uses recording_ids.npy to split at recording level.

    Args:
        processed_dir: Directory containing X.npy, y.npy, and recording_ids.npy
        split_dir: Output directory for splits (default: "data/splits")
        seed: Random seed for reproducibility (default: 42)
    """
    processed_dir = Path(processed_dir)
    split_dir = Path(split_dir)
    
    # load (preprocessed) data
    X = np.load(processed_dir / "X.npy")
    y = np.load(processed_dir / "y.npy")
    
    # CRITICAL: Load recording IDs to prevent data leakage
    recording_ids_path = processed_dir / "recording_ids.npy"
    if not recording_ids_path.exists():
        raise FileNotFoundError(
            f"\n{'='*70}\n"
            f"ERROR: {recording_ids_path} not found!\n"
            f"\nThis means you're using OLD preprocessed data that will cause data leakage.\n"
            f"\nYou MUST re-run preprocessing to generate recording_ids.npy:\n"
            f"  python preprocessing.py\n"
            f"\nThen re-run this script:\n"
            f"  python create_splits.py\n"
            f"{'='*70}"
        )
    
    recording_ids = np.load(recording_ids_path)
    
    print(f"Loaded: X {X.shape}, y {y.shape}, recording_ids {recording_ids.shape}")
    
    # splits at recording level
    X_train, X_val, X_test, y_train, y_val, y_test = create_splits(
        X, y, recording_ids, seed=seed
    )
    
    # Calc class weights
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
    
    print(f"\nSaved splits to: {split_dir}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Create train/val/test splits')
    parser.add_argument('--processed_dir', type=str, default='data/processed', 
                       help='Input directory')
    parser.add_argument('--split_dir', type=str, default='data/splits', 
                       help='Output directory')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()
    
    main(processed_dir=args.processed_dir, split_dir=args.split_dir, seed=args.seed)
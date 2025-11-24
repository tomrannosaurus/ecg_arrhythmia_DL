import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.signal import butter, filtfilt
from pathlib import Path
from tqdm import tqdm
import urllib.request
import zipfile


def download_data(data_dir="data"):
    """Download PhysioNet 2017 dataset."""
    data_dir = Path(data_dir)
    data_dir.mkdir(exist_ok=True)
    
    url = "https://physionet.org/files/challenge-2017/1.0.0/training2017.zip"
    zip_path = data_dir / "training2017.zip"
    extract_dir = data_dir / "training2017"
    
    if extract_dir.exists():
        print("Data already downloaded")
        return
    
    if not zip_path.exists():
        print("Downloading dataset (500MB)...")
        urllib.request.urlretrieve(url, zip_path)
    
    print("Extracting...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(data_dir)
    print("Done")


def bandpass(signal, fs=300, low=0.5, high=40, order=3):
    """Apply Butterworth bandpass filter."""
    b, a = butter(order, [low / (fs / 2), high / (fs / 2)], btype="band")
    return filtfilt(b, a, signal)


def normalize(signal):
    """Z-score normalization."""
    return (signal - np.mean(signal)) / np.std(signal)


def segment_signal(signal, segment_len, overlap):
    """Segment ECG signal into overlapping fixed-length windows.
    
    TODO: Future option to segment by RR intervals instead of fixed time.
    Would require external peak detection (e.g., Pan-Tompkins algorithm).
    """
    step = int(segment_len * (1 - overlap))
    for start in range(0, len(signal) - segment_len + 1, step):
        yield signal[start:start + segment_len]


def preprocess_dataset(data_dir, save_dir, fs=300, segment_sec=5, overlap=0.5):
    """Load all .mat ECG files, preprocess, and save fixed-length segments."""
    segment_len = fs * segment_sec
    save_dir.mkdir(parents=True, exist_ok=True)

    ref = pd.read_csv(data_dir / "REFERENCE.csv", header=None, names=["record", "label"])
    label_map = {"N": 0, "A": 1, "O": 2, "~": 3}

    X, y = [], []

    for _, row in tqdm(ref.iterrows(), total=len(ref), desc="Processing"):
        rec, label = row["record"], row["label"]
        sig = sio.loadmat(data_dir / f"{rec}.mat")["val"][0]
        sig = bandpass(sig, fs)
        sig = normalize(sig)

        for seg in segment_signal(sig, segment_len, overlap):
            X.append(seg.astype(np.float32))
            y.append(label_map[label])

    X = np.stack(X)
    y = np.array(y)

    np.save(save_dir / "X.npy", X)
    np.save(save_dir / "y.npy", y)

    print(f"Saved to {save_dir}")
    print(f"X: {X.shape}, y: {y.shape}")


def main(segment_sec=5, save_dir="data/processed"):
    """
    Main preprocessing pipeline.
    
    Args:
        segment_sec: Segment length in seconds (default: 5)
        save_dir: Output directory for processed data (default: "data/processed")
    """
    download_data()
    data_dir = Path("data/training2017")
    save_dir = Path(save_dir)
    preprocess_dataset(data_dir, save_dir, segment_sec=segment_sec)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Preprocess ECG data')
    parser.add_argument('--segment_sec', type=int, default=5, help='Segment length in seconds')
    parser.add_argument('--save_dir', type=str, default='data/processed', help='Output directory')
    args = parser.parse_args()
    
    main(segment_sec=args.segment_sec, save_dir=args.save_dir)

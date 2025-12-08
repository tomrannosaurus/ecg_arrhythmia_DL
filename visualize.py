import pandas as pd
import scipy.io as sio
import matplotlib.pyplot as plt
from pathlib import Path
from preprocessing import bandpass, normalize


def visualize_preprocessing(data_dir="data/training2017", num_examples=4, save_path="preprocessing_check.png"):
    """Visualize raw vs processed ECG signals."""
    data_dir = Path(data_dir)
    
    # Load reference file
    ref = pd.read_csv(data_dir / "REFERENCE.csv", header=None, names=["record", "label"])
    
    # Create figure
    fig, axes = plt.subplots(num_examples, 2, figsize=(14, 3 * num_examples))
    
    for i in range(num_examples):
        rec = ref.iloc[i, 0]
        label = ref.iloc[i, 1]
        
        # Load signal
        sig = sio.loadmat(data_dir / f"{rec}.mat")["val"][0]
        
        # Show first 3 seconds (900 samples at 300 Hz)
        samples = 900
        
        # Left: raw signal
        axes[i, 0].plot(sig[:samples])
        axes[i, 0].set_title(f"{rec} (Class {label}) - Raw")
        axes[i, 0].set_ylabel("Amplitude")
        axes[i, 0].grid(alpha=0.3)
        
        # Right: processed signal
        sig_proc = bandpass(sig, fs=300)
        sig_proc = normalize(sig_proc)
        axes[i, 1].plot(sig_proc[:samples])
        axes[i, 1].set_title(f"{rec} (Class {label}) - Processed")
        axes[i, 1].set_ylabel("Normalized")
        axes[i, 1].grid(alpha=0.3)
        
        # X labels on bottom row
        if i == num_examples - 1:
            axes[i, 0].set_xlabel("Sample")
            axes[i, 1].set_xlabel("Sample")
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved to {save_path}")
    plt.close()


if __name__ == "__main__":
    visualize_preprocessing()

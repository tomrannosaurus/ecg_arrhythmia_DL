import streamlit as st
import os
from pathlib import Path
import matplotlib.pyplot as plt
from grad_cam_ecg import run_gradcam_analysis

CHECKPOINT_DIR = "checkpoints"

def list_checkpoint_files():
    """Return list of .pt/.pth files in checkpoint directory."""
    p = Path(CHECKPOINT_DIR)
    if not p.exists():
        return []
    return [f for f in p.iterdir() if f.suffix in [".pt", ".pth"]]

def main():
    st.title("ECG Grad-CAM Visualization Tool")

    st.write("### Select Model Checkpoint")
    checkpoints = list_checkpoint_files()

    if not checkpoints:
        st.warning("No model checkpoints found in `checkpoints/` folder.")
        st.stop()

    checkpoint_file = st.selectbox(
        "Choose a model checkpoint:",
        checkpoints,
        format_func=lambda p: p.name
    )

    st.write("### Select Settings")

    num_samples = st.slider("Number of samples per class", 1, 20, 5)

    target_class = st.selectbox(
        "Target class (optional)",
        ["Auto / predicted", "Normal (0)", "AF (1)", "Other (2)", "Noisy (3)"]
    )

    class_map = {
        "Auto / predicted": None,
        "Normal (0)": 0,
        "AF (1)": 1,
        "Other (2)": 2,
        "Noisy (3)": 3
    }
    target_class_val = class_map[target_class]

    st.write("### Run Grad-CAM")
    if st.button("Generate Visualization"):
        st.write("Running analysis... please wait.")

        result = run_gradcam_analysis(
            checkpoint_path=str(checkpoint_file),
            model_name=None,
            num_samples=num_samples,
            target_class=target_class_val,
            split_dir="data/splits",
            output_dir="figures/gradcam",
            dataset_split="test"
        )

        img_path = "figures/gradcam/gradcam_class_comparison.png"
        if os.path.exists(img_path):
            st.image(img_path, caption="Grad-CAM Visualization", use_column_width=True)
        else:
            st.error("Expected output image not found!")

        st.success("Analysis complete.")

if __name__ == "__main__":
    main()
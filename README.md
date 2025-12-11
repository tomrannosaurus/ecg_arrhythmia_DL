# Deep Learning for ECG Arrhythmia Classification

**CS/DS 541 (Fall 2025)**  
**Team:** Ananya Shukla, Tom Arnold, Nate Hindman  
**Institution:** Worcester Polytechnic Institute

---

## Project Status

**Current Phase:** Final deliverables (paper writing and presentation preparation)  
**Experimental Work:** Complete  
**Paper Deadline:** December 11, 2025  
**Presentation Date:** December 1, 2025

---

## Key Results

### Best Performing Models (CUDA, from results_table.csv)

| Model | Test F1 | Test AUROC | Test Accuracy | Parameters |
|-------|---------|------------|---------------|------------|
| **CNN-LSTM** | **0.756** | **0.845** | **0.763** | 308,420 |
| CNN-LSTM (seed 123) | 0.755 | 0.840 | 0.756 | 308,420 |
| CNN-LSTM (seed 1) | 0.750 | 0.838 | 0.754 | 308,420 |
| CNN-LSTM-LN | 0.747 | 0.838 | 0.748 | 308,676 |

**Key Finding:** After resolving hardware-specific training failures, the CNN-LSTM hybrid architecture achieved strong performance (F1=0.756, AUROC=0.845), demonstrating that serial CNN to LSTM architectures can effectively capture both spatial and temporal patterns in ECG signals when properly implemented.

---

## Major Discovery: PyTorch MPS Backend Bug

During development, we discovered a critical bug in PyTorch's MPS (Metal Performance Shaders) backend on Apple Silicon that caused complete LSTM learning failure.

**Symptom:** LSTM models trained on MPS showed learning collapse (F1=0.14) while CNN-only models achieved reasonable performance (F1=0.61).

**Root Cause:** Silent gradient failures in MPS kernel operations (addcmul_, addcdiv_) prevented gradients from flowing correctly through LSTM gates during backpropagation.

**Solution:** Re-running all experiments on CUDA revealed the original CNN-LSTM architecture trained successfully, achieving F1=0.756 and AUROC=0.845.

**Impact:** All hybrid models required complete re-evaluation on CUDA hardware. This finding emphasizes the importance of validating deep learning failures across multiple hardware backends before attributing them to architectural issues.

**References:**
- PyTorch Issue [#82707](https://github.com/pytorch/pytorch/issues/82707)
- Blog post: ["The bug that taught me more about PyTorch than years of using it"](https://elanapearl.github.io/blog/2025/the-bug-that-taught-me-pytorch/)

---

## 📁 Project Overview

This project implements hybrid CNN-RNN architectures for automated ECG arrhythmia classification using the **PhysioNet 2017 Challenge** dataset. We classify single-lead ECG recordings into four categories:
- Normal rhythm
- Atrial Fibrillation (AF)
- Other arrhythmias
- Noisy recordings

---

## Dataset and Preprocessing

**Dataset:** PhysioNet Computing in Cardiology Challenge 2017
- 8,528 single-lead ECG recordings (300 Hz sampling rate)
- Recording lengths: 9-60 seconds (variable)
- Severe class imbalance:
  - Normal: 58%
  - Other: 31%
  - Atrial Fibrillation: 9%
  - Noisy: 2%

**Preprocessing Pipeline:**
1. **Filtering:** 0.5-40 Hz Butterworth bandpass filter (removes baseline wander and high-frequency noise)
2. **Normalization:** Z-score normalization per recording
3. **Segmentation:** 5-second windows with 50% overlap → 71,040 segments (1,500 samples each)
4. **Splitting:** Stratified recording-level splits (prevents data leakage)
5. **Class Balancing:** Weighted cross-entropy loss (weight ratio 24:1)

---

## Model Architectures Explored

### 1. CNN-LSTM (308,420 parameters) **Best Model**
Serial CNN to LSTM architecture with standard 2-layer LSTM. CNN extracts features → LSTM models temporal dependencies → FC classifier.  
**Implementation:** model_cnn_lstm.py  
**Test Performance:** F1=0.756, AUROC=0.845, Accuracy=0.763

### 2. CNN-LSTM-LN (308,676 parameters)
CNN-LSTM with LayerNorm after LSTM output for improved gradient flow.  
**Implementation:** model_cnn_lstm_ln.py  
**Test Performance:** F1=0.747, AUROC=0.838

### 3. CNN-only Baseline (44,228 parameters)
Pure convolutional architecture without temporal modeling. 3 conv blocks → adaptive avg pool → FC classifier.  
**Implementation:** model_cnn_only.py

### 4. Bidirectional LSTM (316,420 parameters)
CNN with BiLSTM that processes temporal sequences in both forward and backward directions.  
**Implementation:** model_bilstm.py

### 5. CNN-BiLSTM-Seq24
CNN with BiLSTM using reduced sequence length (24 timesteps) for computational efficiency.  
**Implementation:** model_cnn_bilstm_seq24.py

### 6. GRU Variant (105,460 parameters)
CNN with GRU cells instead of LSTM for simpler gating mechanism and computational efficiency.  
**Implementation:** model_gru.py

### 7. CNN-LSTM with Attention
Attention mechanism weights LSTM outputs before classification, focusing on diagnostically important temporal regions.  
**Implementation:** model_attention.py

### 8. CNN-LSTM with Residual Connections
Residual connection around LSTM allows direct gradient flow from CNN to classifier, helping with vanishing gradients.  
**Implementation:** model_residual.py

### 9. CNN-LSTM-MeanPool
Uses mean pooling over all LSTM outputs instead of only the final hidden state.  
**Implementation:** model_cnn_lstm_meanpool.py

### 10. CNN-LSTM-Seq16
CNN-LSTM with reduced sequence length (16 timesteps) for faster training and reduced memory usage.  
**Implementation:** model_cnn_lstm_seq16.py

### 11. Simple LSTM
Simplified LSTM architecture with fewer layers (1 layer) and reduced hidden units (64) for diagnostic comparison.  
**Implementation:** model_simple_lstm.py

### 12. LSTM-only (diagnostic)
Pure LSTM architecture without CNN feature extraction to isolate LSTM learning capabilities.  
**Implementation:** model_lstm_only.py

---

## Training Configuration

**Optimized Hyperparameters:**
- Learning Rate: 1e-4 (Adam optimizer)
- Batch Size: 64
- Weight Decay: 1e-4
- Gradient Clipping: 1.0
- Early Stopping: Patience = 10 epochs
- Max Epochs: 50

**Loss Function:** Weighted Cross-Entropy (addresses 24:1 class imbalance)

**Evaluation Metrics:**
- F1-score (weighted, primary metric)
- AUROC (one-vs-rest)
- Accuracy
- Per-class sensitivity and specificity

---

## Quick Start

### Installation

**Option 1: Using pip**
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

**Option 2: Using uv (faster)**
```bash
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -r requirements.txt
```

### Data Preparation
```bash
# 1. Download and preprocess
python preprocessing.py

# 2. Create splits (recording-level, prevents leakage)
python create_splits.py
```

### Training
```bash
# Train CNN-LSTM (best model)
python train.py --model cnn_lstm --lr 0.0001 --batch-size 64 --seed 42

# Train CNN-only baseline
python train.py --model cnn_only --lr 0.0001 --batch-size 64 --seed 42

# Train GRU variant
python train.py --model gru --lr 0.0001 --batch-size 64 --seed 42
```

### Visualization
```bash
# Plot training curves
python plot_training.py checkpoints/cnn_lstm_seed42_*_history.json

# Compare multiple models
python plot_training.py \
    checkpoints/cnn_only_seed42_*_history.json \
    checkpoints/cnn_lstm_seed42_*_history.json \
    --labels "CNN-only" "CNN-LSTM" \
    --compare
```

---

## Analysis Tools

The project includes comprehensive optimization analysis tools for systematic hyperparameter optimization and model comparison.

### Running Optimization Analysis
```bash
# Full analysis with all statistics
python analysis/main.py analyze --checkpoint_dir checkpoints

# Generate visualizations
python analysis/main.py visualize --checkpoint_dir checkpoints --output-dir figures

# Hyperparameter optimization
python analysis/main.py optimize --checkpoint_dir checkpoints --method greedy

# Generate text report
python analysis/main.py report --checkpoint_dir checkpoints --output report.txt
```

### Grad-CAM Interpretability
```bash
# Interactive Grad-CAM visualization
streamlit run grad_cam_app.py

# Command-line Grad-CAM analysis
python grad_cam_ecg.py --checkpoint checkpoints/cnn_lstm_seed42_*.pt --num-samples 10
```

---

## Experimental Results Summary

### Model Comparison (Test Set, CUDA)

| Architecture | F1 Score | AUROC | Accuracy | Params | Device |
|--------------|----------|-------|----------|--------|--------|
| CNN-LSTM | **0.756** | **0.845** | **0.763** | 308K | CUDA |
| CNN-LSTM-LN | 0.747 | 0.838 | 0.748 | 309K | CUDA |
| CNN-only (MPS) | 0.599 | 0.727 | 0.596 | 44K | MPS |

### Key Insights
1. **Serial CNN-LSTM architecture achieves strong performance:** Combining spatial feature extraction with temporal modeling yields F1=0.756 and AUROC=0.845
2. **Hardware backend significantly impacts training:** LSTM models fail silently on MPS but train successfully on CUDA
3. **Layer normalization provides modest improvements:** CNN-LSTM-LN achieves comparable performance to standard CNN-LSTM
4. **Class imbalance remains challenging:** All models show lower performance on minority classes (Atrial Fibrillation, Noisy)

---

## Project Status & Task Checklist

### Phase 1: Data and Baseline ✓
- [x] **[T1]** Download PhysioNet 2017 dataset (8,528 recordings, 300 Hz sampling)
- [x] **[T2]** Implement preprocessing pipeline
  - Bandpass filtering (0.5-40 Hz Butterworth)
  - Z-score normalization
  - Segmentation into 5-second windows with 50% overlap (71,040 segments total)
- [x] **[T3]** Create recording-level train/val/test splits (70/15/15%)
  - Prevents data leakage across splits
  - Stratified by class distribution
- [x] **[T4]** Implement baseline CNN-LSTM model (308,420 parameters)
- [x] **[T5]** Run baseline training with comprehensive metrics (F1, AUROC, sensitivity, specificity)
- [x] **[T6]** Submit Intermediate Progress Report (Nov 14 @ 8:00 AM)

### Phase 2: Optimization and Debugging ✓
- [x] **[T6.1]** Fix data leakage bug (proper recording-level splitting)
- [x] **[T6.2]** Add enhanced training logging (per-epoch loss, accuracy, F1, AUROC)
- [x] **[T7]** Implement class imbalance strategy
  - Weighted cross-entropy loss (24:1 weight ratio)
  - Addresses severe imbalance (58% Normal, 31% Other, 9% AF, 2% Noisy)
- [x] **[T9]** Add regularization (BatchNorm, Dropout) and hyperparameter tuning
  - Learning rate optimization (1e-4)
  - Batch size experiments (32, 64, 128)
  - Weight decay and gradient clipping

### Phase 3: Ablation Studies ✓
- [x] **[T10]** Run comprehensive architecture ablations
  - [x] **[T10.0]** CNN-only baseline (44,228 parameters) - F1=0.661 on MPS
  - [x] **[T10.5]** Bidirectional LSTM (316,420 parameters)
  - [x] **[T10.8]** Simplified LSTM (1 layer, 64 units)
  - [x] **[T10.9]** LSTM-only (diagnostic)
  - [x] **[T10.10]** Residual connections around LSTM
  - [x] **[T10.11]** GRU variant (105,460 parameters)
  - [x] **[T10.12]** Attention mechanism
  - [x] CNN-LSTM with LayerNorm
  - [x] Mean pooling over LSTM outputs (instead of final hidden state)
  - [x] Reduced sequence length variants (Seq16, Seq24)
- [x] **[T10.0]** Test segment length analysis (5s vs 10s vs 20s windows)
- [x] **[T10.4]** Test differential learning rates for CNN and RNN components
- [x] **Discover critical MPS backend bug** causing LSTM training failures on Apple Silicon (F1=0.14)
- [x] **Complete CUDA re-evaluation** of all models after MPS bug discovery
  - CNN-LSTM achieves F1=0.756, AUROC=0.845 on CUDA (vs F1=0.14 on MPS)
  - Validates architecture effectiveness across hardware backends

### Phase 4: Final Deliverables (Current) 🔄
- [x] **[T11]** Implement interpretability analysis
  - [x] Grad-CAM visualization for CNN feature importance
  - [ ] Attention weight visualization for temporal focus [TODO: MOVE THIS TO FUTURE WORK]
  - [ ] Case study analysis of model predictions [TODO: MOVE THIS TO FUTURE WORK]
- [x] **[T12]** Final model training and cross-validation
  - [x] Train on full dataset with best hyperparameters
  - [ ] K-fold cross-validation for robust performance estimates [TODO: MOVE THIS TO FUTURE WORK]
  - [x] Finalize results tables and statistical analysis
- [x] **[T13]** Prepare and deliver presentation (Dec 1, 2025)
  - [x] Build presentation slides with key results
  - [x] Rehearse timing and delivery
- [ ] **[T14]** Write final paper
  - [ ] Methods section (architecture, preprocessing, training)
  - [ ] Results section (tables, figures, statistical tests)
  - [ ] Discussion and interpretation
  - [ ] Limitations and future work
  - [x] References and citations
- [ ] **[T15]** Submit final paper (Dec 11, 2025) and archive GitHub release
  - [ ] Final proofreading and formatting
  - [ ] Tag GitHub release with code, configs, and trained models
  - [ ] Archive datasets and checkpoints

### Key Findings & Lessons Learned
- **MPS Backend Bug:** PyTorch MPS backend on Apple Silicon had silent gradient failures in LSTM operations (addcmul_, addcdiv_), causing complete learning collapse (F1=0.14 vs 0.756 on CUDA)
- **Best Architecture:** CNN-LSTM hybrid achieves F1=0.756, AUROC=0.845, demonstrating serial architectures effectively capture spatial and temporal ECG patterns
- **Class Imbalance Critical:** Weighted loss with 24:1 ratio essential for minority class detection (AF=9%, Noisy=2%)
- **Hardware Validation Important:** Always validate failures across multiple backends before attributing to architecture
- **Recording-Level Splits Essential:** Segment-level splits caused severe data leakage, inflating performance metrics

---

## Repository Structure

```
├── preprocessing.py              # Data download and preprocessing
├── create_splits.py              # Recording-level train/val/test splits
├── dataset.py                    # PyTorch Dataset and DataLoader
├── train.py                      # Training loop with comprehensive logging
├── plot_training.py              # Visualization utilities
├── search_runs.py                # Experiment search and comparison
├── results_table.py              # Generate results tables
├── grad_cam_ecg.py               # Grad-CAM interpretability analysis
├── grad_cam_app.py               # Streamlit app for Grad-CAM visualization
│
├── model_cnn_lstm.py             # CNN-LSTM architecture (best model)
├── model_cnn_only.py             # CNN baseline
├── model_bilstm.py               # Bidirectional LSTM
├── model_gru.py                  # GRU variant
├── model_cnn_lstm_ln.py          # CNN-LSTM with LayerNorm
├── model_simple_lstm.py          # Simplified LSTM
├── model_lstm_only.py            # LSTM only (diagnostic)
├── model_residual.py             # Residual connections
├── model_attention.py            # Attention mechanism
│
├── analysis/                     # Design of Experiments (DOE) analysis
│   ├── main.py                   # Main analysis script
│   ├── core.py                   # Data loading and feature engineering
│   ├── outputs.py                # Visualization and reporting
│   ├── optimize.py               # Hyperparameter optimization
│   └── stats.py                  # Statistical analysis functions
│
├── data/
│   ├── processed/
│   │   ├── X.npy                 # All segments (71,040, 1500)
│   │   ├── y.npy                 # Labels
│   │   └── recording_ids.npy     # Recording IDs (prevents leakage)
│   └── splits/
│       ├── X_train.npy, X_val.npy, X_test.npy
│       ├── y_train.npy, y_val.npy, y_test.npy
│       └── class_weights.npy
│
├── checkpoints/                  # Model weights and training logs
│   ├── {model}_{seed}_{timestamp}.pt
│   ├── {model}_{seed}_{timestamp}_history.json
│   └── {model}_{seed}_{timestamp}_results.json
│
├── logs/                         # Experiment logs
│   └── experiment_log_{timestamp}.txt
│
└── figures/                      # Training curves and visualizations
    ├── gradcam/                  # Grad-CAM visualization outputs
    └── [various training plots]
```

---

## Academic Context

**Course:** CS/DS 541 (Deep Learning)  
**Institution:** Worcester Polytechnic Institute  
**Semester:** Fall 2025  

**Deliverables:**
- Intermediate Progress Report: Submitted November 14, 2025
- Presentation: December 1 and 4, 2025
- Final Paper: December 11, 2025

**Project Goal:** Develop hybrid CNN-RNN architectures for automated ECG arrhythmia classification achieving greater than 80 percent F1-score on the PhysioNet 2017 dataset while implementing all components from scratch.

---

## References

### Datasets
1. PhysioNet Challenge 2017: AF Classification from Short Single-Lead ECG Recording. https://physionet.org/content/challenge-2017/1.0.0/

### Related Work
2. Sun J. Automatic cardiac arrhythmias classification using CNN and attention-based RNN network. Healthc Technol Lett. 2023 Apr 20;10(3):53-61.
3. Yang X, Yang H, Dou M. ADLNet: an adaptive network for arrhythmia classification based on deformable Convolution and LSTM. Signal Image Video Process. 2024.
4. Additional papers in project knowledge base (Scientific Reports, Sensors, HTL journals)

### Technical References
5. PyTorch MPS Backend Bug Documentation:
   - Issue 82707: "Bad performance metrics for BERT model training on MPS"
   - Forum Discussion: "MPS Crash - Assertion failed in _getLSTMGradKernelDAGObject"
   - Simon, E. (2025): "The bug that taught me more about PyTorch than years of using it"

---

## Lessons Learned

1. **Hardware backend validation is critical:** Apparent architectural failures may be hardware or backend-specific issues rather than fundamental design problems
2. **Recording-level splitting prevents data leakage:** Segment-level splitting causes overestimation of model performance due to temporal correlation between segments from the same recording
3. **Serial CNN to LSTM architectures are effective:** Combining spatial feature extraction with temporal sequence modeling achieves strong performance (F1=0.756)
4. **Class imbalance mitigation is challenging:** Despite weighted loss functions and careful sampling strategies, minority class performance remains a significant challenge
5. **Regularization and normalization matter:** Layer normalization and dropout significantly improve model generalization

---

## Future Work

- Multi-lead ECG analysis (12-lead ECG classification)
- Real-time inference optimization for clinical deployment
- Transformer-based temporal modules as alternatives to LSTM
- Attention visualization and interpretability analysis
- Clinical validation with cardiologist expertise
- Deployment as web service for real-time cardiac monitoring


---

## Contact

**Team Email:** tarnold@wpi.edu  
**Institution:** Worcester Polytechnic Institute

---

## License

This project is for academic purposes as part of CS/DS 541 coursework at Worcester Polytechnic Institute.

---

**Last Updated:** December 11, 2025  
**Status:** Near completion (final paper and presentation in progress)
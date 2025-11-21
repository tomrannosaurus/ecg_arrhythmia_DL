# Deep Learning for ECG Arrhythmia Classification
**CS/DS 541 – Fall 2025**  
**Team:** Ananya Shukla, Tom Arnold, Nate Hindman  

---

## Project Overview
This project implements a hybrid CNN–RNN model for ECG arrhythmia classification using the **PhysioNet 2017** dataset.  
Deliverables include an **Intermediate Progress Report (Nov 14 @ 8:00 AM)**, **Presentation (Dec 1 & 4)**, and **Final Paper (Dec 11)**.

---

## Quick Visual: Dependency Map (Mermaid)

```mermaid
flowchart TD
    %% Setup and Baseline
    T1([T1 Data download]) --> T2([T2 Preprocess / normalize / segment])
    T2 --> T3([T3 Train / val / test split])
    T3 --> T4([T4 Baseline CNN-LSTM model])
    T4 --> T5([T5 Baseline metrics])
    T5 --> T6([T6 Progress report Nov 14 at 8 AM])
    %% Refinement
    T5 --> T7([T7 Imbalance strategy])
    T5 --> T8([T8 Data augmentation])
    T7 --> T9([T9 Hyperparameter tuning / regularization])
    T8 --> T9([T9 Hyperparameter tuning / regularization])
    T9 --> T10([T10 Ablations: CNN-only / RNN-only / Hybrid / Variants])
    %% Interpretability and Final Model
    T10 --> T11([T11 Interpretability: Attention, CAM, Grad-CAM])
    T11 --> T12([T12 Final training and evaluation full data])
    %% Deliverables
    T12 --> T13([T13 Presentation slides Dec 1 and Dec 4])
    T12 --> T14([T14 Paper: methods, results, interpretation])
    T14 --> T15([T15 Submit paper Dec 11 + archive release])
```
---

## Tasks (Single-line, with dependencies)
- [ ] **[T1]** Download PhysioNet 2017 dataset and stage backup MIT-BIH (no deps)
- [ ] **[T2]** Implement preprocessing (bandpass, z-score), segmentation scripts **(dep: T1)**
- [ ] **[T3]** Produce train/val/test split with class stats **(dep: T2)**
- [ ] **[T4]** Implement baseline serial CNN→LSTM model in PyTorch **(dep: T3)**
- [ ] **[T5]** Run quick baseline training; compute F1/AUROC/sensitivity/specificity **(dep: T4)**
- [ ] **[T6]** Draft and submit Intermediate Progress Report (summary of setup/baseline) **(dep: T5) — due Nov 14 @ 8:00 AM**

- [ ] **[T7]** Choose and implement primary class-imbalance strategy (weighted loss / sampler / focal) **(dep: T5)**
- [ ] **[T8]** Implement ECG data augmentation (jitter, scaling, time-warp, cutout) **(dep: T5)**
- [ ] **[T9]** Add regularization (BN/Dropout) and run hyperparameter search **(dep: T7, T8)**
- [ ] **[T10]** Run ablations (CNN-only, RNN-only, hybrid; GRU/LSTM/Transformer variant) **(dep: T9)**
- *CNN-LSTM Debugging:* CNN-LSTM fails (F1=0.14, AUROC=0.50) vs CNN-only (F1=0.61, AUROC=0.73)
- [x] **[T10.0]** Test 20s segments -> Both models worse; segment length not the issue
- [ ] **[T10.1]** Check data leakage (recordings split properly) **(prof)**
- [ ] **[T10.2]** Add training accuracy/loss logging **(prof)**
- [ ] **[T10.3]** Verify label assignment per segment **(prof)**
- [ ] **[T10.4]** Test differential learning rates (CNN=1e-4, LSTM=1e-3)
- [ ] **[T10.5]** Test bidirectional LSTM **(prof)**
- [ ] **[T10.6]** Test transfer learning (pretrain CNN, freeze, train LSTM) **(prof)**
- [ ] **[T10.7]** Test average LSTM outputs (not just h_n[-1])
- [ ] **[T10.8]** Test reduced LSTM complexity (1 layer, 64 units)

- [ ] **[T11]** Implement interpretability (attention viz, CAM/Grad-CAM) and analyze cases **(dep: T10)**

- [ ] **[T12]** Train final model on full data; cross-val; finalize metrics/tables **(dep: T11)**
- [ ] **[T13]** Build and rehearse presentation; insert figures/results **(dep: T12) — Dec 1 & Dec 4**
- [ ] **[T14]** Write paper (methods, results, interpretability, limits); finalize figures **(dep: T12)**
- [ ] **[T15]** Submit final paper **(Dec 11)**; tag GitHub release with code, configs, checkpoints **(dep: T14)**

---

## Timeline by Calendar Week (starts Nov 9)

### Week of **Nov 9–Nov 15** (includes Nov 14 due)
- [ ] [T1] Data download
- [ ] [T2] Preprocessing/segmentation
- [ ] [T3] Train/val/test split + class stats
- [ ] [T4] Baseline CNN→LSTM implementation
- [ ] [T5] Baseline training + metrics
- [ ] [T6] **Submit Intermediate Progress Report (Nov 14 @ 8:00 AM)**

### Week of **Nov 16–Nov 22**
- [ ] [T7] Class-imbalance strategy (weighted loss/sampler/focal)
- [ ] [T8] Data augmentation for ECG
- [ ] [T9] Regularization + hyperparameter tuning (lr, batch, hidden units)

### Week of **Nov 23–Nov 29**
- [ ] [T10] Ablations (architecture variants)
- [ ] [T11] Interpretability (attention, CAM/Grad-CAM) and analysis notes

### Week of **Nov 30–Dec 6**
- [ ] [T12] Final training on full data; cross-val; finalize tables
- [ ] [T13] Presentation slides + practice (**Dec 1 & Dec 4**)

### Week of **Dec 7–Dec 11**
- [ ] [T14] Paper write-up (methods, results, interpretation, limitations, references)
- [ ] [T15] **Submit final paper (Dec 11)** and archive release (code + models)

---

## Notes
- Keep tasks one-line and update status in place; dependencies are indicated via **(dep: …)** and the ID map above.
- If a dependency changes, update the **Dependency Map** and the **(dep: …)** tags; no special tooling required.
- Each task is discrete and can be updated directly in GitHub via checkboxes.
-	After Dec 11, optional extensions include multi-lead ECGs, real-time inference, and transformer-based temporal modules.

---

# ECG Arrhythmia Classification

## Setup
```bash
pip install -r requirements.txt
```

## Usage

### 1. Preprocess Data
```bash
python preprocessing.py      # Download and preprocess
python create_splits.py      # Create train/val/test splits
```
### Notes
- Bandpass filter: 0.5-40 Hz
- Segments: 5 sec (1500 samples @ 300 Hz)
- Overlap: 50%
- Splits: 70% train, 15% val, 15% test (stratified)
- Class weights: Use with `nn.CrossEntropyLoss(weight=weights)`


### 2. Train Model
```bash
python train.py              # Train CNN-LSTM baseline
```

### Model Architecture
Serial CNN→LSTM:
- CNN: 3 conv blocks (32→64→128 channels)
- LSTM: 2 layers, 128 hidden units
- Output: 4 classes (Normal, AF, Other, Noisy)


### 3. Test Components
```bash
python dataset.py            # Test DataLoader
python model.py              # Test model architecture
```
### Metrics
- F1 score (weighted)
- AUROC (one-vs-rest)
- Sensitivity per class
- Specificity per class



## Files
- `preprocessing.py` - Download and preprocess ECG data
- `create_splits.py` - Create stratified splits
- `dataset.py` - PyTorch Dataset and DataLoader
- `model.py` - CNN-LSTM architecture
- `train.py` - Training loop with metrics

## Output
- `data/processed/` - Preprocessed segments
- `data/processed/X.npy` - All segments (shape: num_segments, 1500)
- `data/processed/y.npy` - All labels
- `data/splits/` - Train/val/test splits
- `data/splits/X_train.npy, X_val.npy, X_test.npy` - Split features
- `data/splits/y_train.npy, y_val.npy, y_test.npy` - Split labels
- `data/splits/class_weights.npy` - Weights for imbalanced classes
- `checkpoints/best_model.pt` - Trained model






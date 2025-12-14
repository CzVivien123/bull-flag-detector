# Bull Flag Pattern Detection using 1D CNN  
**Deep Learning Class (VITMMA19) – Project Work**

---

## Project Information

- **Selected Topic**: Bull-flag detector  
- **Student Name**: Cz. Vivien  
- **Aiming for +1 Mark**: Yes  

---

## Solution Description

This project focuses on the automatic detection and classification of **bullish and bearish flag-type chart patterns** in financial time series data (EUR/USD, 15-minute timeframe).

The task is formulated as a **multiclass classification problem**, where each detected pattern segment is assigned to one of six classes:
- Bullish Normal
- Bullish Wedge
- Bullish Pennant
- Bearish Normal
- Bearish Wedge
- Bearish Pennant

### Model Architecture

A **1D Convolutional Neural Network (CNN)** is used to process fixed-length multivariate time-series windows.  
Each input sample consists of:
- OHLC price channels (4 features)
- Fixed temporal length (256 bars), including configurable *pre-bars* to capture flagpole context

The CNN extracts local temporal patterns using convolutional layers, followed by fully connected layers for classification.

---

## Training Methodology

- Stratified train/validation split  
- Cross-entropy loss with class weighting  
- AdamW optimizer  
- Early stopping based on validation loss  
- Hyperparameter exploration (pre-bars, dropout, weight decay)

---

## Evaluation

Model performance is evaluated using:
- Accuracy  
- Macro F1-score  
- Weighted F1-score  
- Confusion matrix  

An **inference step** demonstrates how the trained model predicts the class of an unseen pattern segment.

---

## Data Preparation

The raw dataset is **not stored in the repository**.

Instead, the project automatically downloads a prepared dataset from a **GitHub Release** when executed.

### Data Handling Workflow

1. Check for required CSV and JSON files in `/app/data`
2. Download `dataset.zip` from GitHub Releases if missing
3. Extract files into the data directory
4. (Optional) Verify integrity using SHA-256 checksum
5. Preprocess labeled segments into fixed-length tensors

This process is fully automated and requires **no manual intervention**.

---

## Logging

All stages of the pipeline are logged to **standard output**, which Docker captures into:

```bash
log/run.log
```

The log contains:
- Configuration and hyperparameters
- Data loading and preprocessing confirmation
- Model architecture summary
- Training and validation metrics per epoch
- Final evaluation metrics (accuracy, F1-score, confusion matrix)
- Inference results

---

## Docker Instructions

The project is fully containerized.

### Build

```bash
docker build -t dl-project .
```

### Run (with log capture)

```bash
docker run --rm dl-project > log/run.log 2>&1
```

This runs the complete pipeline:
-Dataset download (if needed)
-Preprocessing
-Training
-Evaluation
-Inference

---

## File Structure

```lua
bull-flag-detector/
│
├── src/
│   ├── 01-data-preprocessing.py
│   ├── 02-training.py
│   ├── 03-evaluation.py
│   ├── 04-inference.py
│   ├── pipeline.py
│   ├── config.py
│   └── utils.py
│
├── notebook/
│   ├── cnn_ablation.ipynb
│   ├── compare_with_baseline.ipynb
│   └── overfit_one_batch.ipynb
│
├── log/
│   └── run.log
│
├── Dockerfile
├── requirements.txt
├── .dockerignore
├── .gitignore
└── README.md
```

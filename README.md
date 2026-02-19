# AIHRM-CVD
Adaptive Imbalance-Aware Hybrid Risk Mining Framework for Cardiovascular Disease Prediction

## Overview

AIHRM-CVD is a machine learning framework designed to improve robustness in cardiovascular disease (CVD) risk prediction under class imbalance conditions.

Traditional machine learning models often suffer from instability when trained on imbalanced medical datasets. This project introduces an imbalance-aware hybrid ensemble framework that integrates resampling techniques with weighted model aggregation to enhance reliability and clinical relevance.

## Key Contributions

- Removal of duplicate data to prevent data leakage
- SMOTE-based imbalance handling for minority class stabilization
- Weighted soft-voting ensemble combining Logistic Regression, Random Forest, and XGBoost
- Comparative evaluation against baseline models
- Performance validation using Accuracy, Precision, Recall, F1-score, and ROC-AUC

## Proposed Framework: AIHRM-CVD

### Step 1: Data Preprocessing
- Duplicate removal
- Missing value handling
- One-hot encoding of categorical variables
- Train-test split
- Feature scaling

### Step 2: Imbalance Handling
- Applied Synthetic Minority Oversampling Technique (SMOTE)

### Step 3: Hybrid Ensemble Model
- Logistic Regression (weight = 3)
- Random Forest (weight = 1)
- XGBoost (weight = 1)
- Soft voting aggregation

## Experimental Results

| Model | Accuracy | Precision | Recall | F1 Score |
|--------|----------|-----------|--------|----------|
| Logistic Regression | 0.836 | 0.848 | 0.848 | 0.848 |
| Random Forest | 0.770 | 0.787 | 0.787 | 0.787 |
| XGBoost | 0.704 | 0.727 | 0.727 | 0.727 |
| **AIHRM-CVD** | **0.852** | **0.875** | **0.848** | **0.862** |

Additional Metrics:
- ROC-AUC: 0.89
- False Negatives: 5
- True Positives: 28

The proposed framework maintains strong recall while improving overall F1-score and classification stability.

## Visual Outputs

Generated plots:
- F1 Score Comparison (outputs/plots/f1_comparison.png)
- Confusion Matrix (outputs/plots/confusion_matrix.png)
- ROC Curve (outputs/plots/roc_curve.png)

## Project Structure

AIHRM-CVD/
│
├── data/
│ ├── HeartDiseaseTrain-Test.csv
│ └── processed.pkl
│
├── models/
│ ├── aihrm_cvd_model.pkl
│ └── scaler.pkl
│
├── outputs/
│ ├── metrics/
│ │ ├── baseline_results.csv
│ │ └── aihrm_results.csv
│ │
│ └── plots/
│ ├── f1_comparison.png
│ ├── confusion_matrix.png
│ └── roc_curve.png
│
├── src/
│ ├── 01_preprocessing.py
│ ├── 02_baseline_models.py
│ ├── 03_aihrm_model.py
│ ├── 04_visualization.py
│ ├── 05_advanced_plots.py
│ └── check_data.py
│
└── README.md

## Installation

Install required dependencies:

pip install pandas scikit-learn xgboost imbalanced-learn matplotlib joblib

## Execution Pipeline

Run the full pipeline in order:

python src/01_preprocessing.py
python src/02_baseline_models.py
python src/03_aihrm_model.py
python src/04_visualization.py
python src/05_advanced_plots.py

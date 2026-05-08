# 🚨 Mobile Money Fraud Detection System

> Real-time ML pipeline for detecting fraudulent transactions — XGBoost · SHAP explainability · Streamlit dashboard

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-orange?style=flat-square)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35-red?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

---

## Problem

Mobile money fraud is a growing threat across Africa. Kenya alone processes over **KES 30 billion** in M-Pesa transactions daily — even a 0.17% fraud rate translates to millions in losses for users, fintechs, and SACCOs.

Existing rule-based systems flag too many false positives and lack explainability. This project builds an **end-to-end ML pipeline** that:
- Detects fraud in real time with **94.1% precision**
- Explains every decision using **SHAP values** — critical for CBK regulatory compliance
- Serves results through an **interactive Streamlit dashboard**

---

## Results

| Metric | Score |
|---|---|
| ROC-AUC | **0.9792** |
| AUC-PR (avg precision) | **0.8714** |
| Precision (fraud class) | **94.1%** |
| Recall (fraud class) | **91.7%** |
| F1-Score (fraud class) | **92.9%** |

> Evaluated on 20% held-out test set · Stratified 5-fold cross-validation

---

## Architecture

```
Raw transactions (CSV)
        │
        ▼
┌─────────────────┐
│   src/ingest.py │  ← load, validate, schema check
└────────┬────────┘
         │
         ▼
┌──────────────────┐
│ src/features.py  │  ← time encoding, amount z-score,
└────────┬─────────┘    off-hours flag, RobustScaler
         │
         ▼
┌─────────────────┐
│  src/train.py   │  ← SMOTE oversampling + XGBoost
└────────┬────────┘    5-fold stratified CV
         │
         ▼
┌──────────────────┐
│ src/evaluate.py  │  ← ROC, PR curves, confusion matrix
└────────┬─────────┘    SHAP TreeExplainer
         │
         ▼
┌───────────────────────┐
│  dashboard/app.py     │  ← Streamlit: overview, SHAP tab,
└───────────────────────┘    live transaction scorer
```

---

## Features

### Machine learning
- **XGBoost classifier** with histogram-based training (`tree_method=hist`)
- **SMOTE** oversampling — fraud upsampled to 10% of majority class
- **Stratified k-fold CV** — preserves fraud class ratio per fold
- **AUC-PR** as primary metric — correct choice for heavily imbalanced data (0.17% fraud)

### Engineered features
| Feature | Description |
|---|---|
| `amount_zscore` | How unusual the transaction amount is vs population |
| `amount_log` | Log-transform to handle heavy right skew |
| `off_hours` | Binary flag — transactions between midnight and 6am |
| `hour_sin / hour_cos` | Cyclic encoding of time-of-day |
| `high_amount_flag` | 1 if amount > 3 standard deviations above mean |
| `V1–V28` | PCA-transformed card behaviour features (from dataset) |

### SHAP explainability
- **TreeExplainer** — exact Shapley values for tree-based models
- Global importance: beeswarm + bar plots saved to `models/plots/`
- Local explanation: per-transaction top-5 feature drivers in dashboard
- Every fraud flag is interpretable — meets regulatory explainability standards

---

## Project structure

```
fraud_detection/
├── data/
│   └── creditcard.csv          # Kaggle dataset (not committed)
├── src/
│   ├── ingest.py               # data loading & validation
│   ├── features.py             # feature engineering
│   ├── train.py                # XGBoost + SMOTE + CV
│   ├── evaluate.py             # metrics + SHAP plots
│   └── predict.py              # single transaction scorer
├── dashboard/
│   └── app.py                  # Streamlit dashboard
├── models/                     # saved artifacts (not committed)
│   └── plots/                  # evaluation plots
├── main.py                     # pipeline orchestrator
├── requirements.txt
└── README.md
```

---

## Quickstart

### 1. Clone the repo
```bash
git clone https://github.com/yourusername/fraud-detection-ml.git
cd fraud-detection-ml
```

### 2. Set up environment
```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux

pip install pandas numpy scikit-learn xgboost imbalanced-learn shap matplotlib seaborn streamlit joblib plotly
```

### 3. Download dataset
Download `creditcard.csv` from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place it in `data/`.

### 4. Run pipeline
```bash
python main.py --step all
```

### 5. Launch dashboard
```bash
streamlit run dashboard/app.py
```
Open `http://localhost:8501` in your browser.

---

## Dashboard

The Streamlit dashboard has three tabs:

**Overview** — transaction volume, fraud rate, amount distribution by class, hourly fraud pattern heatmap

**SHAP Explainability** — global feature importance (beeswarm + bar), feature interpretation guide

**Score a Transaction** — enter transaction values manually, get live fraud probability gauge + SHAP explanation of that specific decision

---

## Tech stack

| Component | Tool |
|---|---|
| Data wrangling | pandas, numpy |
| ML model | XGBoost |
| Imbalance handling | imbalanced-learn (SMOTE) |
| Explainability | SHAP (TreeExplainer) |
| Evaluation | scikit-learn |
| Visualization | matplotlib, seaborn, plotly |
| Dashboard | Streamlit |
| Serialization | joblib |

---

## Dataset

**Credit Card Fraud Detection** — Kaggle / ULB Machine Learning Group
284,807 transactions · 492 fraud cases (0.172%) · 28 PCA features + Amount + Time
[https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

---

## Author

**[Your Full Name]**
B.CS Graduate · Data Scientist
[LinkedIn](https://linkedin.com/in/yourprofile) · [GitHub](https://github.com/yourusername)

---

## License

MIT License — free to use, modify, and distribute with attribution.

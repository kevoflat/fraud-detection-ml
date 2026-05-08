"""
src/predict.py
Scoring function — takes a single transaction dict or DataFrame,
returns fraud probability + SHAP explanation for that transaction.
Used by the Streamlit dashboard.
"""

import pandas as pd
import numpy as np
import joblib
import shap
import logging
from pathlib import Path

log = logging.getLogger(__name__)

MODEL_PATH = Path("models/xgb_fraud.pkl")
THRESHOLD = 0.5   # classification threshold — tune based on business cost


def load_model():
    return joblib.load(MODEL_PATH)


def preprocess_single(txn: dict, feature_cols: list[str]) -> pd.DataFrame:
    """Convert raw transaction dict → feature DataFrame ready for scoring."""
    import numpy as np

    df = pd.DataFrame([txn])

    # Time features
    df["hour"] = (df["Time"] % 86400) / 3600
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["off_hours"] = ((df["hour"] >= 0) & (df["hour"] < 6)).astype(int)

    # Amount features — use global stats from training set
    # (in production, load these from a saved scaler artifact)
    from sklearn.preprocessing import RobustScaler
    scaler = RobustScaler()
    df["amount_scaled"] = scaler.fit_transform(df[["Amount"]])
    df["amount_log"] = np.log1p(df["Amount"])
    df["amount_zscore"] = (df["Amount"] - 88.35) / 250.12   # training set stats
    df["high_amount_flag"] = (df["amount_zscore"] > 3).astype(int)

    return df[feature_cols]


def score(txn: dict, feature_cols: list[str], pipeline=None) -> dict:
    """
    Score a single transaction.
    Returns: fraud_prob, is_fraud, shap_explanation (top 5 features)
    """
    if pipeline is None:
        pipeline = load_model()

    X = preprocess_single(txn, feature_cols)
    fraud_prob = pipeline.predict_proba(X)[0, 1]
    is_fraud = fraud_prob >= THRESHOLD

    # SHAP for this single transaction
    clf = pipeline.named_steps["clf"]
    explainer = shap.TreeExplainer(clf)
    shap_vals = explainer.shap_values(X)[0]

    # Top 5 contributing features
    top_idx = np.argsort(np.abs(shap_vals))[::-1][:5]
    explanation = [
        {
            "feature": feature_cols[i],
            "shap_value": round(float(shap_vals[i]), 4),
            "feature_value": round(float(X.iloc[0, i]), 4),
        }
        for i in top_idx
    ]

    return {
        "fraud_probability": round(float(fraud_prob), 4),
        "is_fraud": bool(is_fraud),
        "decision": "FRAUD — BLOCK" if is_fraud else "LEGITIMATE — APPROVE",
        "top_features": explanation,
    }


def score_batch(df: pd.DataFrame, feature_cols: list[str], pipeline=None) -> pd.DataFrame:
    """Score a full DataFrame of transactions."""
    if pipeline is None:
        pipeline = load_model()

    probs = pipeline.predict_proba(df[feature_cols])[:, 1]
    df = df.copy()
    df["fraud_probability"] = probs
    df["prediction"] = (probs >= THRESHOLD).astype(int)
    return df

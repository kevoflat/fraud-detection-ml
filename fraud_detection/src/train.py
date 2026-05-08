"""
src/train.py
XGBoost classifier with SMOTE oversampling + stratified k-fold CV.
Saves best model to models/xgb_fraud.pkl
"""

import pandas as pd
import numpy as np
import joblib
import logging
from pathlib import Path

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, classification_report
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from xgboost import XGBClassifier

log = logging.getLogger(__name__)

MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "xgb_fraud.pkl"
FEATURE_PATH = Path("data/features.parquet")


XGB_PARAMS = {
    "n_estimators": 300,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "scale_pos_weight": 1,          # SMOTE handles imbalance; keep neutral
    "eval_metric": "aucpr",         # area under precision-recall — better for imbalanced
    "random_state": 42,
    "n_jobs": -1,
    "tree_method": "hist",          # fast histogram-based training
}

SMOTE_PARAMS = {
    "sampling_strategy": 0.1,       # upsample fraud to 10% of majority class
    "random_state": 42,
    "k_neighbors": 5,
}


def load_features(feature_cols: list[str]) -> tuple:
    df = pd.read_parquet(FEATURE_PATH)
    X = df[feature_cols].values
    y = df["Class"].values
    log.info(f"X shape: {X.shape} | fraud={y.sum()} ({y.mean()*100:.3f}%)")
    return X, y


def build_pipeline() -> ImbPipeline:
    return ImbPipeline([
        ("smote", SMOTE(**SMOTE_PARAMS)),
        ("clf", XGBClassifier(**XGB_PARAMS)),
    ])


def cross_validate(X, y, n_splits: int = 5) -> np.ndarray:
    """Stratified k-fold CV — returns AUC-PR scores per fold."""
    log.info(f"Running {n_splits}-fold stratified CV...")
    pipeline = build_pipeline()
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = cross_val_score(
        pipeline, X, y,
        cv=skf,
        scoring="average_precision",   # AUC-PR
        n_jobs=-1,
    )
    log.info(f"CV AUC-PR: {scores.mean():.4f} ± {scores.std():.4f}")
    return scores


def train_final(X, y) -> ImbPipeline:
    """Retrain on full dataset after CV."""
    log.info("Training final model on full data...")
    pipeline = build_pipeline()
    pipeline.fit(X, y)
    log.info("Training complete ✓")
    return pipeline


def save_model(pipeline: ImbPipeline) -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    log.info(f"Model saved → {MODEL_PATH}")


def run(feature_cols: list[str]) -> ImbPipeline:
    X, y = load_features(feature_cols)
    cv_scores = cross_validate(X, y)
    pipeline = train_final(X, y)
    save_model(pipeline)
    return pipeline, cv_scores


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    from features import PCA_COLS
    feature_cols = PCA_COLS + [
        "amount_scaled", "amount_log", "amount_zscore",
        "high_amount_flag", "hour_sin", "hour_cos", "off_hours"
    ]
    run(feature_cols)

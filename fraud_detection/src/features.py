"""
src/features.py
Feature engineering on top of the raw PCA features.
Adds domain-style features: amount scaling, time-of-day, velocity proxies.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from pathlib import Path
import logging

log = logging.getLogger(__name__)

FEATURE_PATH = Path("data/features.parquet")
PCA_COLS = [f"V{i}" for i in range(1, 29)]


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Convert raw 'Time' (seconds elapsed) into cyclic hour-of-day features."""
    df = df.copy()
    # Time is seconds from first transaction — map to hour in day (mod 86400)
    df["hour"] = (df["Time"] % 86400) / 3600
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    # Off-hours flag: midnight–6am is high-risk window
    df["off_hours"] = ((df["hour"] >= 0) & (df["hour"] < 6)).astype(int)
    return df


def scale_amount(df: pd.DataFrame) -> pd.DataFrame:
    """RobustScaler on Amount (resistant to outliers)."""
    df = df.copy()
    scaler = RobustScaler()
    df["amount_scaled"] = scaler.fit_transform(df[["Amount"]])
    # Log-transform for heavy right skew
    df["amount_log"] = np.log1p(df["Amount"])
    return df, scaler


def add_amount_zscore(df: pd.DataFrame) -> pd.DataFrame:
    """Z-score of Amount — flags unusually large transactions."""
    df = df.copy()
    mu, sigma = df["Amount"].mean(), df["Amount"].std()
    df["amount_zscore"] = (df["Amount"] - mu) / sigma
    df["high_amount_flag"] = (df["amount_zscore"] > 3).astype(int)
    return df


def select_features(df: pd.DataFrame) -> list[str]:
    engineered = [
        "amount_scaled", "amount_log", "amount_zscore",
        "high_amount_flag", "hour_sin", "hour_cos", "off_hours"
    ]
    return PCA_COLS + engineered


def build(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], object]:
    log.info("Engineering features...")
    df = add_time_features(df)
    df, scaler = scale_amount(df)
    df = add_amount_zscore(df)

    feature_cols = select_features(df)
    log.info(f"Total features: {len(feature_cols)}")

    FEATURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(FEATURE_PATH, index=False)
    log.info(f"Saved feature matrix → {FEATURE_PATH}")

    return df, feature_cols, scaler


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    from ingest import run as ingest_run
    df, _ = ingest_run()
    build(df)

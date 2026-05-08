"""
src/ingest.py
Load, validate, and do initial EDA on the Kaggle creditcard.csv dataset.
Expected columns: Time, V1–V28 (PCA features), Amount, Class
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

RAW_PATH = Path("data/creditcard.csv")
PROCESSED_PATH = Path("data/processed.parquet")


def load_raw(path: Path = RAW_PATH) -> pd.DataFrame:
    log.info(f"Loading dataset from {path}")
    df = pd.read_csv(path)
    log.info(f"Shape: {df.shape}")
    return df


def validate(df: pd.DataFrame) -> None:
    expected_cols = {"Time", "Amount", "Class"} | {f"V{i}" for i in range(1, 29)}
    missing = expected_cols - set(df.columns)
    assert not missing, f"Missing columns: {missing}"
    assert df["Class"].isin([0, 1]).all(), "Class must be binary 0/1"
    assert df.isnull().sum().sum() == 0, "Nulls detected — check dataset"
    log.info("Validation passed ✓")


def summarise(df: pd.DataFrame) -> dict:
    fraud = df["Class"].sum()
    total = len(df)
    summary = {
        "total_transactions": total,
        "fraud_count": fraud,
        "legit_count": total - fraud,
        "fraud_rate_pct": round(fraud / total * 100, 4),
        "amount_mean": round(df["Amount"].mean(), 2),
        "amount_max": round(df["Amount"].max(), 2),
    }
    for k, v in summary.items():
        log.info(f"  {k}: {v}")
    return summary


def run():
    df = load_raw()
    validate(df)
    summary = summarise(df)
    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(PROCESSED_PATH, index=False)
    log.info(f"Saved processed data → {PROCESSED_PATH}")
    return df, summary


if __name__ == "__main__":
    run()

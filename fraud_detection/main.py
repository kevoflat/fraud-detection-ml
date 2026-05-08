"""
main.py - End-to-end pipeline runner.
Usage:
    python main.py --step all
    python main.py --step ingest
    python main.py --step features
    python main.py --step train
    python main.py --step evaluate
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def get_feature_cols():
    from features import PCA_COLS
    return PCA_COLS + [
        "amount_scaled", "amount_log", "amount_zscore",
        "high_amount_flag", "hour_sin", "hour_cos", "off_hours"
    ]


def step_ingest():
    from ingest import run
    log.info("=" * 50)
    log.info("STEP 1 - DATA INGESTION")
    log.info("=" * 50)
    df, summary = run()
    return df, summary


def step_features(df=None):
    from features import build
    import pandas as pd
    log.info("=" * 50)
    log.info("STEP 2 - FEATURE ENGINEERING")
    log.info("=" * 50)
    if df is None:
        df = pd.read_parquet("data/processed.parquet")
    df_feat, feature_cols, scaler = build(df)
    return df_feat, feature_cols


def step_train(feature_cols):
    from train import run
    log.info("=" * 50)
    log.info("STEP 3 - MODEL TRAINING (XGBoost + SMOTE)")
    log.info("=" * 50)
    pipeline, cv_scores = run(feature_cols)
    log.info(f"CV AUC-PR mean: {cv_scores.mean():.4f}")
    return pipeline


def step_evaluate(feature_cols):
    from evaluate import run
    log.info("=" * 50)
    log.info("STEP 4 - EVALUATION + SHAP")
    log.info("=" * 50)
    metrics = run(feature_cols)
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Fraud detection pipeline runner")
    parser.add_argument(
        "--step",
        choices=["ingest", "features", "train", "evaluate", "all"],
        default="all",
        help="Which pipeline step to run"
    )
    args = parser.parse_args()

    feature_cols = get_feature_cols()

    if args.step == "ingest":
        step_ingest()
    elif args.step == "features":
        step_features()
    elif args.step == "train":
        step_train(feature_cols)
    elif args.step == "evaluate":
        step_evaluate(feature_cols)
    elif args.step == "all":
        
        df, _ = step_ingest()
        _, feature_cols = step_features(df)
        step_train(feature_cols)
        step_evaluate(feature_cols)
        log.info("=" * 50)
        log.info("Pipeline complete!")
        log.info("Run dashboard: streamlit run dashboard/app.py")
        log.info("=" * 50)


if __name__ == "__main__":
    main()
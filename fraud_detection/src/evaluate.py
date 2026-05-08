"""
src/evaluate.py
Model evaluation: classification metrics, ROC curve, PR curve, SHAP values.
All plots saved to models/plots/
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import shap
import joblib
import logging
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, roc_auc_score,
    average_precision_score, roc_curve,
    precision_recall_curve, confusion_matrix,
    ConfusionMatrixDisplay
)

log = logging.getLogger(__name__)

MODEL_PATH = Path("models/xgb_fraud.pkl")
FEATURE_PATH = Path("data/features.parquet")
PLOT_DIR = Path("models/plots")
SHAP_PATH = Path("models/shap_values.pkl")


def load_test_split(feature_cols: list[str], test_size: float = 0.2):
    df = pd.read_parquet(FEATURE_PATH)
    X = df[feature_cols]
    y = df["Class"]
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=42)


def evaluate_metrics(pipeline, X_test, y_test) -> dict:
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    y_pred = pipeline.predict(X_test)

    metrics = {
        "roc_auc": round(roc_auc_score(y_test, y_prob), 4),
        "avg_precision": round(average_precision_score(y_test, y_prob), 4),
        "report": classification_report(y_test, y_pred, target_names=["Legit", "Fraud"]),
    }
    log.info(f"ROC-AUC:  {metrics['roc_auc']}")
    log.info(f"AUC-PR:   {metrics['avg_precision']}")
    log.info(f"\n{metrics['report']}")
    return metrics, y_prob, y_pred


def plot_roc_pr(y_test, y_prob) -> None:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)
    axes[0].plot(fpr, tpr, color="#185FA5", lw=2, label=f"AUC = {auc:.4f}")
    axes[0].plot([0,1],[0,1], "k--", lw=1)
    axes[0].set(title="ROC Curve", xlabel="FPR", ylabel="TPR")
    axes[0].legend()

    # PR curve
    prec, rec, _ = precision_recall_curve(y_test, y_prob)
    ap = average_precision_score(y_test, y_prob)
    axes[1].plot(rec, prec, color="#E24B4A", lw=2, label=f"AP = {ap:.4f}")
    axes[1].set(title="Precision-Recall Curve", xlabel="Recall", ylabel="Precision")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(PLOT_DIR / "roc_pr.png", dpi=150, bbox_inches="tight")
    log.info("Saved ROC-PR plot ✓")
    plt.close()


def plot_confusion(y_test, y_pred) -> None:
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Legit", "Fraud"])
    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "confusion_matrix.png", dpi=150, bbox_inches="tight")
    log.info("Saved confusion matrix ✓")
    plt.close()


def compute_shap(pipeline, X_test: pd.DataFrame, n_samples: int = 500) -> tuple:
    """
    Extract the XGBClassifier from pipeline and compute SHAP values.
    Uses TreeExplainer — fastest for tree-based models.
    """
    log.info(f"Computing SHAP values on {n_samples} test samples...")
    clf = pipeline.named_steps["clf"]

    # Sample for speed
    X_sample = X_test.sample(n=min(n_samples, len(X_test)), random_state=42)

    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X_sample)

    # Save for dashboard reuse
    joblib.dump({"shap_values": shap_values, "X_sample": X_sample}, SHAP_PATH)
    log.info(f"SHAP values saved → {SHAP_PATH}")

    return explainer, shap_values, X_sample


def plot_shap_summary(shap_values, X_sample: pd.DataFrame) -> None:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_sample, show=False, max_display=15)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "shap_summary.png", dpi=150, bbox_inches="tight")
    log.info("Saved SHAP summary plot ✓")
    plt.close()


def plot_shap_bar(shap_values, X_sample: pd.DataFrame) -> None:
    plt.figure(figsize=(8, 6))
    shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False, max_display=15)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "shap_bar.png", dpi=150, bbox_inches="tight")
    log.info("Saved SHAP bar plot ✓")
    plt.close()


def run(feature_cols: list[str]) -> dict:
    pipeline = joblib.load(MODEL_PATH)
    X_train, X_test, y_train, y_test = load_test_split(feature_cols)

    metrics, y_prob, y_pred = evaluate_metrics(pipeline, X_test, y_test)
    plot_roc_pr(y_test, y_prob)
    plot_confusion(y_test, y_pred)

    explainer, shap_values, X_sample = compute_shap(pipeline, X_test)
    plot_shap_summary(shap_values, X_sample)
    plot_shap_bar(shap_values, X_sample)

    return metrics


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    from features import PCA_COLS
    feature_cols = PCA_COLS + [
        "amount_scaled", "amount_log", "amount_zscore",
        "high_amount_flag", "hour_sin", "hour_cos", "off_hours"
    ]
    run(feature_cols)

"""
dashboard/app.py
Streamlit dashboard — real-time fraud monitoring + SHAP explainability.
Run: streamlit run dashboard/app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent / "src"))
from predict import score, score_batch, THRESHOLD
from features import PCA_COLS

# ── Config ────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fraud Detection Dashboard",
    page_icon="🚨",
    layout="wide",
)

FEATURE_COLS = PCA_COLS + [
    "amount_scaled", "amount_log", "amount_zscore",
    "high_amount_flag", "hour_sin", "hour_cos", "off_hours"
]

# ── Load artifacts ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_pipeline():
    return joblib.load("models/xgb_fraud.pkl")

@st.cache_data
def load_shap_data():
    data = joblib.load("models/shap_values.pkl")
    return data["shap_values"], data["X_sample"]

@st.cache_data
def load_feature_data():
    return pd.read_parquet("data/features.parquet")

# ── Sidebar ────────────────────────────────────────────────────────────────────
st.sidebar.title("⚙️ Settings")
threshold = st.sidebar.slider(
    "Decision threshold", 0.1, 0.9, THRESHOLD, 0.05,
    help="Lower → catch more fraud (higher false positives)"
)
st.sidebar.markdown("---")
st.sidebar.markdown("**Stack:** XGBoost · SMOTE · SHAP · Streamlit")
st.sidebar.markdown("**Dataset:** Kaggle Credit Card Fraud (284,807 txns)")

# ── Header ─────────────────────────────────────────────────────────────────────
st.title("🚨 Mobile Money Fraud Detection System")
st.markdown("Real-time transaction scoring · XGBoost + SHAP explainability")
st.markdown("---")

# ── KPI Metrics ───────────────────────────────────────────────────────────────
df = load_feature_data()
total = len(df)
fraud_n = df["Class"].sum()
fraud_rate = fraud_n / total * 100

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Transactions", f"{total:,}")
col2.metric("Fraud Cases", f"{fraud_n:,}", delta=f"{fraud_rate:.3f}% of total", delta_color="inverse")
col3.metric("Model AUC-ROC", "0.9792")
col4.metric("AUC-PR", "0.8714")

st.markdown("---")

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📊 Overview", "🔍 SHAP Explainability", "⚡ Score a Transaction"])

# ── Tab 1: Overview ────────────────────────────────────────────────────────────
with tab1:
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Transaction amount distribution")
        fig = px.histogram(
            df, x="Amount", color=df["Class"].map({0: "Legit", 1: "Fraud"}),
            nbins=80, barmode="overlay", opacity=0.7,
            color_discrete_map={"Legit": "#185FA5", "Fraud": "#E24B4A"},
            labels={"color": "Class"},
        )
        fig.update_layout(height=320, margin=dict(t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.subheader("Fraud vs legit — amount stats")
        stats = df.groupby("Class")["Amount"].describe().rename(
            index={0: "Legit", 1: "Fraud"}
        )
        st.dataframe(stats.style.format("{:.2f}"), use_container_width=True)

    st.subheader("Fraud rate over time (hourly)")
    df_time = df.copy()
    df_time["hour"] = (df_time["Time"] % 86400 / 3600).astype(int)
    hourly = df_time.groupby("hour")["Class"].agg(["sum", "count"])
    hourly["fraud_rate"] = hourly["sum"] / hourly["count"] * 100
    fig2 = px.bar(
        hourly.reset_index(), x="hour", y="fraud_rate",
        color="fraud_rate",
        color_continuous_scale=["#EAF3DE", "#EF9F27", "#E24B4A"],
        labels={"hour": "Hour of day", "fraud_rate": "Fraud rate (%)"},
    )
    fig2.update_layout(height=280, margin=dict(t=10, b=10))
    st.plotly_chart(fig2, use_container_width=True)

# ── Tab 2: SHAP ────────────────────────────────────────────────────────────────
with tab2:
    st.subheader("Global SHAP — feature importance")
    st.markdown("How much each feature contributes to the fraud score across all transactions.")

    shap_plot = Path("models/plots/shap_summary.png")
    shap_bar = Path("models/plots/shap_bar.png")

    col_s1, col_s2 = st.columns(2)
    if shap_bar.exists():
        col_s1.image(str(shap_bar), caption="Mean |SHAP| — overall importance", use_column_width=True)
    if shap_plot.exists():
        col_s2.image(str(shap_plot), caption="SHAP beeswarm — direction of impact", use_column_width=True)

    if not shap_bar.exists():
        st.info("Run `python src/evaluate.py` first to generate SHAP plots.")

    st.markdown("---")
    st.subheader("What the top features mean")
    st.markdown("""
| Feature | Interpretation |
|---|---|
| `V14`, `V17`, `V12` | PCA-transformed card behaviour — high absolute value = anomalous |
| `amount_zscore` | How unusual the transaction amount is vs population mean |
| `off_hours` | 1 if transaction occurred between midnight–6am |
| `high_amount_flag` | 1 if amount > 3 standard deviations above mean |
| `hour_sin / hour_cos` | Cyclic encoding of time-of-day pattern |
    """)

# ── Tab 3: Score a transaction ─────────────────────────────────────────────────
with tab3:
    st.subheader("⚡ Score a single transaction in real time")
    st.markdown("Enter transaction details — the model will return a fraud probability + SHAP explanation.")

    with st.form("score_form"):
        col1, col2 = st.columns(2)
        with col1:
            amount = st.number_input("Amount (KES)", min_value=0.0, value=250.0, step=10.0)
            time_sec = st.number_input("Time (seconds since day start)", min_value=0, max_value=86400, value=14400)
        with col2:
            v1 = st.number_input("V1", value=-1.36)
            v2 = st.number_input("V2", value=-0.07)
            v14 = st.number_input("V14 (most important)", value=-2.1)

        submitted = st.form_submit_button("🔍 Score transaction", type="primary")

    if submitted:
        pipeline = load_pipeline()

        # Build a dummy transaction with zeros for unused PCA features
        txn = {f"V{i}": 0.0 for i in range(1, 29)}
        txn.update({"V1": v1, "V2": v2, "V14": v14, "Amount": amount, "Time": time_sec})

        with st.spinner("Scoring..."):
            result = score(txn, FEATURE_COLS, pipeline)

        fraud_prob = result["fraud_probability"]
        is_fraud = fraud_prob >= threshold

        st.markdown("### Result")
        if is_fraud:
            st.error(f"🚨 **FRAUD DETECTED** — probability: `{fraud_prob:.4f}`")
        else:
            st.success(f"✅ **LEGITIMATE** — probability: `{fraud_prob:.4f}`")

        # Gauge chart
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=fraud_prob * 100,
            title={"text": "Fraud probability (%)"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#E24B4A" if is_fraud else "#639922"},
                "steps": [
                    {"range": [0, 30], "color": "#EAF3DE"},
                    {"range": [30, 60], "color": "#FAEEDA"},
                    {"range": [60, 100], "color": "#FCEBEB"},
                ],
                "threshold": {"line": {"color": "#1a1a18", "width": 2}, "value": threshold * 100}
            }
        ))
        fig_gauge.update_layout(height=280)
        st.plotly_chart(fig_gauge, use_container_width=True)

        st.markdown("### SHAP explanation — top 5 drivers")
        shap_df = pd.DataFrame(result["top_features"])
        shap_df["direction"] = shap_df["shap_value"].apply(lambda x: "↑ Increases fraud risk" if x > 0 else "↓ Reduces fraud risk")
        st.dataframe(shap_df, use_container_width=True)

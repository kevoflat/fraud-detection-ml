import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent / "src"))
from predict import score, THRESHOLD
from features import PCA_COLS

st.set_page_config(page_title="Fraud Detection | Kelvin Mwangi", page_icon="🚨", layout="wide")

FEATURE_COLS = PCA_COLS + ["amount_scaled","amount_log","amount_zscore","high_amount_flag","hour_sin","hour_cos","off_hours"]
MODEL_PATH  = Path("models/xgb_fraud.pkl")
SCALER_PATH = Path("models/scaler.pkl")
SHAP_PATH   = Path("models/shap_values.pkl")
FEATURE_PATH= Path("data/features.parquet")

@st.cache_resource
def load_pipeline():
    return joblib.load(MODEL_PATH)

@st.cache_resource
def load_scaler():
    return joblib.load(SCALER_PATH)

@st.cache_data
def load_summary_stats():
    df = pd.read_parquet(FEATURE_PATH, columns=["Time","Amount","Class"])
    total = len(df)
    fraud_n = int(df["Class"].sum())
    df["hour"] = (df["Time"] % 86400 / 3600).astype(int)
    hourly = df.groupby("hour")["Class"].agg(["sum","count"])
    hourly["fraud_rate"] = hourly["sum"] / hourly["count"] * 100
    amount_stats = df.groupby("Class")["Amount"].describe().rename(index={0:"Legit",1:"Fraud"})
    amounts = df.groupby("Class")["Amount"].apply(list).to_dict()
    return total, fraud_n, hourly, amount_stats, amounts

def preprocess_txn(txn, scaler):
    df = pd.DataFrame([txn])
    df["hour"] = (df["Time"] % 86400) / 3600
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["off_hours"] = ((df["hour"] >= 0) & (df["hour"] < 6)).astype(int)
    df["amount_scaled"] = scaler.transform(df[["Amount"]])
    df["amount_log"] = np.log1p(df["Amount"])
    df["amount_zscore"] = (df["Amount"] - 88.35) / 250.12
    df["high_amount_flag"] = (df["amount_zscore"] > 3).astype(int)
    return df[FEATURE_COLS]

def score_transaction(txn, threshold):
    pipeline = load_pipeline()
    scaler = load_scaler()
    X = preprocess_txn(txn, scaler)
    fraud_prob = float(pipeline.predict_proba(X)[0, 1])
    clf = pipeline.named_steps["clf"]
    explainer = shap.TreeExplainer(clf)
    shap_vals = explainer.shap_values(X)[0]
    top_idx = np.argsort(np.abs(shap_vals))[::-1][:5]
    explanation = [
        {"Feature": FEATURE_COLS[i],
         "SHAP value": round(float(shap_vals[i]), 4),
         "Feature value": round(float(X.iloc[0, i]), 4),
         "Direction": "Up Increases risk" if shap_vals[i] > 0 else "Down Reduces risk"}
        for i in top_idx
    ]
    return fraud_prob, fraud_prob >= threshold, explanation

with st.sidebar:
    st.markdown("## Settings")
    threshold = st.slider("Decision threshold", 0.1, 0.9, 0.5, 0.05)
    st.markdown("---")
    st.markdown("**Stack:** XGBoost · SMOTE · SHAP · Streamlit")
    st.markdown("**Dataset:** Kaggle Credit Card Fraud — 284,807 txns")
    st.markdown("---")
    st.markdown("**Kelvin Mwangi** — Data Scientist")

st.title("🚨 Mobile Money Fraud Detection System")
st.markdown("End-to-end ML pipeline · XGBoost + SMOTE + SHAP · Real-time scoring")
st.markdown("---")

total, fraud_n, hourly, amount_stats, amounts = load_summary_stats()
fraud_rate = fraud_n / total * 100

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total Transactions", f"{total:,}")
c2.metric("Fraud Cases", f"{fraud_n:,}", delta=f"{fraud_rate:.3f}%", delta_color="inverse")
c3.metric("ROC-AUC", "0.9792")
c4.metric("AUC-PR", "0.8714")
c5.metric("Precision (Fraud)", "94.1%")
st.markdown("---")

tab1, tab2, tab3 = st.tabs(["📊 Overview", "🔍 SHAP Explainability", "⚡ Score Transaction"])

with tab1:
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Amount distribution: Fraud vs Legit")
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=amounts[0][:2000], name="Legit", marker_color="#185FA5", opacity=0.65, nbinsx=60))
        fig.add_trace(go.Histogram(x=amounts[1], name="Fraud", marker_color="#E24B4A", opacity=0.75, nbinsx=60))
        fig.update_layout(barmode="overlay", height=320, margin=dict(t=10,b=10))
        st.plotly_chart(fig, use_container_width=True)
    with col_b:
        st.subheader("Amount statistics by class")
        st.dataframe(amount_stats.style.format("{:.2f}"), use_container_width=True)

    st.subheader("Hourly fraud rate")
    fig2 = px.bar(hourly.reset_index(), x="hour", y="fraud_rate",
                  color="fraud_rate", color_continuous_scale=["#EAF3DE","#EF9F27","#E24B4A"],
                  labels={"hour":"Hour of day","fraud_rate":"Fraud rate (%)"})
    fig2.update_layout(height=280, margin=dict(t=10,b=10))
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Model evaluation plots")
    ec1, ec2 = st.columns(2)
    roc_pr = Path("models/plots/roc_pr.png")
    conf   = Path("models/plots/confusion_matrix.png")
    if roc_pr.exists():
        ec1.image(str(roc_pr), caption="ROC & PR curves", use_column_width=True)
    if conf.exists():
        ec2.image(str(conf), caption="Confusion matrix", use_column_width=True)

with tab2:
    st.subheader("Global SHAP — feature importance")
    col_s1, col_s2 = st.columns(2)
    shap_bar = Path("models/plots/shap_bar.png")
    shap_sum = Path("models/plots/shap_summary.png")
    if shap_bar.exists():
        col_s1.image(str(shap_bar), caption="Mean SHAP — importance ranking", use_column_width=True)
    if shap_sum.exists():
        col_s2.image(str(shap_sum), caption="SHAP beeswarm — direction of impact", use_column_width=True)
    st.markdown("---")
    st.subheader("Feature interpretation")
    st.dataframe(pd.DataFrame({
        "Feature": ["V14","V17","V12","amount_zscore","off_hours","high_amount_flag","hour_sin/cos"],
        "Type": ["PCA","PCA","PCA","Engineered","Engineered","Engineered","Engineered"],
        "Interpretation": [
            "Strongest fraud signal — high absolute value = anomalous card behaviour",
            "Transaction velocity / recency pattern",
            "Merchant category signal",
            "How unusual the amount is vs population — fraud clusters at extremes",
            "1 if transaction between midnight-6am — elevated fraud window",
            "1 if amount > 3 standard deviations above mean",
            "Cyclic hour encoding — captures intra-day fraud timing",
        ]
    }), use_container_width=True, hide_index=True)

with tab3:
    st.subheader("Score a single transaction in real time")
    col1, col2, col3 = st.columns(3)
    with col1:
        amount   = st.number_input("Amount (KES)", min_value=0.0, value=250.0, step=10.0)
        time_sec = st.number_input("Time (seconds since day start)", 0, 86400, 14400)
    with col2:
        v1  = st.number_input("V1",  value=-1.36)
        v2  = st.number_input("V2",  value=-0.07)
        v3  = st.number_input("V3",  value=2.54)
    with col3:
        v14 = st.number_input("V14 (top feature)", value=-2.10)
        v17 = st.number_input("V17", value=-0.56)
        v12 = st.number_input("V12", value=-2.96)

    if st.button("Score transaction", type="primary"):
        txn = {f"V{i}": 0.0 for i in range(1, 29)}
        txn.update({"V1":v1,"V2":v2,"V3":v3,"V14":v14,"V17":v17,"V12":v12,"Amount":amount,"Time":time_sec})
        with st.spinner("Scoring..."):
            fraud_prob, is_fraud, explanation = score_transaction(txn, threshold)
        if is_fraud:
            st.error(f"FRAUD DETECTED — probability: {fraud_prob:.4f}")
        else:
            st.success(f"LEGITIMATE — probability: {fraud_prob:.4f}")
        fig_g = go.Figure(go.Indicator(
            mode="gauge+number",
            value=fraud_prob * 100,
            title={"text": "Fraud probability (%)"},
            number={"suffix": "%", "valueformat": ".1f"},
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
        fig_g.update_layout(height=260, margin=dict(t=30,b=10))
        st.plotly_chart(fig_g, use_container_width=True)
        st.markdown("#### Top 5 SHAP drivers for this transaction")
        st.dataframe(pd.DataFrame(explanation), use_container_width=True, hide_index=True)

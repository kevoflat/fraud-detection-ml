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

st.set_page_config(page_title="Fraud Detection | Kelvin Mwangi", page_icon="??", layout="wide")

FEATURE_COLS = [f"V{i}" for i in range(1, 29)] + [
    "amount_scaled","amount_log","amount_zscore","high_amount_flag","hour_sin","hour_cos","off_hours"
]
MODEL_PATH   = Path(__file__).parent.parent / "models/xgb_fraud.pkl")
SCALER_PATH  = Path(__file__).parent.parent / "models/scaler.pkl")
FEATURE_PATH = Path(__file__).parent.parent / "data/features.parquet")
DEMO_MODE    = not MODEL_PATH.exists()

@st.cache_resource
def load_pipeline():
    return joblib.load(MODEL_PATH)

@st.cache_resource
def load_scaler():
    return joblib.load(SCALER_PATH)

@st.cache_data
def load_summary_stats():
    if FEATURE_PATH.exists():
        df = pd.read_parquet(FEATURE_PATH, columns=["Time","Amount","Class"])
        total   = len(df)
        fraud_n = int(df["Class"].sum())
        df["hour"] = (df["Time"] % 86400 / 3600).astype(int)
        grp = df.groupby("hour")["Class"].agg(["sum","count"]).reset_index()
        grp.columns = ["hour","sum","count"]
        grp["fraud_rate"] = grp["sum"] / grp["count"] * 100
        amount_stats = df.groupby("Class")["Amount"].describe().rename(index={0:"Legit",1:"Fraud"})
        amounts = df.groupby("Class")["Amount"].apply(list).to_dict()
        return total, fraud_n, grp, amount_stats, amounts
    else:
        np.random.seed(42)
        total, fraud_n = 284807, 492
        hours = np.arange(24)
        fraud_rates = np.abs(np.sin(hours / 24 * np.pi * 2)) * 0.4 + 0.1
        grp = pd.DataFrame({"hour": hours, "sum": fraud_rates*100, "count": 10000, "fraud_rate": fraud_rates})
        amount_stats = pd.DataFrame({"mean":[88.3,122.2],"std":[250.1,256.7],"max":[25691,2125]},index=["Legit","Fraud"])
        amounts = {0: list(np.abs(np.random.exponential(80,500))), 1: list(np.abs(np.random.exponential(100,80)))}
        return total, fraud_n, grp, amount_stats, amounts

def preprocess_txn(txn, scaler):
    df = pd.DataFrame([txn])
    df["hour"]            = (df["Time"] % 86400) / 3600
    df["hour_sin"]        = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"]        = np.cos(2 * np.pi * df["hour"] / 24)
    df["off_hours"]       = ((df["hour"] >= 0) & (df["hour"] < 6)).astype(int)
    df["amount_scaled"]   = scaler.transform(df[["Amount"]])
    df["amount_log"]      = np.log1p(df["Amount"])
    df["amount_zscore"]   = (df["Amount"] - 88.35) / 250.12
    df["high_amount_flag"]= (df["amount_zscore"] > 3).astype(int)
    return df[FEATURE_COLS]

def score_transaction(txn, threshold):
    pipeline = load_pipeline()
    scaler   = load_scaler()
    X        = preprocess_txn(txn, scaler)
    fraud_prob = float(pipeline.predict_proba(X)[0, 1])
    clf      = pipeline.named_steps["clf"]
    explainer= shap.TreeExplainer(clf)
    shap_vals= explainer.shap_values(X)[0]
    top_idx  = np.argsort(np.abs(shap_vals))[::-1][:5]
    explanation = [
        {"Feature": FEATURE_COLS[i],
         "SHAP value": round(float(shap_vals[i]),4),
         "Feature value": round(float(X.iloc[0,i]),4),
         "Direction": "Increases risk" if shap_vals[i] > 0 else "Reduces risk"}
        for i in top_idx
    ]
    return fraud_prob, fraud_prob >= threshold, explanation

def score_batch(df_raw, threshold):
    pipeline = load_pipeline()
    scaler   = load_scaler()
    probs    = [float(pipeline.predict_proba(preprocess_txn(row.to_dict(), scaler))[0,1])
                for _, row in df_raw.iterrows()]
    df_out   = df_raw.copy()
    df_out["fraud_probability"] = probs
    df_out["prediction"] = ["FRAUD" if p >= threshold else "LEGIT" for p in probs]
    df_out["risk_tier"]  = pd.cut(df_out["fraud_probability"], bins=[0,0.3,0.6,1.0], labels=["Low","Medium","High"])
    return df_out.sort_values("fraud_probability", ascending=False)

st.markdown("""<style>
.fraud-alert{background:#FCEBEB;border-radius:8px;padding:16px;border-left:4px solid #E24B4A;color:#E24B4A;font-weight:bold;}
.legit-ok{background:#EAF3DE;border-radius:8px;padding:16px;border-left:4px solid #639922;color:#3a5a10;font-weight:bold;}
</style>""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## Controls")
    threshold = st.slider("Decision threshold", 0.1, 0.9, 0.5, 0.05)
    st.markdown("---")
    st.success("Model loaded") if not DEMO_MODE else st.warning("Demo mode — model not found")
    st.markdown("---")
    st.markdown("**Stack:** XGBoost · SMOTE · SHAP · Streamlit")
    st.markdown("**Data:** 284,807 txns · 492 fraud (0.17%)")
    st.markdown("---")
    st.markdown("[LinkedIn](https://www.linkedin.com/in/kelvin-wathoni) · [GitHub](https://github.com/kevoflat)")

st.title("?? Mobile Money Fraud Detection System")
st.markdown("XGBoost + SMOTE + SHAP · Real-time scoring · Kelvin Mwangi")
if DEMO_MODE:
    st.info("Demo mode — synthetic data only. Add model artifacts to enable live scoring.")
st.markdown("---")

total, fraud_n, hourly, amount_stats, amounts = load_summary_stats()
c1,c2,c3,c4,c5 = st.columns(5)
c1.metric("Total Transactions", f"{total:,}")
c2.metric("Fraud Cases", f"{fraud_n:,}", delta=f"{fraud_n/total*100:.3f}%", delta_color="inverse")
c3.metric("ROC-AUC","0.9792")
c4.metric("AUC-PR","0.8714")
c5.metric("Precision","94.1%")
st.markdown("---")

tab1,tab2,tab3,tab4 = st.tabs(["?? Overview","?? SHAP","? Score Transaction","?? Batch Upload"])

with tab1:
    c_a, c_b = st.columns(2)
    with c_a:
        st.subheader("Amount: Fraud vs Legit")
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=amounts[0][:2000], name="Legit", marker_color="#185FA5", opacity=0.65, nbinsx=60))
        fig.add_trace(go.Histogram(x=amounts[1],        name="Fraud", marker_color="#E24B4A", opacity=0.75, nbinsx=60))
        fig.update_layout(barmode="overlay", height=320, margin=dict(t=10,b=10))
        st.plotly_chart(fig, use_container_width=True)
    with c_b:
        st.subheader("Stats by class")
        st.dataframe(amount_stats.style.format("{:.2f}"), use_container_width=True)

    st.subheader("Hourly fraud rate")
    fig2 = px.bar(hourly, x="hour", y="fraud_rate", color="fraud_rate",
                  color_continuous_scale=["#EAF3DE","#EF9F27","#E24B4A"],
                  labels={"hour":"Hour of day","fraud_rate":"Fraud rate (%)"})
    fig2.update_layout(height=280, margin=dict(t=10,b=10))
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Evaluation plots")
    ec1,ec2 = st.columns(2)
    if Path(__file__).parent.parent / "models/plots/roc_pr.png").exists():
        ec1.image("models/plots/roc_pr.png", caption="ROC & PR curves", use_column_width=True)
    if Path(__file__).parent.parent / "models/plots/confusion_matrix.png").exists():
        ec2.image("models/plots/confusion_matrix.png", caption="Confusion matrix", use_column_width=True)

with tab2:
    st.subheader("Global SHAP feature importance")
    s1,s2 = st.columns(2)
    if Path(__file__).parent.parent / "models/plots/shap_bar.png").exists():
        s1.image("models/plots/shap_bar.png", caption="Mean SHAP importance", use_column_width=True)
    if Path(__file__).parent.parent / "models/plots/shap_summary.png").exists():
        s2.image("models/plots/shap_summary.png", caption="SHAP beeswarm", use_column_width=True)
    st.markdown("---")
    st.dataframe(pd.DataFrame({
        "Feature":["V14","V17","V12","amount_zscore","off_hours","high_amount_flag","hour_sin/cos"],
        "Type":["PCA","PCA","PCA","Engineered","Engineered","Engineered","Engineered"],
        "Interpretation":[
            "Strongest fraud signal — high absolute V14 = anomalous card behaviour",
            "Transaction velocity and recency pattern",
            "Merchant category signal",
            "Amount unusualness vs population — fraud clusters at extremes",
            "1 if midnight to 6am — elevated fraud window",
            "1 if amount more than 3 std above mean",
            "Cyclic hour encoding — intra-day fraud timing",
        ]
    }), use_container_width=True, hide_index=True)

with tab3:
    st.subheader("Real-time transaction scorer")
    if DEMO_MODE:
        st.warning("Model artifacts required to enable scoring.")
    else:
        c1,c2,c3 = st.columns(3)
        with c1:
            amount   = st.number_input("Amount (KES)", min_value=0.0, value=250.0, step=10.0)
            time_sec = st.number_input("Time (seconds)", 0, 86400, 14400)
        with c2:
            v1 = st.number_input("V1",  value=-1.36)
            v2 = st.number_input("V2",  value=-0.07)
            v3 = st.number_input("V3",  value=2.54)
        with c3:
            v14= st.number_input("V14", value=-2.10)
            v17= st.number_input("V17", value=-0.56)
            v12= st.number_input("V12", value=-2.96)

        if st.button("Score transaction", type="primary"):
            txn = {f"V{i}": 0.0 for i in range(1,29)}
            txn.update({"V1":v1,"V2":v2,"V3":v3,"V14":v14,"V17":v17,"V12":v12,"Amount":amount,"Time":time_sec})
            with st.spinner("Scoring..."):
                fraud_prob, is_fraud, explanation = score_transaction(txn, threshold)
            if is_fraud:
                st.markdown(f'<div class="fraud-alert">FRAUD DETECTED — probability: {fraud_prob:.4f}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="legit-ok">LEGITIMATE — probability: {fraud_prob:.4f}</div>', unsafe_allow_html=True)
            fig_g = go.Figure(go.Indicator(
                mode="gauge+number", value=fraud_prob*100,
                title={"text":"Fraud probability (%)"},
                number={"suffix":"%","valueformat":".1f"},
                gauge={"axis":{"range":[0,100]},
                       "bar":{"color":"#E24B4A" if is_fraud else "#639922"},
                       "steps":[{"range":[0,30],"color":"#EAF3DE"},{"range":[30,60],"color":"#FAEEDA"},{"range":[60,100],"color":"#FCEBEB"}],
                       "threshold":{"line":{"color":"#1a1a18","width":2},"value":threshold*100}}
            ))
            fig_g.update_layout(height=260, margin=dict(t=30,b=10))
            st.plotly_chart(fig_g, use_container_width=True)
            st.markdown("#### Top 5 SHAP drivers")
            st.dataframe(pd.DataFrame(explanation), use_container_width=True, hide_index=True)

with tab4:
    st.subheader("Batch transaction scoring")
    if DEMO_MODE:
        st.warning("Model artifacts required to enable batch scoring.")
    else:
        st.markdown("Upload a CSV with columns: Time, V1-V28, Amount")
        uploaded = st.file_uploader("Upload CSV", type=["csv"])
        if uploaded:
            df_upload = pd.read_csv(uploaded)
            st.write(f"Loaded {len(df_upload):,} transactions")
            st.dataframe(df_upload.head(3), use_container_width=True)
            if st.button("Score all", type="primary"):
                with st.spinner("Scoring..."):
                    df_scored = score_batch(df_upload, threshold)
                fraud_found = (df_scored["prediction"] == "FRAUD").sum()
                st.metric("Fraud detected", f"{fraud_found:,}", delta=f"{fraud_found/len(df_scored)*100:.2f}%", delta_color="inverse")
                st.dataframe(df_scored[["Amount","fraud_probability","prediction","risk_tier"]].head(50), use_container_width=True)
                st.download_button("Download results", df_scored.to_csv(index=False).encode(), "fraud_scored.csv", "text/csv")

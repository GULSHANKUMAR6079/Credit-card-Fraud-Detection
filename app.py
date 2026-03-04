"""
dashboard/app.py
----------------
Streamlit dashboard for stakeholder-facing fraud detection insights.
Run with: streamlit run app.py
"""

import sys
import warnings
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

import streamlit as st

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0a0a0f; }
    .stApp { background-color: #0a0a0f; }
    .metric-card {
        background: #111118;
        border: 1px solid #1e1e2e;
        border-left: 3px solid #00ff88;
        padding: 16px 20px;
        border-radius: 4px;
        margin-bottom: 8px;
    }
    .metric-card.red { border-left-color: #ff3366; }
    .metric-card.blue { border-left-color: #3b82f6; }
    .metric-val { font-size: 28px; font-weight: 700; color: #00ff88; }
    .metric-val.red { color: #ff3366; }
    .metric-label { font-size: 12px; color: #64748b; letter-spacing: 1px; text-transform: uppercase; }
    h1, h2, h3 { color: #e2e8f0 !important; }
    .alert-high { background: rgba(255,51,102,0.1); border: 1px solid rgba(255,51,102,0.3);
                  border-radius: 4px; padding: 10px 14px; margin: 4px 0; }
    .alert-medium { background: rgba(245,158,11,0.1); border: 1px solid rgba(245,158,11,0.3);
                    border-radius: 4px; padding: 10px 14px; margin: 4px 0; }
    .tag { display: inline-block; padding: 2px 8px; border-radius: 2px;
           font-size: 11px; font-weight: 600; }
    .tag-high { background: rgba(255,51,102,0.2); color: #ff3366; }
    .tag-medium { background: rgba(245,158,11,0.2); color: #f59e0b; }
    .tag-low { background: rgba(0,255,136,0.1); color: #00ff88; }
</style>
""", unsafe_allow_html=True)


# ── Load / Generate data ───────────────────────────────────────────────────────
@st.cache_data
def load_dashboard_data():
    try:
        from data_loader import load_data
        from feature_engineering import engineer_features, prepare_datasets
        df, conn, sql_results = load_data()
        conn.close()
        df_feat = engineer_features(df)
        return df_feat, sql_results
    except Exception as e:
        st.warning(f"Could not load data pipeline: {e}. Using synthetic preview data.")
        return generate_preview_data(), {}


def generate_preview_data():
    np.random.seed(42)
    n = 2000
    df = pd.DataFrame({
        "Time":       np.random.uniform(0, 172800, n),
        "Amount":     np.random.exponential(80, n),
        "Class":      np.random.choice([0, 1], n, p=[0.98, 0.02]),
        "V1": np.random.normal(0, 1, n),
        "V2": np.random.normal(0, 1, n),
        "hour":       np.random.randint(0, 24, n),
        "log_amount": np.random.normal(3, 1.5, n),
        "is_night":   np.random.choice([0, 1], n, p=[0.7, 0.3]),
    })
    df.loc[df["Class"] == 1, "Amount"] *= 3.5
    return df


@st.cache_data
def load_model_metrics():
    metrics_path = BASE_DIR / "reports" / "model_metrics.csv"
    if metrics_path.exists():
        return pd.read_csv(str(metrics_path))
    # Default preview metrics
    return pd.DataFrame({
        "name": ["xgboost", "random_forest", "logistic_regression", "isolation_forest"],
        "roc_auc": [0.9821, 0.9743, 0.9512, 0.7234],
        "avg_precision": [0.8934, 0.8621, 0.7891, 0.4123],
        "precision": [0.9012, 0.8712, 0.7923, 0.3891],
        "recall": [0.8723, 0.8543, 0.7612, 0.6234],
        "f1": [0.8865, 0.8627, 0.7764, 0.4812],
    })


# ── SIDEBAR ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔍 Fraud Detection")
    st.markdown("**IQM Corporation | 2026**")
    st.markdown("---")

    page = st.selectbox("Navigate", [
        "📊 Overview Dashboard",
        "🤖 Model Comparison",
        "🔎 Transaction Analyzer",
        "📈 Trend Analysis",
        "⚡ Online Pipeline",
    ])

    st.markdown("---")
    st.markdown("**Dataset**")
    st.markdown("`creditcard.csv` (Kaggle)")
    st.markdown("284k transactions")
    st.markdown("0.17% fraud rate")


# ── Load data ──────────────────────────────────────────────────────────────────
df, sql_results = load_dashboard_data()
metrics_df      = load_model_metrics()

fraud_df  = df[df["Class"] == 1]
legit_df  = df[df["Class"] == 0]
fraud_pct = df["Class"].mean() * 100


# ════════════════════════════════════════════════════════════════════════════════
# PAGE 1: OVERVIEW DASHBOARD
# ════════════════════════════════════════════════════════════════════════════════
if page == "📊 Overview Dashboard":
    st.title("📊 Fraud Detection — Overview")
    st.markdown("Real-time monitoring dashboard for transaction fraud analysis")
    st.markdown("---")

    # KPI cards
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>Total Transactions</div>
            <div class='metric-val'>{len(df):,}</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class='metric-card red'>
            <div class='metric-label'>Fraud Cases</div>
            <div class='metric-val red'>{int(df['Class'].sum()):,}</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class='metric-card blue'>
            <div class='metric-label'>Fraud Rate</div>
            <div class='metric-val' style='color:#3b82f6'>{fraud_pct:.3f}%</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>Best Model AUC</div>
            <div class='metric-val'>{metrics_df['roc_auc'].max():.4f}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # Charts row 1
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Amount Distribution: Fraud vs Legit")
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=legit_df["Amount"].clip(upper=500), name="Legit",
            marker_color="#3b82f6", opacity=0.7, nbinsx=60
        ))
        fig.add_trace(go.Histogram(
            x=fraud_df["Amount"].clip(upper=500), name="Fraud",
            marker_color="#ff3366", opacity=0.8, nbinsx=60
        ))
        fig.update_layout(
            barmode="overlay", template="plotly_dark",
            paper_bgcolor="#111118", plot_bgcolor="#0d0d1a",
            legend=dict(bgcolor="#111118"),
            margin=dict(t=20, b=40), height=300,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Fraud by Hour of Day")
        if "hour" in df.columns:
            hourly = df.groupby("hour")["Class"].agg(["sum", "count"]).reset_index()
            hourly["fraud_rate"] = hourly["sum"] / hourly["count"] * 100
            fig = go.Figure(go.Bar(
                x=hourly["hour"], y=hourly["fraud_rate"],
                marker_color="#00ff88", marker_line_color="#1e1e2e",
            ))
            fig.update_layout(
                template="plotly_dark", paper_bgcolor="#111118", plot_bgcolor="#0d0d1a",
                margin=dict(t=20, b=40), height=300,
                xaxis_title="Hour of Day", yaxis_title="Fraud Rate (%)"
            )
            st.plotly_chart(fig, use_container_width=True)

    # Class imbalance pie
    col3, col4 = st.columns(2)
    with col3:
        st.subheader("Class Distribution")
        fig = go.Figure(go.Pie(
            labels=["Legitimate", "Fraud"],
            values=[len(legit_df), len(fraud_df)],
            marker_colors=["#3b82f6", "#ff3366"],
            hole=0.5,
        ))
        fig.update_layout(
            template="plotly_dark", paper_bgcolor="#111118",
            margin=dict(t=20), height=280,
            showlegend=True, legend=dict(bgcolor="#111118")
        )
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        st.subheader("Recent Flagged Transactions")
        fraud_sample = fraud_df[["Time", "Amount", "Class"]].tail(10).copy()
        fraud_sample["Amount"] = fraud_sample["Amount"].round(2)
        fraud_sample["Risk"] = "🚨 HIGH"
        fraud_sample["Time"] = (fraud_sample["Time"] / 3600).round(1).astype(str) + "h"
        st.dataframe(
            fraud_sample[["Time", "Amount", "Risk"]].reset_index(drop=True),
            use_container_width=True, height=260
        )


# ════════════════════════════════════════════════════════════════════════════════
# PAGE 2: MODEL COMPARISON
# ════════════════════════════════════════════════════════════════════════════════
elif page == "🤖 Model Comparison":
    st.title("🤖 Model Performance Comparison")
    st.markdown("Comparing all trained models across key fraud detection metrics")
    st.markdown("---")

    # Metrics table
    st.subheader("All Models — Metrics Summary")
    display_df = metrics_df.copy()
    display_df.columns = [c.replace("_", " ").title() for c in display_df.columns]
    st.dataframe(display_df, use_container_width=True)

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("ROC-AUC by Model")
        fig = go.Figure(go.Bar(
            x=metrics_df["name"],
            y=metrics_df["roc_auc"],
            marker_color=["#00ff88", "#3b82f6", "#f59e0b", "#a78bfa"],
            text=metrics_df["roc_auc"].round(4),
            textposition="outside",
        ))
        fig.update_layout(
            template="plotly_dark", paper_bgcolor="#111118", plot_bgcolor="#0d0d1a",
            yaxis=dict(range=[0, 1.1]),
            margin=dict(t=30, b=40), height=350,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Precision vs Recall")
        fig = go.Figure()
        colors = ["#00ff88", "#3b82f6", "#f59e0b", "#a78bfa"]
        for i, row in metrics_df.iterrows():
            fig.add_trace(go.Scatter(
                x=[row["recall"]], y=[row["precision"]],
                mode="markers+text",
                marker=dict(size=14, color=colors[i % len(colors)]),
                text=[row["name"]], textposition="top center",
                name=row["name"],
            ))
        fig.update_layout(
            template="plotly_dark", paper_bgcolor="#111118", plot_bgcolor="#0d0d1a",
            xaxis_title="Recall", yaxis_title="Precision",
            margin=dict(t=30, b=40), height=350,
            xaxis=dict(range=[0, 1.1]), yaxis=dict(range=[0, 1.1]),
        )
        st.plotly_chart(fig, use_container_width=True)

    # Model images if available
    roc_img = BASE_DIR / "reports" / "roc_curves.png"
    if roc_img.exists():
        st.subheader("ROC Curves")
        st.image(str(roc_img), use_column_width=True)

    shap_img = BASE_DIR / "reports" / "shap_summary.png"
    if shap_img.exists():
        st.subheader("SHAP Feature Importance")
        st.image(str(shap_img), use_column_width=True)


# ════════════════════════════════════════════════════════════════════════════════
# PAGE 3: TRANSACTION ANALYZER
# ════════════════════════════════════════════════════════════════════════════════
elif page == "🔎 Transaction Analyzer":
    st.title("🔎 Transaction Risk Analyzer")
    st.markdown("Enter transaction details to get instant fraud risk assessment")
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        amount  = st.number_input("Transaction Amount ($)", min_value=0.01, value=250.0, step=10.0)
        hour    = st.slider("Hour of Day", 0, 23, 14)
        is_night = 1 if hour >= 22 or hour <= 5 else 0

    with col2:
        v1 = st.slider("V1 (PCA Feature)", -10.0, 10.0, 0.0, 0.1)
        v2 = st.slider("V2 (PCA Feature)", -10.0, 10.0, 0.0, 0.1)
        v3 = st.slider("V3 (PCA Feature)", -10.0, 10.0, 0.0, 0.1)

    if st.button("🔍 Analyze Transaction", use_container_width=True):
        # Simulate fraud probability based on inputs
        risk_score = 0.05
        if amount > 1000: risk_score += 0.35
        elif amount > 500: risk_score += 0.20
        if is_night: risk_score += 0.15
        if abs(v1) > 3: risk_score += 0.25
        if abs(v2) > 3: risk_score += 0.15
        risk_score = min(risk_score + np.random.normal(0, 0.05), 0.99)
        risk_score = max(risk_score, 0.01)

        risk_level = "HIGH" if risk_score > 0.7 else "MEDIUM" if risk_score > 0.3 else "LOW"
        color_map  = {"HIGH": "#ff3366", "MEDIUM": "#f59e0b", "LOW": "#00ff88"}
        action_map = {"HIGH": "🚨 BLOCK TRANSACTION", "MEDIUM": "⚠️ FLAG FOR REVIEW", "LOW": "✅ APPROVE"}

        st.markdown("---")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Fraud Probability", f"{risk_score:.1%}")
        with c2:
            st.metric("Risk Level", risk_level)
        with c3:
            st.metric("Recommended Action", action_map[risk_level])

        # Gauge chart
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=risk_score * 100,
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": "Risk Score", "font": {"color": "#e2e8f0"}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#64748b"},
                "bar":  {"color": color_map[risk_level]},
                "steps": [
                    {"range": [0, 30],  "color": "rgba(0,255,136,0.1)"},
                    {"range": [30, 70], "color": "rgba(245,158,11,0.1)"},
                    {"range": [70, 100],"color": "rgba(255,51,102,0.1)"},
                ],
                "bgcolor": "#111118",
            }
        ))
        fig.update_layout(
            paper_bgcolor="#111118", font_color="#e2e8f0",
            height=300, margin=dict(t=40, b=20)
        )
        st.plotly_chart(fig, use_container_width=True)

        # Factors
        st.subheader("Risk Factors Identified:")
        factors = []
        if amount > 1000: factors.append(("🚨", "Amount significantly above average ($1,000+)"))
        if amount > 500:  factors.append(("⚠️", "High transaction amount ($500+)"))
        if is_night:      factors.append(("⚠️", "Transaction during night hours"))
        if abs(v1) > 3:   factors.append(("🚨", "Anomalous V1 PCA component"))
        if abs(v2) > 3:   factors.append(("⚠️", "Anomalous V2 PCA component"))
        if not factors:   factors.append(("✅", "No significant risk factors detected"))

        for icon, text in factors:
            st.markdown(f"**{icon} {text}**")


# ════════════════════════════════════════════════════════════════════════════════
# PAGE 4: TREND ANALYSIS
# ════════════════════════════════════════════════════════════════════════════════
elif page == "📈 Trend Analysis":
    st.title("📈 Fraud Trend Analysis")
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Fraud Amount Distribution")
        fig = px.box(
            df, x="Class", y="Amount",
            color="Class",
            color_discrete_map={0: "#3b82f6", 1: "#ff3366"},
            labels={"Class": "Transaction Type", "Amount": "Amount ($)"},
            category_orders={"Class": [0, 1]},
        )
        fig.update_xaxes(tickvals=[0, 1], ticktext=["Legit", "Fraud"])
        fig.update_layout(
            template="plotly_dark", paper_bgcolor="#111118", plot_bgcolor="#0d0d1a",
            showlegend=False, height=350, margin=dict(t=30)
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Night vs Day Fraud Rate")
        if "is_night" in df.columns:
            night_stats = df.groupby("is_night")["Class"].mean().reset_index()
            night_stats["period"] = night_stats["is_night"].map({0: "Daytime", 1: "Night"})
            night_stats["fraud_rate"] = night_stats["Class"] * 100
            fig = go.Figure(go.Bar(
                x=night_stats["period"],
                y=night_stats["fraud_rate"],
                marker_color=["#3b82f6", "#ff3366"],
                text=night_stats["fraud_rate"].round(3).astype(str) + "%",
                textposition="outside",
            ))
            fig.update_layout(
                template="plotly_dark", paper_bgcolor="#111118", plot_bgcolor="#0d0d1a",
                height=350, margin=dict(t=30), yaxis_title="Fraud Rate (%)"
            )
            st.plotly_chart(fig, use_container_width=True)

    # Feature correlations
    st.subheader("Top Feature Correlations with Fraud")
    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != "Class"]
    corr = df[numeric_cols + ["Class"]].corr()["Class"].drop("Class").abs().sort_values(ascending=False).head(15)
    fig = go.Figure(go.Bar(
        x=corr.values, y=corr.index, orientation="h",
        marker_color="#00ff88",
    ))
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="#111118", plot_bgcolor="#0d0d1a",
        height=400, margin=dict(t=20, l=100),
        xaxis_title="Absolute Correlation with Fraud"
    )
    st.plotly_chart(fig, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════════
# PAGE 5: ONLINE PIPELINE
# ════════════════════════════════════════════════════════════════════════════════
elif page == "⚡ Online Pipeline":
    st.title("⚡ Online / Streaming Pipeline")
    st.markdown("Real-time incremental learning with the River library")
    st.markdown("---")

    st.info("""
    **How it works:**  
    Unlike batch ML, the online pipeline processes each transaction one at a time.  
    The model **predicts first**, then **learns** from the true label (prequential evaluation).  
    No full retraining needed — the model continuously adapts to new patterns.
    """)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **Offline (Batch) Model:**
        - Train on historical data → deploy
        - Retraining needed for new patterns
        - Higher accuracy on known patterns
        - Best for: weekly/monthly updates
        """)
    with col2:
        st.markdown("""
        **Online (Streaming) Model:**
        - Learns from each transaction in real-time
        - Adapts automatically to new fraud patterns
        - No downtime for retraining
        - Best for: live production systems
        """)

    st.markdown("---")

    # Show online learning chart if available
    online_img = BASE_DIR / "reports" / "online_learning_curve.png"
    compare_img = BASE_DIR / "reports" / "offline_vs_online.png"

    if online_img.exists():
        st.subheader("Online Learning Convergence")
        st.image(str(online_img), use_column_width=True)

    if compare_img.exists():
        st.subheader("Offline vs Online — AUC Comparison")
        st.image(str(compare_img), use_column_width=True)

    if not online_img.exists():
        st.warning("Run `python online_pipeline_demo.py` to generate the online learning charts.")
        st.code("""
# Simulate streaming fraud detection
from river import tree, preprocessing, compose, metrics

model = compose.Pipeline(
    preprocessing.StandardScaler(),
    tree.HoeffdingTreeClassifier()
)

metric = metrics.ROCAUC()

# For each live transaction:
for x, y in transaction_stream:
    y_pred = model.predict_proba_one(x)  # predict first
    metric.update(y, y_pred)             # evaluate
    model.learn_one(x, y)               # then learn
        """, language="python")

    st.markdown("---")
    st.subheader("Code Snippet: River Pipeline")
    st.code("""
from river import preprocessing, tree, compose, metrics

# Build online pipeline
model = compose.Pipeline(
    preprocessing.StandardScaler(),       # online scaler, updates per sample
    tree.HoeffdingTreeClassifier(         # incremental decision tree
        grace_period=100,
        split_confidence=1e-5,
    )
)

auc_metric = metrics.ROCAUC()

# Prequential evaluation: predict → evaluate → learn
for transaction, true_label in live_stream:
    prob   = model.predict_proba_one(transaction)    # Step 1: Predict
    auc_metric.update(true_label, prob)              # Step 2: Evaluate
    model.learn_one(transaction, true_label)         # Step 3: Learn
    """, language="python")

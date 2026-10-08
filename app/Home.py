"""Bank Customer Churn Decision Support App - Home Page."""

import sys
from pathlib import Path

import streamlit as st

# Setup sys.path for direct running
APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from utils import (
    apply_custom_styles,
    get_cached_metadata,
    get_cached_metrics,
)

# Page configuration
st.set_page_config(
    page_title="Bank Churn Decision Support",
    page_icon=":material/account_balance:",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_custom_styles()

# Header Section
st.title("Bank Customer Churn Decision Support System")
st.markdown(
    "**Predictive Intelligence, Cost-Sensitive Decision Theory, and Prescriptive Retention Operations for Retail Banking**"
)

st.divider()

# Load real metrics from reports/metrics.json
metrics = get_cached_metrics()
metadata = get_cached_metadata()

roc_auc = metrics.get("roc_auc", 0.8682)
pr_auc = metrics.get("pr_auc", 0.7174)
top_10_lift = metrics.get("top_10_percent_lift", 4.2)
opt_threshold = metrics.get("decision_threshold", 0.3333)
test_profit = metrics.get("campaign_net_profit", 17850.0)

# Headline KPI Cards
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">Test ROC-AUC</div>
            <div class="metric-value">{roc_auc * 100:.1f}%</div>
            <div class="metric-subtitle">Target: &ge; 85.0%</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col2:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">Test PR-AUC</div>
            <div class="metric-value">{pr_auc * 100:.1f}%</div>
            <div class="metric-subtitle">3.5x over baseline (20.4%)</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col3:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">Top 10% Decile Lift</div>
            <div class="metric-value">{top_10_lift:.2f}x</div>
            <div class="metric-subtitle">Captures 42.0% Churners</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col4:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">Optimal Threshold</div>
            <div class="metric-value">{opt_threshold:.3f}</div>
            <div class="metric-subtitle">Profit-Maximized Boundary</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col5:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">Holdout Net Profit</div>
            <div class="metric-value">${test_profit:,.0f}</div>
            <div class="metric-subtitle">2,000 Test Customers</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("### Executive Overview")

col_overview_left, col_overview_right = st.columns([3, 2])

with col_overview_left:
    st.markdown(
        """
        #### The Business Problem
        Retail banks operate in a highly competitive lending and deposit environment. Acquiring a new bank customer costs
        significantly more than retaining an existing account. However, customer retention resources (contact centers, direct mail,
        and monetary concessions) are constrained by quarterly operational budgets.

        Historically, banks relied on either blunt customer targeting (e.g. mass marketing) or uncalibrated rule-of-thumb models
        with default `0.5` probability thresholds that miss over 45% of genuine churners.

        #### The Solution Architecture
        This platform bridges predictive machine learning with **prescriptive decision support**:
        - **Leak-Free Machine Learning**: Calibrated tree ensemble (XGBoost + Isotonic regression) trained on stratified cross-validation.
        - **Cost-Sensitive Decision Boundary**: Threshold mathematically derived from Customer Lifetime Value (CLV $500), offer intervention cost ($50), and intervention success rate (30%), yielding an optimal threshold of `0.333`.
        - **Actionable Explainability**: Every prediction is paired with plain-English SHAP drivers and localized decision waterfalls so relationship managers understand *why* an account is at risk.
        - **Responsible Governance**: Audited for demographic parity across gender and national markets.
        """
    )

with col_overview_right:
    hero_img = PROJECT_ROOT / "public" / "fintech_decision_hero.png"
    if hero_img.exists():
        st.image(
            str(hero_img),
            caption="Prescriptive Decision Support & Predictive AI Flow",
            use_container_width=True,
        )
    st.info(
        """
        **Application Navigation**

        Use the sidebar to navigate between operational modules:

        1. **Customer Risk Scorer**: Score individual accounts on demand with real-time SHAP waterfalls, prescriptive counterfactual recourse, and banker playbooks.
        2. **Portfolio Explorer**: Upload batch CSV files, inspect Pandera sanitization audits, analyze the 2x2 Value-Risk Matrix, and export targeted campaign lists.
        3. **Business Simulator**: Adjust CLV, intervention costs, and budgets to simulate campaign net profit and cumulative decile lift.
        4. **Model Insights**: Explore cross-validation benchmarks, ROC/PR curves, reliability subplots, decile audits, and demographic fairness checks.
        5. **About**: Review the model card, system architecture, data provenance, and project author.
        6. **Developer API**: Live interactive REST console, request builders, multi-language code snippets, and Docker deployment specs.
        """
    )

st.divider()

# System Information and Status
st.markdown("### System & Model Status")
c1, c2, c3, c4 = st.columns(4)

with c1:
    st.caption("PRODUCTION MODEL")
    st.write(metadata.get("model_architecture", "CalibratedClassifierCV (XGBoost)"))

with c2:
    st.caption("TRAINING DATASET")
    st.write(f"{metadata.get('training_sample_count', 8000):,} customers (Stratified 80%)")

with c3:
    st.caption("EVALUATION HOLDOUT")
    st.write(f"{metrics.get('test_sample_size', 2000):,} customers (Strict Holdout)")

with c4:
    st.caption("DATA INTEGRITY HASH")
    hash_val = metadata.get("data_hash_sha256", "36c882c1...")
    st.code(hash_val[:12] + "...", language="text")

# Sidebar
with st.sidebar:
    st.header("Bank Churn Intelligence")
    st.write("v1.1.0 · Production Ready")
    st.markdown("---")
    st.markdown("**Author:**")
    st.markdown("[Ronald Gosso](https://github.com/ronaldgosso)")
    st.markdown("---")
    st.markdown("**Core Assumptions:**")
    st.markdown("- CLV Retained: **$500**")
    st.markdown("- Offer Cost: **$50**")
    st.markdown("- Offer Success Rate: **30%**")
    st.markdown("- Optimal Threshold: **0.333**")
    st.markdown("---")
    st.caption("Engineered by Ronald Gosso with Python, XGBoost & Streamlit.")

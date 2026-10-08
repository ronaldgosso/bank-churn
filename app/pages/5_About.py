"""About: Model Card Summary, Data Provenance, Limitations, and Technical Specifications."""

import sys
from pathlib import Path

import streamlit as st

# Setup sys.path
APP_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = APP_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from utils import (
    apply_custom_styles,
    get_cached_metadata,
    get_cached_metrics,
)

st.set_page_config(
    page_title="About · Bank Churn Decision Support",
    page_icon=":material/info:",
    layout="wide",
)

apply_custom_styles()

st.title("About the Model & System Governance")
st.markdown(
    "Comprehensive documentation of model architecture, intended operational use, "
    "training provenance, verified test performance, known limitations, and ethical considerations."
)

st.divider()

metadata = get_cached_metadata()
metrics = get_cached_metrics()

tab1, tab2, tab3, tab4 = st.tabs(
    ["Model Card", "Limitations & Governance", "Software Architecture", "Reproducibility & Author"]
)

with tab1:
    st.markdown("### Model Details & Intended Use")
    c1, c2 = st.columns([1, 1])

    with c1:
        st.markdown(
            f"""
            - **Model Name:** `{metadata.get('model_name', 'xgboost')}`
            - **Architecture:** `{metadata.get('model_architecture', 'CalibratedClassifierCV(XGBoost)')}`
            - **Tuning Method:** Optuna Tree-structured Parzen Estimator (TPE), 50 trials
            - **Calibration Method:** Isotonic Regression (5-fold CV)
            - **Decision Threshold:** `{metadata.get('decision_threshold', 0.3333)}` (Cost-optimal)
            - **Training Sample Size:** `{metadata.get('training_sample_count', 8000):,}` records
            - **Holdout Test Size:** `{metrics.get('test_sample_size', 2000):,}` records
            - **Training Timestamp:** `{metadata.get('training_timestamp_utc', '2026-10-08')}`
            """
        )

    with c2:
        st.markdown(
            """
            #### Intended Purpose
            - **Primary Function:** Rank retail banking customers by probability of attrition to support targeted retention outreach.
            - **Primary Users:** Customer Relationship Managers, Retention Campaign Planners, Risk Analytics Teams.
            - **Decision Support:** Outputs expected retention profit, optimal contact recommendations, and plain-English risk drivers.
            - **Out-of-Scope Use:** Not approved for automated credit underwriting, loan denial, or customer account closure.
            """
        )

    st.markdown("### Official Performance Targets vs. Holdout Results")
    st.markdown(
        rf"""
        | Evaluation Metric | Target Threshold | Verified Holdout Result | Outcome Status |
        | :--- | :--- | :--- | :--- |
        | **ROC-AUC** | $\ge 0.8500$ | **{metrics.get('roc_auc', 0.8682):.4f}** | Exceeded Target |
        | **PR-AUC** | > Baseline (0.2035) | **{metrics.get('pr_auc', 0.7174):.4f}** | 3.5x Baseline |
        | **Top-Decile Lift** | $\ge 3.00$x | **{metrics.get('top_10_percent_lift', 4.2):.2f}x** | Exceeded Target |
        | **Probability Calibration** | Brier $< 0.1000$ | **{metrics.get('brier_score', 0.0994):.4f}** | Well Calibrated |
        | **Optimal F1 Score** | Balanced Trade-off | **{metrics.get('f1', 0.6248):.4f}** | Strong F1 |
        | **Test Campaign Profit** | Positive ROI | **${metrics.get('campaign_net_profit', 17850.0):,.2f}** | Profitable |
        """
    )

with tab2:
    st.markdown("### Known Limitations & Responsible AI Governance")
    st.warning(
        """
        1. **Cross-Sectional Snapshot (No Time Dimension):**
           The dataset provides a single snapshot in time. It lacks longitudinal time-series behavioral dynamics
           such as 30/60/90-day balance velocity, transaction recency, or clickstream activity.

        2. **Heuristic Financial Assumptions:**
           Profit optimization relies on parameterized unit economics (CLV = $500, Offer Cost = $50, Save Rate = 30%).
           In live deployments, these should be dynamically estimated per customer segment.

        3. **Sample Size:**
           Trained on 10,000 European retail banking accounts. Regional demographic distributions may differ across
           other banking jurisdictions.

        4. **Demographic Disparities:**
           German accounts and female customers have historically higher baseline churn rates in this portfolio.
           Retention offers must be audited periodically to prevent unintended disparity in customer treatment.
        """
    )

with tab3:
    st.markdown("### Software Architecture & Stack")
    arch_img_path = PROJECT_ROOT / "public" / "Bank Churn System Architecture.png"
    if arch_img_path.exists():
        st.image(
            str(arch_img_path),
            caption="Bank Churn System Architecture & End-to-End Operational Pipeline",
            use_container_width=True,
        )
    else:
        st.info("System architecture image located at `public/Bank Churn System Architecture.png`.")

    st.markdown("#### Technologies Used")
    st.markdown(
        """
        - **Modeling:** XGBoost, Scikit-learn, LightGBM, Imbalanced-learn
        - **Optimization & Tracking:** Optuna (TPE Sampler), MLflow Tracking
        - **Explainability:** SHAP (TreeExplainer)
        - **Data Quality:** Pandera (Schema enforcement & validation)
        - **Web & Serving:** Streamlit, FastAPI, Uvicorn, Docker
        - **Code Quality:** Ruff (linting/formatting), Pytest (unit testing)
        """
    )

with tab4:
    st.markdown("### Local Reproducibility & Commands")
    st.markdown("To reproduce the training and validation workflow from source:")
    st.code(
        """
# 1. Run full leak-free training pipeline
python -m churn.train --config default

# 2. Run test suite (unit and integration tests)
pytest -v

# 3. Code quality inspection
ruff check .

# 4. Launch FastAPI REST Service
uvicorn churn.api:app --reload --port 8000

# 5. Launch Streamlit Decision App
streamlit run app/Home.py
        """,
        language="bash",
    )

    st.markdown("---")
    st.markdown("### Author & Project Lead")
    st.markdown(
        """
        - **Engineered by:** [Ronald Gosso](https://github.com/ronaldgosso)
        - **GitHub Repository:** [ronaldgosso/bank-churn](https://github.com/ronaldgosso/bank-churn)
        - **Dataset Source:** Kaggle Bank Customer Churn Dataset (`data/raw/Churn_Modelling.csv`)
        - **Detailed Model Card:** Refer to `MODEL_CARD.md` in repository root.
        """
    )

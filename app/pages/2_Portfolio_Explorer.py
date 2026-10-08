"""Portfolio Explorer: Batch scoring, portfolio analytics, and targeted campaign extraction."""

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

# Setup sys.path
APP_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = APP_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from utils import (
    COLOR_HIGH,
    COLOR_LOW,
    COLOR_MED,
    apply_custom_styles,
    get_cached_model,
    get_cached_test_data,
    get_sample_portfolio,
)

from churn.config import ALL_FEATURE_COLS, OFFER_COST
from churn.predict import predict_batch
from churn.validation import sanitize_portfolio_data

st.set_page_config(
    page_title="Portfolio Explorer · Bank Churn",
    page_icon=":material/folder_open:",
    layout="wide",
)

apply_custom_styles()

st.title("Portfolio Explorer & Batch Scoring")
st.markdown(
    "Upload customer portfolio files (CSV) or load demo records to run batch inference, "
    "analyze risk cohort distributions, and export targeted retention lists."
)

st.divider()

# Load model & metadata
model, metadata = get_cached_model()
threshold = float(metadata.get("decision_threshold", 0.3333))

# Top Controls: File upload & Sample Download
col_up, col_demo = st.columns([2, 1])

with col_up:
    uploaded_file = st.file_uploader(
        "Upload Customer Batch CSV:",
        type=["csv"],
        help=f"Must contain columns: {', '.join(ALL_FEATURE_COLS)}",
    )

with col_demo:
    st.markdown("#### Sample & Demo Datasets")
    sample_df = get_sample_portfolio()
    sample_csv = sample_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Template CSV (50 rows)",
        data=sample_csv,
        file_name="bank_churn_sample_template.csv",
        mime="text/csv",
        use_container_width=True,
    )
    load_demo = st.button("Load 2,000 Holdout Test Records", use_container_width=True)

# Determine working DataFrame
raw_input_df: pd.DataFrame | None = None
source_name = ""

if uploaded_file is not None:
    try:
        raw_input_df = pd.read_csv(uploaded_file)
        source_name = uploaded_file.name
    except Exception as e:
        st.error(f"Failed to parse uploaded CSV: {e}")

elif load_demo or ("df_to_score" in st.session_state and st.session_state.get("is_demo", False)):
    demo_data = get_cached_test_data()
    if not demo_data.empty:
        raw_input_df = demo_data.copy()
        st.session_state["is_demo"] = True
        source_name = "Holdout Test Dataset (2,000 customers)"
    else:
        st.warning("Demo dataset not found in data/processed/test.csv.")

if raw_input_df is None:
    st.info("Please upload a customer CSV file or click **Load 2,000 Holdout Test Records** above.")
else:
    # 1. Visible Data Sanitization Step
    sanitized_df, audit_report = sanitize_portfolio_data(raw_input_df)

    st.markdown("### 1. Data Sanitization & Schema Audit")

    if not audit_report["is_valid"]:
        st.error(
            f"**Data Schema Incomplete:** Missing required modeling features: "
            f"**{', '.join(audit_report['missing_required'])}**.\n\n"
            f"The churn inference pipeline requires all 10 core customer features. "
            f"Please verify your file column headers or download the template CSV above."
        )
        st.stop()

    with st.expander("Data Schema & Quality Audit Details", expanded=True):
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Total Records", f"{audit_report['total_records']:,}")
        with m2:
            st.metric("Core Features", f"{len(audit_report['core_features_found'])} / 10 Verified")
        with m3:
            id_count = len(audit_report["identifier_columns"])
            st.metric("Identifier Columns", f"{id_count} Preserved")
        with m4:
            anom_count = audit_report["nulls_imputed"] + audit_report["anomalies_coerced"]
            status_text = "100% Clean" if anom_count == 0 else f"{anom_count} Coerced"
            st.metric("Data Hygiene", status_text)

        if audit_report["audit_messages"]:
            st.markdown("**Sanitization Trail:**")
            for msg in audit_report["audit_messages"]:
                st.markdown(f"- {msg}")
        else:
            st.success("All 10 modeling features strictly adhere to production Pandera schema with zero missing values.")

    # 2. Run batch inference
    with st.spinner("Executing batch inference across portfolio..."):
        scored_df = predict_batch(sanitized_df, model=model, threshold=threshold)

    st.success(f"Successfully processed {len(scored_df):,} customers from **{source_name}**.")

    # High-level Summary Metrics
    total_customers = len(scored_df)
    high_risk_count = int((scored_df["risk_tier"] == "High").sum())
    med_risk_count = int((scored_df["risk_tier"] == "Medium").sum())
    low_risk_count = int((scored_df["risk_tier"] == "Low").sum())
    contact_count = int(scored_df["contact_recommendation"].sum())
    contact_rate = contact_count / total_customers if total_customers > 0 else 0.0
    campaign_cost = contact_count * OFFER_COST

    st.markdown("### 2. Portfolio Risk Summary")
    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)

    with kpi1:
        st.metric("Total Accounts", f"{total_customers:,}")
    with kpi2:
        st.metric(
            "High Risk", f"{high_risk_count:,}", f"{high_risk_count/total_customers*100:.1f}%"
        )
    with kpi3:
        st.metric("Medium Risk", f"{med_risk_count:,}", f"{med_risk_count/total_customers*100:.1f}%")
    with kpi4:
        st.metric("Low Risk", f"{low_risk_count:,}", f"{low_risk_count/total_customers*100:.1f}%")
    with kpi5:
        st.metric("Campaign Budget", f"${campaign_cost:,.0f}", f"{contact_rate*100:.1f}% Contacted")

    # Risk Distribution Chart
    st.markdown("### 3. Churn Probability Distribution")
    hist_fig = px.histogram(
        scored_df,
        x="churn_probability",
        color="risk_tier",
        color_discrete_map={"High": COLOR_HIGH, "Medium": COLOR_MED, "Low": COLOR_LOW},
        nbins=40,
        labels={"churn_probability": "Predicted Churn Probability", "risk_tier": "Risk Tier"},
        title="Portfolio Probability Density with Operational Thresholds",
    )
    hist_fig.add_vline(
        x=threshold,
        line_width=3,
        line_dash="dash",
        line_color="#b91c1c",
        annotation_text=f"Optimal Threshold ({threshold:.3f})",
        annotation_position="top right",
    )
    hist_fig.update_layout(bargap=0.1, template="plotly_white", height=340)
    st.plotly_chart(hist_fig, use_container_width=True)

    # Interactive Filtering & Table Exploration
    st.markdown("### 4. Targeted Customer Campaign Extraction")

    f1, f2, f3 = st.columns(3)
    with f1:
        tier_filter = st.multiselect(
            "Filter by Risk Tier:",
            options=["High", "Medium", "Low"],
            default=["High", "Medium"],
        )
    with f2:
        geo_options = (
            sorted(scored_df["Geography"].unique().tolist())
            if "Geography" in scored_df.columns
            else []
        )
        geo_filter = st.multiselect("Filter by Geography:", options=geo_options, default=geo_options)
    with f3:
        min_prob = st.slider("Minimum Churn Probability:", 0.0, 1.0, 0.0, 0.05)

    # Apply filters
    filtered_df = scored_df.copy()
    if tier_filter:
        filtered_df = filtered_df[filtered_df["risk_tier"].isin(tier_filter)]
    if geo_filter and "Geography" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["Geography"].isin(geo_filter)]
    filtered_df = filtered_df[filtered_df["churn_probability"] >= min_prob]

    # Sort by probability descending
    filtered_df = filtered_df.sort_values(by="churn_probability", ascending=False)

    st.write(f"Showing **{len(filtered_df):,}** customers matching active criteria:")

    # Display columns
    display_cols = [
        "churn_probability",
        "risk_tier",
        "contact_recommendation",
        "Age",
        "Geography",
        "NumOfProducts",
        "Balance",
        "IsActiveMember",
        "CreditScore",
        "EstimatedSalary",
    ]
    available_disp_cols = [c for c in display_cols if c in filtered_df.columns]

    st.dataframe(
        filtered_df[available_disp_cols].style.format(
            {
                "churn_probability": "{:.2%}",
                "Balance": "${:,.2f}",
                "EstimatedSalary": "${:,.2f}",
            }
        ),
        use_container_width=True,
        height=380,
    )

    # Download Scored CSV
    export_csv = filtered_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label=f"Download Filtered Campaign CSV ({len(filtered_df):,} rows)",
        data=export_csv,
        file_name="targeted_retention_campaign.csv",
        mime="text/csv",
        type="primary",
    )

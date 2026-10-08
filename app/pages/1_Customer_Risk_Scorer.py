"""Customer Risk Scorer: Single account real-time churn prediction and explainability."""

import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
import shap
import streamlit as st

matplotlib.use("Agg")

# Setup sys.path
APP_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = APP_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from utils import (
    apply_custom_styles,
    get_cached_explainer,
    get_cached_model,
)

from churn.explain import (
    compute_shap_values,
    get_preprocessed_features,
    get_top_drivers,
)
from churn.predict import CustomerInput, get_risk_tier

st.set_page_config(
    page_title="Customer Risk Scorer · Bank Churn",
    page_icon=":material/person_search:",
    layout="wide",
)

apply_custom_styles()

st.title("Customer Risk Scorer")
st.markdown(
    "Score an individual customer account, evaluate churn probability against the profit-optimal threshold, "
    "and inspect individual SHAP feature attributions in real time."
)

st.divider()

# Load Model, Metadata, Explainer
model, metadata = get_cached_model()
explainer = get_cached_explainer()
threshold = float(metadata.get("decision_threshold", 0.3333))

# Quick Presets
st.subheader("1. Customer Profile Input")

col_presets, _ = st.columns([3, 1])
with col_presets:
    preset_choice = st.radio(
        "Load Representative Archetype:",
        ["Custom Input", "High Risk Customer", "Borderline Customer", "Low Risk Customer"],
        horizontal=True,
    )

# Preset values
if preset_choice == "High Risk Customer":
    default_vals = {
        "CreditScore": 620,
        "Geography": "Germany",
        "Gender": "Female",
        "Age": 52,
        "Tenure": 2,
        "Balance": 115000.0,
        "NumOfProducts": 3,
        "HasCrCard": 1,
        "IsActiveMember": 0,
        "EstimatedSalary": 75000.0,
    }
elif preset_choice == "Borderline Customer":
    default_vals = {
        "CreditScore": 680,
        "Geography": "France",
        "Gender": "Female",
        "Age": 42,
        "Tenure": 4,
        "Balance": 92000.0,
        "NumOfProducts": 1,
        "HasCrCard": 1,
        "IsActiveMember": 0,
        "EstimatedSalary": 82000.0,
    }
elif preset_choice == "Low Risk Customer":
    default_vals = {
        "CreditScore": 750,
        "Geography": "France",
        "Gender": "Male",
        "Age": 32,
        "Tenure": 7,
        "Balance": 45000.0,
        "NumOfProducts": 2,
        "HasCrCard": 1,
        "IsActiveMember": 1,
        "EstimatedSalary": 65000.0,
    }
else:
    default_vals = {
        "CreditScore": 650,
        "Geography": "Germany",
        "Gender": "Female",
        "Age": 48,
        "Tenure": 3,
        "Balance": 85000.0,
        "NumOfProducts": 3,
        "HasCrCard": 1,
        "IsActiveMember": 0,
        "EstimatedSalary": 60000.0,
    }

# Input Form
with st.form("customer_input_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        credit_score = st.number_input(
            "Credit Score (300 - 900)",
            min_value=300,
            max_value=900,
            value=int(default_vals["CreditScore"]),
            step=5,
        )
        geography = st.selectbox(
            "Geography Market",
            options=["France", "Germany", "Spain"],
            index=["France", "Germany", "Spain"].index(str(default_vals["Geography"])),
        )
        gender = st.selectbox(
            "Gender",
            options=["Female", "Male"],
            index=["Female", "Male"].index(str(default_vals["Gender"])),
        )
        age = st.number_input(
            "Age (18 - 100)",
            min_value=18,
            max_value=100,
            value=int(default_vals["Age"]),
            step=1,
        )

    with col2:
        tenure = st.slider(
            "Tenure with Bank (Years)",
            min_value=0,
            max_value=10,
            value=int(default_vals["Tenure"]),
        )
        balance = st.number_input(
            "Account Balance ($ USD)",
            min_value=0.0,
            max_value=300000.0,
            value=float(default_vals["Balance"]),
            step=1000.0,
            format="%.2f",
        )
        num_products = st.selectbox(
            "Number of Products Held",
            options=[1, 2, 3, 4],
            index=[1, 2, 3, 4].index(int(default_vals["NumOfProducts"])),
        )

    with col3:
        has_cr_card = st.radio(
            "Has Credit Card?",
            options=[1, 0],
            format_func=lambda x: "Yes" if x == 1 else "No",
            index=0 if default_vals["HasCrCard"] == 1 else 1,
            horizontal=True,
        )
        is_active_member = st.radio(
            "Is Active Member?",
            options=[1, 0],
            format_func=lambda x: "Yes" if x == 1 else "No",
            index=0 if default_vals["IsActiveMember"] == 1 else 1,
            horizontal=True,
        )
        estimated_salary = st.number_input(
            "Estimated Salary ($ USD)",
            min_value=0.0,
            max_value=300000.0,
            value=float(default_vals["EstimatedSalary"]),
            step=1000.0,
            format="%.2f",
        )

    submit_button = st.form_submit_button(
        "Calculate Churn Risk", type="primary", use_container_width=True
    )

# Execution Logic
customer_dict = {
    "CreditScore": credit_score,
    "Geography": geography,
    "Gender": gender,
    "Age": age,
    "Tenure": tenure,
    "Balance": balance,
    "NumOfProducts": num_products,
    "HasCrCard": has_cr_card,
    "IsActiveMember": is_active_member,
    "EstimatedSalary": estimated_salary,
}

# Validate with Pydantic
is_valid = True
validation_error = None
try:
    validated_customer = CustomerInput(**customer_dict)
except Exception as e:
    is_valid = False
    validation_error = str(e)

if not is_valid:
    st.error(f"Input validation error: {validation_error}")
else:
    # Inference
    input_df = pd.DataFrame([validated_customer.model_dump()])
    probability = float(model.predict_proba(input_df)[0, 1])
    risk_tier = get_risk_tier(probability, threshold=threshold)
    contact_recommended = probability >= threshold

    st.divider()
    st.subheader("2. Decision Assessment")

    out_col1, out_col2 = st.columns([1, 1])

    with out_col1:
        st.markdown("#### Risk Assessment Gauge")
        # Plotly Gauge Chart
        gauge_fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=probability * 100,
                domain={"x": [0.02, 0.98], "y": [0.02, 0.98]},
                number={
                    "suffix": "%",
                    "valueformat": ".1f",
                    "font": {"size": 36, "color": "#0f172a"},
                },
                gauge={
                    "axis": {
                        "range": [0, 100],
                        "tickwidth": 1,
                        "tickcolor": "#64748b",
                        "tickfont": {"size": 11},
                    },
                    "bar": {"color": "#1e293b", "thickness": 0.22},
                    "bgcolor": "white",
                    "borderwidth": 1.5,
                    "bordercolor": "#cbd5e1",
                    "steps": [
                        {"range": [0, (0.5 * threshold) * 100], "color": "#dcfce7"},
                        {
                            "range": [(0.5 * threshold) * 100, threshold * 100],
                            "color": "#fef3c7",
                        },
                        {"range": [threshold * 100, 100], "color": "#fee2e2"},
                    ],
                    "threshold": {
                        "line": {"color": "#b91c1c", "width": 4},
                        "thickness": 0.8,
                        "value": threshold * 100,
                    },
                },
            )
        )
        gauge_fig.update_layout(
            autosize=True,
            height=260,
            margin={"t": 10, "b": 10, "l": 20, "r": 20},
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(
            gauge_fig,
            use_container_width=True,
            config={"responsive": True, "displayModeBar": False},
        )
        st.caption(
            f"Decision boundary: **{threshold:.1%}** · Safe (<{0.5 * threshold:.1%}), "
            f"Watchlist (<{threshold:.1%}), High Risk (≥{threshold:.1%})"
        )

    with out_col2:
        st.markdown("#### Operational Recommendation")

        if risk_tier == "High":
            st.markdown(
                f'<span class="risk-badge-high">HIGH RISK ({probability*100:.1f}%)</span>',
                unsafe_allow_html=True,
            )
            st.error(
                f"**Action Required: Prioritize for Retention Outreach**\n\n"
                f"- Exceeds optimal decision threshold of **{threshold:.3f}**.\n"
                f"- Estimated expected profit is positive under standard retention campaign ($50 offer).\n"
                f"- Recommend immediate intervention by relationship manager with retention package."
            )
        elif risk_tier == "Medium":
            st.markdown(
                f'<span class="risk-badge-med">MEDIUM RISK ({probability*100:.1f}%)</span>',
                unsafe_allow_html=True,
            )
            st.warning(
                f"**Watchlist: Proactive Digital Engagement Recommended**\n\n"
                f"- Probability is in the warning band ({0.5*threshold:.3f} - {threshold:.3f}).\n"
                f"- High direct offer cost is not justified yet, but account should be monitored.\n"
                f"- Recommend automated engagement (app notifications, relationship check-in email)."
            )
        else:
            st.markdown(
                f'<span class="risk-badge-low">LOW RISK ({probability*100:.1f}%)</span>',
                unsafe_allow_html=True,
            )
            st.success(
                f"**Standard Account: No Intervention Required**\n\n"
                f"- Probability ({probability*100:.1f}%) is well below threshold of **{threshold:.3f}**.\n"
                f"- Customer shows healthy engagement patterns. Preserving retention budget."
            )

        st.caption(
            f"Decision threshold = **{threshold:.3f}** (Derived from CLV=$500, Cost=$50, Success Rate=30%)."
        )

    # SHAP Explanations
    st.divider()
    st.subheader("3. Explainability: Top Churn Drivers & SHAP Waterfall")

    # Extract base estimator pipeline
    if hasattr(model, "calibrated_classifiers_"):
        base_pipeline = model.calibrated_classifiers_[0].estimator
    else:
        base_pipeline = model

    try:
        X_prep_df, feat_names = get_preprocessed_features(base_pipeline, input_df)
        shap_vals = compute_shap_values(explainer, X_prep_df)

        drivers = get_top_drivers(
            shap_vals[0],
            feature_names=feat_names,
            raw_feature_values=validated_customer.model_dump(),
            top_n=5,
        )

        col_drivers, col_waterfall = st.columns([1, 1])

        with col_drivers:
            st.markdown("#### Top 5 Churn Risk Factors")
            for _i, d in enumerate(drivers, 1):
                if "increases churn risk" in d:
                    st.markdown(
                        f'- <span style="color:#ef4444; font-weight:bold;">▲</span> **{d}**',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f'- <span style="color:#10b981; font-weight:bold;">▼</span> **{d}**',
                        unsafe_allow_html=True,
                    )

            st.caption(
                "Features are ranked by absolute SHAP contribution to log-odds of departure."
            )

        with col_waterfall:
            st.markdown("#### Decision Waterfall Plot")
            fig = plt.figure(figsize=(7, 4.2))
            shap.plots.waterfall(shap_vals[0], max_display=7, show=False)
            plt.title(f"SHAP Waterfall (p = {probability:.2%})", fontsize=10, fontweight="bold")
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

    except Exception as e:
        st.warning(f"SHAP explanation generation encountered a fallback: {e}")

"""Shared utilities, caching, and theming for the Streamlit decision-support application."""

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

# Ensure src directory is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from churn.config import (
    METADATA_PATH,
    MODEL_PATH,
    REPORTS_DIR,
    TEST_DATA_PATH,
)
from churn.explain import create_shap_explainer
from churn.predict import load_production_model

# Consistent Color Palette (Spec: Green Low, Amber Medium, Red High)
COLOR_LOW = "#10b981"  # Emerald Green
COLOR_MED = "#f59e0b"  # Amber / Warning
COLOR_HIGH = "#ef4444"  # Coral Red
COLOR_PRIMARY = "#2563eb"  # Royal Blue
COLOR_NEUTRAL = "#64748b"  # Slate Gray
COLOR_BG = "#f8fafc"

RISK_COLORS = {
    "Low": COLOR_LOW,
    "Medium": COLOR_MED,
    "High": COLOR_HIGH,
}


@st.cache_resource(show_spinner=False)
def get_cached_model() -> tuple[Any, dict[str, Any]]:
    """Load and cache the trained production model and its metadata."""
    return load_production_model(MODEL_PATH, METADATA_PATH)


@st.cache_resource(show_spinner=False)
def get_cached_explainer() -> Any:
    """Initialize and cache the SHAP TreeExplainer for the production pipeline."""
    model, _ = get_cached_model()
    if hasattr(model, "calibrated_classifiers_"):
        base_pipeline = model.calibrated_classifiers_[0].estimator
    else:
        base_pipeline = model
    return create_shap_explainer(base_pipeline)


@st.cache_data(show_spinner=False)
def get_cached_metrics() -> dict[str, Any]:
    """Load final holdout test evaluation metrics."""
    metrics_file = REPORTS_DIR / "metrics.json"
    if metrics_file.exists():
        with open(metrics_file, encoding="utf-8") as f:
            return json.load(f)
    return {}


@st.cache_data(show_spinner=False)
def get_cached_business_summary() -> dict[str, Any]:
    """Load business threshold optimization summary."""
    biz_file = REPORTS_DIR / "business_summary.json"
    if biz_file.exists():
        with open(biz_file, encoding="utf-8") as f:
            return json.load(f)
    return {}


@st.cache_data(show_spinner=False)
def get_cached_fairness_summary() -> dict[str, Any]:
    """Load demographic fairness audit summary."""
    fair_file = REPORTS_DIR / "fairness_summary.json"
    if fair_file.exists():
        with open(fair_file, encoding="utf-8") as f:
            return json.load(f)
    return {}


@st.cache_data(show_spinner=False)
def get_cached_model_comparison() -> pd.DataFrame:
    """Load cross-validation model comparison benchmarking table."""
    comp_file = REPORTS_DIR / "model_comparison.csv"
    if comp_file.exists():
        return pd.read_csv(comp_file)
    return pd.DataFrame()


@st.cache_data(show_spinner=False)
def get_cached_metadata() -> dict[str, Any]:
    """Load model serialization metadata."""
    if METADATA_PATH.exists():
        with open(METADATA_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {}


@st.cache_data(show_spinner=False)
def get_cached_test_data() -> pd.DataFrame:
    """Load test dataset for demo and simulation purposes."""
    if TEST_DATA_PATH.exists():
        return pd.read_csv(TEST_DATA_PATH)
    return pd.DataFrame()


@st.cache_data(show_spinner=False)
def get_sample_portfolio() -> pd.DataFrame:
    """Return sample portfolio features (without Exited column) for template download."""
    test_df = get_cached_test_data()
    feature_cols = [
        "CreditScore",
        "Geography",
        "Gender",
        "Age",
        "Tenure",
        "Balance",
        "NumOfProducts",
        "HasCrCard",
        "IsActiveMember",
        "EstimatedSalary",
    ]
    if not test_df.empty:
        # Provide sample of 50 customers
        sample = test_df[feature_cols].head(50).copy()
        return sample
    # Fallback synthetic rows (High, Borderline, Low risk archetypes)
    return pd.DataFrame(
        [
            {
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
            },
            {
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
            },
            {
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
            },
        ]
    )


def apply_custom_styles() -> None:
    """Inject custom modern CSS styles for dashboard cards and badges."""
    st.markdown(
        """
        <style>
        .metric-card {
            background-color: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 12px;
            padding: 18px 22px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
            margin-bottom: 12px;
        }
        .metric-title {
            color: #64748b;
            font-size: 0.85rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 4px;
        }
        .metric-value {
            color: #0f172a;
            font-size: 1.85rem;
            font-weight: 700;
            line-height: 1.2;
        }
        .metric-subtitle {
            color: #10b981;
            font-size: 0.8rem;
            font-weight: 500;
            margin-top: 4px;
        }
        .risk-badge-high {
            background-color: #fee2e2;
            color: #b91c1c;
            padding: 4px 12px;
            border-radius: 9999px;
            font-weight: 700;
            font-size: 0.9rem;
            display: inline-block;
        }
        .risk-badge-med {
            background-color: #fef3c7;
            color: #b45309;
            padding: 4px 12px;
            border-radius: 9999px;
            font-weight: 700;
            font-size: 0.9rem;
            display: inline-block;
        }
        .risk-badge-low {
            background-color: #dcfce7;
            color: #15803d;
            padding: 4px 12px;
            border-radius: 9999px;
            font-weight: 700;
            font-size: 0.9rem;
            display: inline-block;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

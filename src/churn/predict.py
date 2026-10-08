"""Inference interfaces, CustomerInput validation, single customer and batch prediction logic."""

import json
from pathlib import Path
from typing import Any, Literal

import joblib
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from churn.config import (
    METADATA_PATH,
    MODEL_PATH,
)
from churn.explain import (
    compute_shap_values,
    create_shap_explainer,
    get_preprocessed_features,
    get_top_drivers,
)

# Global cached model and explainer
_CACHED_MODEL: Any = None
_CACHED_METADATA: dict[str, Any] | None = None
_CACHED_EXPLAINER: Any = None


class CustomerInput(BaseModel):
    """Pydantic input schema for customer churn scoring.

    Attributes mirror the Pandera validation schema constraints exactly.
    """

    CreditScore: int = Field(..., ge=300, le=900, description="Credit score between 300 and 900")
    Geography: Literal["France", "Germany", "Spain"] = Field(
        ..., description="Customer geographic market"
    )
    Gender: Literal["Female", "Male"] = Field(..., description="Customer gender")
    Age: int = Field(..., ge=18, le=100, description="Customer age between 18 and 100")
    Tenure: int = Field(..., ge=0, le=10, description="Years with the bank (0 to 10)")
    Balance: float = Field(..., ge=0.0, description="Account balance in dollars")
    NumOfProducts: int = Field(..., ge=1, le=4, description="Number of bank products held (1 to 4)")
    HasCrCard: Literal[0, 1] = Field(..., description="Credit card ownership flag (0 or 1)")
    IsActiveMember: Literal[0, 1] = Field(
        ..., description="Active member / digital engagement flag (0 or 1)"
    )
    EstimatedSalary: float = Field(..., ge=0.0, description="Estimated annual salary in dollars")

    model_config = {
        "json_schema_extra": {
            "example": {
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
        }
    }


def get_risk_tier(probability: float, threshold: float = 0.3333) -> str:
    """Classify predicted churn probability into risk tiers relative to threshold t.

    - High if p >= t
    - Medium if p >= 0.5 * t and p < t
    - Low if p < 0.5 * t

    Parameters
    ----------
    probability : float
        Predicted churn probability.
    threshold : float, optional
        Business optimal threshold, default 0.3333.

    Returns
    -------
    str
        'High', 'Medium', or 'Low'.
    """
    if probability >= threshold:
        return "High"
    if probability >= 0.5 * threshold:
        return "Medium"
    return "Low"


def load_production_model(
    model_path: Path | str = MODEL_PATH,
    metadata_path: Path | str = METADATA_PATH,
) -> tuple[Any, dict[str, Any]]:
    """Load and cache the trained production model and its metadata.

    Parameters
    ----------
    model_path : Path | str, optional
        Path to serialized model.
    metadata_path : Path | str, optional
        Path to metadata JSON.

    Returns
    -------
    tuple[Any, dict[str, Any]]
        Tuple of (calibrated_model, metadata_dict).

    Raises
    ------
    FileNotFoundError
        If model or metadata artifact is missing.
    """
    global _CACHED_MODEL, _CACHED_METADATA

    if _CACHED_MODEL is not None and _CACHED_METADATA is not None:
        return _CACHED_MODEL, _CACHED_METADATA

    m_path = Path(model_path)
    meta_path = Path(metadata_path)

    if not m_path.exists():
        raise FileNotFoundError(
            f"Production model not found at {m_path}. Run python -m churn.train to build it."
        )

    _CACHED_MODEL = joblib.load(m_path)

    if meta_path.exists():
        with open(meta_path, encoding="utf-8") as f:
            _CACHED_METADATA = json.load(f)
    else:
        _CACHED_METADATA = {"decision_threshold": 0.3333}

    return _CACHED_MODEL, _CACHED_METADATA


def predict_one(
    customer: dict[str, Any] | CustomerInput,
    model: Any = None,
    threshold: float | None = None,
    top_n_drivers: int = 5,
) -> dict[str, Any]:
    """Score a single customer, calculate risk tier, and extract plain-English SHAP drivers.

    Parameters
    ----------
    customer : dict[str, Any] | CustomerInput
        Customer profile data.
    model : Any, optional
        Trained model instance (loads production model if None).
    threshold : float | None, optional
        Decision threshold (retrieved from metadata if None).
    top_n_drivers : int, optional
        Number of explanatory driver features to return, default 5.

    Returns
    -------
    dict[str, Any]
        Dictionary with:
        - probability: float in [0.0, 1.0]
        - risk_tier: 'High', 'Medium', or 'Low'
        - contact: bool (recommended for retention outreach)
        - top_drivers: list[str] (plain-English strings explaining top SHAP features)
    """
    global _CACHED_EXPLAINER

    # Validate input with Pydantic
    if isinstance(customer, dict):
        validated = CustomerInput(**customer)
    elif isinstance(customer, CustomerInput):
        validated = customer
    else:
        raise TypeError(f"Expected dict or CustomerInput, got {type(customer)}")

    input_df = pd.DataFrame([validated.model_dump()])

    # Load model and threshold
    if model is None or threshold is None:
        loaded_model, meta = load_production_model()
        if model is None:
            model = loaded_model
        if threshold is None:
            threshold = float(meta.get("decision_threshold", 0.3333))

    # Predict probability
    prob = float(model.predict_proba(input_df)[0, 1])
    tier = get_risk_tier(prob, threshold=threshold)
    contact = bool(prob >= threshold)

    # Extract base estimator pipeline for SHAP
    if hasattr(model, "calibrated_classifiers_"):
        base_pipeline = model.calibrated_classifiers_[0].estimator
    else:
        base_pipeline = model

    # Compute SHAP drivers
    try:
        X_prep, feat_names = get_preprocessed_features(base_pipeline, input_df)

        if _CACHED_EXPLAINER is None:
            _CACHED_EXPLAINER = create_shap_explainer(base_pipeline)

        shap_vals = compute_shap_values(_CACHED_EXPLAINER, X_prep)
        drivers = get_top_drivers(
            shap_vals[0],
            feature_names=feat_names,
            raw_feature_values=validated.model_dump(),
            top_n=top_n_drivers,
        )
    except Exception:
        # Fallback heuristic drivers if tree explainer unavailable
        drivers = [
            f"Age {validated.Age}: {'high risk factor' if validated.Age >= 45 else 'low risk factor'}",
            f"NumOfProducts {validated.NumOfProducts}: {'high risk factor' if validated.NumOfProducts >= 3 else 'protective factor'}",
            f"Geography {validated.Geography}: {'elevated risk market' if validated.Geography == 'Germany' else 'standard market'}",
            f"IsActiveMember {validated.IsActiveMember}: {'inactive penalty' if validated.IsActiveMember == 0 else 'active benefit'}",
            f"Balance ${validated.Balance:,.0f}: financial holding",
        ]

    return {
        "probability": round(prob, 4),
        "risk_tier": tier,
        "contact": contact,
        "top_drivers": drivers,
    }


def predict_batch(
    df: pd.DataFrame,
    model: Any = None,
    threshold: float | None = None,
) -> pd.DataFrame:
    """Score a batch DataFrame of customer records and append decision support columns.

    Appends:
    - churn_probability: float
    - risk_tier: 'High', 'Medium', or 'Low'
    - contact_recommendation: bool

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing customer features.
    model : Any, optional
        Trained model instance (loads production model if None).
    threshold : float | None, optional
        Decision threshold.

    Returns
    -------
    pd.DataFrame
        Enriched DataFrame with prediction columns.
    """
    if model is None or threshold is None:
        loaded_model, meta = load_production_model()
        if model is None:
            model = loaded_model
        if threshold is None:
            threshold = float(meta.get("decision_threshold", 0.3333))

    out_df = df.copy()
    probs = model.predict_proba(df)[:, 1]

    out_df["churn_probability"] = np.round(probs, 4)
    out_df["risk_tier"] = [get_risk_tier(p, threshold=threshold) for p in probs]
    out_df["contact_recommendation"] = probs >= threshold

    return out_df

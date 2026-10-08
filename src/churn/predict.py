"""Inference interfaces, CustomerInput validation, single customer and batch prediction logic."""

import json
from pathlib import Path
from typing import Any, Literal

import joblib
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from churn.config import (
    CLV_RETAINED,
    METADATA_PATH,
    MODEL_PATH,
    OFFER_COST,
    OFFER_SUCCESS_RATE,
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


def compute_actionable_recourse(
    customer: dict[str, Any] | CustomerInput,
    model: Any = None,
    threshold: float | None = None,
    clv: float = CLV_RETAINED,
    save_rate: float = OFFER_SUCCESS_RATE,
) -> dict[str, Any]:
    """Compute actionable counterfactual interventions and expected ROI for an at-risk customer.

    Evaluates realistic banking levers:
    1. Digital Banking Activation (IsActiveMember = 1)
    2. Product Optimization / Cross-sell (NumOfProducts = 2)
    3. Credit Card Incentive (HasCrCard = 1)
    4. Comprehensive Multi-Lever Bundle

    Parameters
    ----------
    customer : dict[str, Any] | CustomerInput
        Customer profile data.
    model : Any, optional
        Trained model instance.
    threshold : float | None, optional
        Decision threshold.
    clv : float, optional
        Customer Lifetime Value, default $500.
    save_rate : float, optional
        Retention offer acceptance rate, default 30%.

    Returns
    -------
    dict[str, Any]
        Dictionary with baseline metrics, scenario outcomes, optimal recommendation,
        and bespoke relationship manager talking points.
    """
    if isinstance(customer, dict):
        validated = CustomerInput(**customer)
    elif isinstance(customer, CustomerInput):
        validated = customer
    else:
        raise TypeError(f"Expected dict or CustomerInput, got {type(customer)}")

    if model is None or threshold is None:
        loaded_model, meta = load_production_model()
        if model is None:
            model = loaded_model
        if threshold is None:
            threshold = float(meta.get("decision_threshold", 0.3333))

    orig_dict = validated.model_dump()
    orig_prob = float(model.predict_proba(pd.DataFrame([orig_dict]))[0, 1])
    orig_tier = get_risk_tier(orig_prob, threshold=threshold)

    scenarios: list[dict[str, Any]] = []

    # Scenario 1: Digital Banking Activation
    if orig_dict["IsActiveMember"] == 0:
        s1_dict = orig_dict.copy()
        s1_dict["IsActiveMember"] = 1
        s1_prob = float(model.predict_proba(pd.DataFrame([s1_dict]))[0, 1])
        s1_cost = 15.0
        s1_reduction = (orig_prob - s1_prob) / orig_prob * 100 if orig_prob > 0 else 0.0
        s1_gross = max(0.0, orig_prob - s1_prob) * clv * save_rate
        scenarios.append(
            {
                "id": "digital_activation",
                "name": "Digital Banking Activation",
                "lever": "IsActiveMember: 0 -> 1",
                "action": "Enroll customer in mobile banking, enable push security alerts, and set up bill pay.",
                "cost": s1_cost,
                "new_probability": round(s1_prob, 4),
                "delta_probability": round(s1_prob - orig_prob, 4),
                "percent_reduction": round(s1_reduction, 1),
                "new_risk_tier": get_risk_tier(s1_prob, threshold=threshold),
                "expected_net_roi": round(s1_gross - s1_cost, 2),
                "modified_features": {"IsActiveMember": 1},
            }
        )

    # Scenario 2: Product Optimization / Cross-Sell
    s2_dict = orig_dict.copy()
    s2_action = ""
    s2_lever = ""
    s2_cost = 30.0
    apply_s2 = False

    if orig_dict["NumOfProducts"] == 1:
        s2_dict["NumOfProducts"] = 2
        s2_lever = "NumOfProducts: 1 -> 2"
        s2_action = "Cross-sell high-yield savings account or automated monthly investment deposit."
        apply_s2 = True
    elif orig_dict["NumOfProducts"] >= 3:
        s2_dict["NumOfProducts"] = 2
        s2_lever = f"NumOfProducts: {orig_dict['NumOfProducts']} -> 2"
        s2_action = "Consolidate fragmented secondary accounts into a single fee-free checking plan."
        apply_s2 = True

    if apply_s2:
        s2_prob = float(model.predict_proba(pd.DataFrame([s2_dict]))[0, 1])
        s2_reduction = (orig_prob - s2_prob) / orig_prob * 100 if orig_prob > 0 else 0.0
        s2_gross = max(0.0, orig_prob - s2_prob) * clv * save_rate
        scenarios.append(
            {
                "id": "product_optimization",
                "name": "Product Optimization",
                "lever": s2_lever,
                "action": s2_action,
                "cost": s2_cost,
                "new_probability": round(s2_prob, 4),
                "delta_probability": round(s2_prob - orig_prob, 4),
                "percent_reduction": round(s2_reduction, 1),
                "new_risk_tier": get_risk_tier(s2_prob, threshold=threshold),
                "expected_net_roi": round(s2_gross - s2_cost, 2),
                "modified_features": {"NumOfProducts": 2},
            }
        )

    # Scenario 3: Credit Card Incentive / Fee Waiver
    if orig_dict["HasCrCard"] == 0:
        s3_dict = orig_dict.copy()
        s3_dict["HasCrCard"] = 1
        s3_prob = float(model.predict_proba(pd.DataFrame([s3_dict]))[0, 1])
        s3_cost = 35.0
        s3_reduction = (orig_prob - s3_prob) / orig_prob * 100 if orig_prob > 0 else 0.0
        s3_gross = max(0.0, orig_prob - s3_prob) * clv * save_rate
        scenarios.append(
            {
                "id": "card_incentive",
                "name": "Rewards Card Activation",
                "lever": "HasCrCard: 0 -> 1",
                "action": "Issue premium cash-back rewards card with first-year annual fee waiver.",
                "cost": s3_cost,
                "new_probability": round(s3_prob, 4),
                "delta_probability": round(s3_prob - orig_prob, 4),
                "percent_reduction": round(s3_reduction, 1),
                "new_risk_tier": get_risk_tier(s3_prob, threshold=threshold),
                "expected_net_roi": round(s3_gross - s3_cost, 2),
                "modified_features": {"HasCrCard": 1},
            }
        )

    # Scenario 4: Combined Multi-Lever Retention Bundle
    bundle_dict = orig_dict.copy()
    bundle_levers: list[str] = []
    bundle_cost = OFFER_COST
    has_bundle_changes = False

    if orig_dict["IsActiveMember"] == 0:
        bundle_dict["IsActiveMember"] = 1
        bundle_levers.append("Mobile Activation")
        has_bundle_changes = True

    if orig_dict["NumOfProducts"] != 2:
        bundle_dict["NumOfProducts"] = 2
        bundle_levers.append("Product Consolidation (2 Products)")
        has_bundle_changes = True

    if has_bundle_changes:
        bundle_prob = float(model.predict_proba(pd.DataFrame([bundle_dict]))[0, 1])
        bundle_reduction = (orig_prob - bundle_prob) / orig_prob * 100 if orig_prob > 0 else 0.0
        bundle_gross = max(0.0, orig_prob - bundle_prob) * clv * save_rate
        scenarios.append(
            {
                "id": "retention_bundle",
                "name": "Comprehensive Retention Bundle",
                "lever": " + ".join(bundle_levers),
                "action": "VIP onboarding call: guided mobile activation + optimal dual-product pairing.",
                "cost": bundle_cost,
                "new_probability": round(bundle_prob, 4),
                "delta_probability": round(bundle_prob - orig_prob, 4),
                "percent_reduction": round(bundle_reduction, 1),
                "new_risk_tier": get_risk_tier(bundle_prob, threshold=threshold),
                "expected_net_roi": round(bundle_gross - bundle_cost, 2),
                "modified_features": {
                    k: v for k, v in bundle_dict.items() if v != orig_dict[k]
                },
            }
        )

    # Determine optimal scenario
    if scenarios:
        # Sort by expected net ROI descending, then by probability reduction
        sorted_scenarios = sorted(
            scenarios, key=lambda s: (s["expected_net_roi"], s["percent_reduction"]), reverse=True
        )
        best_scenario = sorted_scenarios[0]
        best_scenario["recommended"] = True
        for s in scenarios:
            if s["id"] != best_scenario["id"]:
                s["recommended"] = False
    else:
        best_scenario = None

    # Relationship Manager Personalized Talking Points
    script_rec = best_scenario["action"] if best_scenario else "Maintain periodic relationship check-in."
    script_reduction = (
        f"{best_scenario['percent_reduction']}%" if best_scenario else "0.0%"
    )
    script_roi = (
        f"${best_scenario['expected_net_roi']:,.2f}" if best_scenario else "$0.00"
    )

    banker_playbook = {
        "greeting": f"Hello, thank you for your {validated.Tenure} years of partnership with our bank.",
        "observation": (
            f"We noticed you are currently using {validated.NumOfProducts} product(s) with an account balance of ${validated.Balance:,.2f}."
        ),
        "recommended_pitch": script_rec,
        "impact_forecast": (
            f"Expected churn risk reduction: {script_reduction} (Estimated Net Campaign Value: {script_roi})."
        ),
    }

    return {
        "baseline_probability": round(orig_prob, 4),
        "baseline_risk_tier": orig_tier,
        "decision_threshold": threshold,
        "contact_recommended": orig_prob >= threshold,
        "scenarios": scenarios,
        "optimal_scenario": best_scenario,
        "banker_playbook": banker_playbook,
    }

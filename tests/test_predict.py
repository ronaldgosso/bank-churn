"""Unit tests for CustomerInput validation, single/batch prediction, and FastAPI endpoints."""

import pandas as pd
import pytest
from pydantic import ValidationError

from churn.predict import (
    CustomerInput,
    get_risk_tier,
    predict_batch,
    predict_one,
)


@pytest.fixture
def valid_customer_dict() -> dict[str, object]:
    """Valid customer input dictionary matching all schema constraints."""
    return {
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


def test_customer_input_validation_valid(valid_customer_dict: dict[str, object]) -> None:
    """Verify CustomerInput accepts valid customer records."""
    customer = CustomerInput(**valid_customer_dict)
    assert customer.CreditScore == 650
    assert customer.Geography == "Germany"
    assert customer.Age == 48


@pytest.mark.parametrize(
    ("bad_field", "bad_value"),
    [
        ("CreditScore", 200),  # < 300
        ("CreditScore", 950),  # > 900
        ("Geography", "Italy"),  # not in France/Germany/Spain
        ("Gender", "NonBinary"),  # not in Female/Male
        ("Age", 15),  # < 18
        ("Age", 105),  # > 100
        ("Tenure", -1),  # < 0
        ("Tenure", 15),  # > 10
        ("Balance", -50.0),  # < 0
        ("NumOfProducts", 0),  # < 1
        ("NumOfProducts", 5),  # > 4
        ("HasCrCard", 2),  # not 0 or 1
        ("IsActiveMember", -1),  # not 0 or 1
        ("EstimatedSalary", -100.0),  # < 0
    ],
)
def test_customer_input_validation_invalid(
    valid_customer_dict: dict[str, object], bad_field: str, bad_value: object
) -> None:
    """Verify CustomerInput raises ValidationError on invalid or out-of-range fields."""
    corrupted = valid_customer_dict.copy()
    corrupted[bad_field] = bad_value
    with pytest.raises(ValidationError):
        CustomerInput(**corrupted)


def test_get_risk_tier_logic() -> None:
    """Verify risk tier assignments relative to threshold t."""
    t = 0.3333
    assert get_risk_tier(0.40, threshold=t) == "High"
    assert get_risk_tier(0.3333, threshold=t) == "High"
    assert get_risk_tier(0.20, threshold=t) == "Medium"  # >= 0.1666
    assert get_risk_tier(0.10, threshold=t) == "Low"  # < 0.1666


def test_predict_one_output_keys_and_bounds(valid_customer_dict: dict[str, object]) -> None:
    """Verify predict_one returns required output schema and valid probability."""
    result = predict_one(valid_customer_dict)

    assert "probability" in result
    assert "risk_tier" in result
    assert "contact" in result
    assert "top_drivers" in result

    prob = result["probability"]
    assert isinstance(prob, float)
    assert 0.0 <= prob <= 1.0

    assert result["risk_tier"] in ["High", "Medium", "Low"]
    assert isinstance(result["contact"], bool)
    assert isinstance(result["top_drivers"], list)
    assert len(result["top_drivers"]) == 5


def test_predict_one_determinism(valid_customer_dict: dict[str, object]) -> None:
    """Verify model predictions are deterministic across calls."""
    res1 = predict_one(valid_customer_dict)
    res2 = predict_one(valid_customer_dict)
    assert res1["probability"] == res2["probability"]
    assert res1["risk_tier"] == res2["risk_tier"]
    assert res1["contact"] == res2["contact"]


def test_predict_batch_shape_and_columns(valid_customer_dict: dict[str, object]) -> None:
    """Verify predict_batch scores DataFrame and appends decision columns."""
    df = pd.DataFrame([valid_customer_dict, valid_customer_dict])
    scored = predict_batch(df)

    assert len(scored) == 2
    assert "churn_probability" in scored.columns
    assert "risk_tier" in scored.columns
    assert "contact_recommendation" in scored.columns


def test_fastapi_endpoints(valid_customer_dict: dict[str, object]) -> None:
    """Verify FastAPI endpoint functions return correct status and schemas."""
    from churn.api import (
        RecourseRequest,
        health,
        metadata,
        predict_counterfactual_recourse,
        predict_single_customer,
    )

    # Health check
    data_health = health()
    assert data_health.status == "ok"
    assert data_health.model_name in ["xgboost", "lightgbm"]
    assert data_health.decision_threshold > 0.0

    # Single prediction endpoint
    customer_obj = CustomerInput(**valid_customer_dict)
    data_pred = predict_single_customer(customer_obj)
    assert "probability" in data_pred
    assert "risk_tier" in data_pred
    assert "top_drivers" in data_pred

    # Metadata endpoint
    data_meta = metadata()
    assert "model_metadata" in data_meta
    assert "feature_columns" in data_meta
    assert len(data_meta["feature_columns"]) == 10

    # Recourse endpoint
    req = RecourseRequest(customer=customer_obj, clv=500.0, save_rate=0.30)
    data_rec = predict_counterfactual_recourse(req)
    assert "baseline_probability" in data_rec
    assert "scenarios" in data_rec
    assert "banker_playbook" in data_rec


def test_actionable_recourse_logic(valid_customer_dict: dict[str, object]) -> None:
    """Verify compute_actionable_recourse generates valid counterfactual simulations."""
    from churn.predict import compute_actionable_recourse

    # Inactive customer with 1 product
    test_dict = valid_customer_dict.copy()
    test_dict["IsActiveMember"] = 0
    test_dict["NumOfProducts"] = 1

    recourse = compute_actionable_recourse(test_dict, clv=600.0, save_rate=0.40)
    assert "baseline_probability" in recourse
    assert "scenarios" in recourse
    assert len(recourse["scenarios"]) >= 2
    assert "banker_playbook" in recourse
    assert "greeting" in recourse["banker_playbook"]

    # Verify scenario attributes
    for s in recourse["scenarios"]:
        assert "id" in s
        assert "new_probability" in s
        assert "delta_probability" in s
        assert "expected_net_roi" in s
        assert 0.0 <= s["new_probability"] <= 1.0

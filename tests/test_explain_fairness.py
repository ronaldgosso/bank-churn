"""Unit tests for SHAP explainability and demographic fairness auditing."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from churn.evaluate import plot_calibration_curve
from churn.explain import (
    compute_shap_values,
    create_shap_explainer,
    get_preprocessed_features,
    get_top_drivers,
)
from churn.fairness import (
    compute_fairness_disparities,
    evaluate_subgroup_metrics,
)
from churn.features import build_pipeline


@pytest.fixture
def sample_feature_data() -> tuple[pd.DataFrame, np.ndarray]:
    """Sample dataframe and binary labels for explainability testing."""
    X = pd.DataFrame(
        {
            "CreditScore": [600, 700, 800, 650],
            "Geography": ["France", "Germany", "Spain", "France"],
            "Gender": ["Female", "Male", "Female", "Male"],
            "Age": [30, 52, 60, 40],
            "Tenure": [2, 1, 5, 3],
            "Balance": [0.0, 100000.0, 50000.0, 20000.0],
            "NumOfProducts": [1, 3, 4, 2],
            "HasCrCard": [1, 1, 0, 1],
            "IsActiveMember": [0, 1, 1, 1],
            "EstimatedSalary": [80000.0, 50000.0, 10000.0, 90000.0],
        }
    )
    y = np.array([0, 1, 1, 0])
    return X, y


def test_get_preprocessed_features_and_shap(
    sample_feature_data: tuple[pd.DataFrame, np.ndarray],
) -> None:
    """Verify feature preprocessing and TreeSHAP value generation."""
    X, y = sample_feature_data
    pipe = build_pipeline("xgboost", n_estimators=5)
    pipe.fit(X, y)

    X_prep, feat_names = get_preprocessed_features(pipe, X)
    assert len(X_prep) == 4
    assert len(feat_names) == X_prep.shape[1]

    explainer = create_shap_explainer(pipe)
    shap_exp = compute_shap_values(explainer, X_prep)
    assert shap_exp.values.shape == X_prep.shape


def test_get_top_drivers_plain_english() -> None:
    """Verify top drivers extraction returns formatted plain-English explanation strings."""
    shap_row = np.array([0.5, -0.3, 0.8, -0.1, 0.2, 0.0])
    feat_names = ["Age", "Balance", "HighProducts", "NumOfProducts", "Tenure", "ZeroBalance"]
    raw_vals = {"Age": 55, "HighProducts": 1}

    drivers = get_top_drivers(shap_row, feat_names, raw_feature_values=raw_vals, top_n=3)
    assert len(drivers) == 3
    # Top driver should be HighProducts (+0.80)
    assert "HighProducts" in drivers[0]
    assert "increases churn risk" in drivers[0]


def test_evaluate_subgroup_metrics_and_disparities() -> None:
    """Verify demographic fairness evaluation and disparity ratio calculations."""
    df = pd.DataFrame(
        {
            "Gender": ["Female", "Female", "Male", "Male"],
            "Exited": [1, 0, 1, 0],
            "probability": [0.8, 0.4, 0.6, 0.1],
            "predicted_label": [1, 0, 1, 0],
        }
    )

    fair_df = evaluate_subgroup_metrics(
        df, group_col="Gender", target_col="Exited", prob_col="probability", pred_col="predicted_label"
    )
    assert len(fair_df) == 2
    assert "selection_rate" in fair_df.columns
    assert "recall" in fair_df.columns

    disparities = compute_fairness_disparities(fair_df)
    assert "selection_rate_ratio" in disparities
    assert "recall_gap" in disparities
    assert disparities["selection_rate_ratio"] == 1.0


def test_plot_calibration_curve_file_save(tmp_path: Path) -> None:
    """Verify calibration curve generates and writes image file."""
    y_true = np.array([0, 1, 0, 1, 1, 0])
    y_prob = np.array([0.1, 0.9, 0.2, 0.8, 0.7, 0.3])
    save_file = tmp_path / "test_calibration.png"

    plot_calibration_curve(y_true, y_prob, n_bins=5, save_path=save_file)
    assert save_file.exists()
    assert save_file.stat().st_size > 0

"""Unit tests for model evaluation, cross-validation, and training functions."""

import numpy as np
import pandas as pd
import pytest

from churn.evaluate import (
    calculate_metrics,
    evaluate_cv,
    get_confusion_matrix,
    get_oof_predictions,
)
from churn.features import build_pipeline
from churn.train import calibrate_model, compare_baseline_models, run_feature_ablation


@pytest.fixture
def synthetic_classification_data() -> tuple[pd.DataFrame, pd.Series]:
    """Create a synthetic reproducible dataset for quick test verification."""
    np.random.seed(42)
    n = 60
    df = pd.DataFrame(
        {
            "CreditScore": np.random.randint(400, 850, size=n),
            "Geography": np.random.choice(["France", "Germany", "Spain"], size=n),
            "Gender": np.random.choice(["Female", "Male"], size=n),
            "Age": np.random.randint(20, 70, size=n),
            "Tenure": np.random.randint(0, 10, size=n),
            "Balance": np.random.uniform(0, 100000, size=n),
            "NumOfProducts": np.random.choice([1, 2, 3, 4], size=n),
            "HasCrCard": np.random.choice([0, 1], size=n),
            "IsActiveMember": np.random.choice([0, 1], size=n),
            "EstimatedSalary": np.random.uniform(20000, 150000, size=n),
        }
    )
    # Target correlated with age and Germany
    prob = 0.1 + (df["Age"] > 45) * 0.4 + (df["Geography"] == "Germany") * 0.3
    y = pd.Series((np.random.uniform(0, 1, size=n) < prob).astype(int), name="Exited")
    return df, y


def test_calculate_metrics_values() -> None:
    """Verify calculate_metrics returns complete dictionary within valid metric bounds."""
    y_true = np.array([0, 1, 0, 1, 0, 1])
    y_prob = np.array([0.1, 0.9, 0.2, 0.8, 0.3, 0.7])

    metrics = calculate_metrics(y_true, y_prob, threshold=0.5)

    expected_keys = {"roc_auc", "pr_auc", "recall", "precision", "f1", "brier_score"}
    assert set(metrics.keys()) == expected_keys

    assert metrics["roc_auc"] == 1.0
    assert metrics["pr_auc"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["brier_score"] < 0.1


def test_confusion_matrix_shape() -> None:
    """Verify confusion matrix returns 2x2 integer array."""
    y_true = np.array([0, 1, 0, 1])
    y_prob = np.array([0.2, 0.8, 0.1, 0.3])
    cm = get_confusion_matrix(y_true, y_prob, threshold=0.5)
    assert cm.shape == (2, 2)
    assert cm.sum() == 4


def test_evaluate_cv_and_oof_predictions(
    synthetic_classification_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    """Verify evaluate_cv runs StratifiedKFold and computes mean/std across folds."""
    X, y = synthetic_classification_data
    pipe = build_pipeline("dummy")

    mean_m, std_m, fold_df = evaluate_cv(pipe, X, y, n_splits=3, random_state=42)

    assert "pr_auc" in mean_m
    assert "roc_auc" in mean_m
    assert len(fold_df) == 3

    oof_probs, y_out = get_oof_predictions(pipe, X, y, n_splits=3, random_state=42)
    assert len(oof_probs) == len(X)
    assert (oof_probs >= 0.0).all() and (oof_probs <= 1.0).all()
    assert (y_out == y.values).all()


def test_compare_baseline_models_and_ablation(
    synthetic_classification_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    """Verify baseline comparison returns all 5 models and ablation computes metric deltas."""
    X, y = synthetic_classification_data

    # Test baseline comparison runs
    df_cmp = compare_baseline_models(X, y, cv=2, random_state=42)
    assert len(df_cmp) == 5
    assert set(df_cmp["model"].tolist()) == {
        "dummy",
        "logistic_regression",
        "random_forest",
        "xgboost",
        "lightgbm",
    }

    # Test ablation
    ablation = run_feature_ablation(X, y, model_name="dummy", cv=2, random_state=42)
    assert "delta" in ablation
    assert "pr_auc" in ablation["delta"]


def test_calibrate_model_wrapper(
    synthetic_classification_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    """Verify calibrate_model wraps pipeline with CalibratedClassifierCV."""
    X, y = synthetic_classification_data
    pipe = build_pipeline("logistic_regression", random_state=42)

    cal_model, uncal_m, cal_m = calibrate_model(
        pipe, X, y, cv=2, method="sigmoid", random_state=42
    )
    assert hasattr(cal_model, "fit")
    assert "brier_score" in uncal_m
    assert "brier_score" in cal_m

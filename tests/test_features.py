"""Unit tests for feature engineering, preprocessors, and pipeline builders."""

import numpy as np
import pandas as pd
import pytest

from churn.features import (
    ALL_ENGINEERED_FEATURES,
    FeatureEngineer,
    build_linear_preprocessor,
    build_pipeline,
)


@pytest.fixture
def handmade_3row_df() -> pd.DataFrame:
    """Fixture providing a hand-crafted 3-row DataFrame for exact arithmetic verification."""
    return pd.DataFrame(
        {
            "CreditScore": [600, 700, 800],
            "Geography": ["France", "Germany", "Spain"],
            "Gender": ["Female", "Male", "Female"],
            "Age": [30, 50, 60],
            "Tenure": [2, 0, 5],
            "Balance": [0.0, 100000.0, 50000.0],
            "NumOfProducts": [1, 3, 4],
            "HasCrCard": [1, 1, 0],
            "IsActiveMember": [0, 1, 1],
            "EstimatedSalary": [99999.0, 49999.0, 0.0],
        }
    )


def test_feature_values_correct_on_handmade_df(handmade_3row_df: pd.DataFrame) -> None:
    """Verify exact formula values for all 8 engineered features on hand-crafted rows."""
    fe = FeatureEngineer(include_features=True)
    out_df = fe.transform(handmade_3row_df)

    # Row 0: Balance=0, Salary=99999, Tenure=2, Age=30, Products=1, HasCrCard=1, IsActive=0, CreditScore=600
    assert out_df.loc[0, "BalanceSalaryRatio"] == pytest.approx(0.0 / (99999.0 + 1.0))
    assert out_df.loc[0, "ZeroBalance"] == 1
    assert out_df.loc[0, "ProductsPerTenure"] == pytest.approx(1 / (2 + 1.0))
    assert out_df.loc[0, "TenureByAge"] == pytest.approx(2 / 30.0)
    assert out_df.loc[0, "CreditScoreByAge"] == pytest.approx(600 / 30.0)
    assert out_df.loc[0, "ActiveCardHolder"] == 0  # 1 * 0
    assert out_df.loc[0, "HighProducts"] == 0  # 1 < 3
    assert out_df.loc[0, "IsSenior"] == 0  # 30 < 50

    # Row 1: Balance=100000, Salary=49999, Tenure=0, Age=50, Products=3, HasCrCard=1, IsActive=1, CreditScore=700
    assert out_df.loc[1, "BalanceSalaryRatio"] == pytest.approx(100000.0 / (49999.0 + 1.0))  # 2.0
    assert out_df.loc[1, "ZeroBalance"] == 0
    assert out_df.loc[1, "ProductsPerTenure"] == pytest.approx(3 / (0 + 1.0))  # 3.0
    assert out_df.loc[1, "TenureByAge"] == pytest.approx(0 / 50.0)  # 0.0
    assert out_df.loc[1, "CreditScoreByAge"] == pytest.approx(700 / 50.0)  # 14.0
    assert out_df.loc[1, "ActiveCardHolder"] == 1  # 1 * 1
    assert out_df.loc[1, "HighProducts"] == 1  # 3 >= 3
    assert out_df.loc[1, "IsSenior"] == 1  # 50 >= 50

    # Row 2: Balance=50000, Salary=0, Tenure=5, Age=60, Products=4, HasCrCard=0, IsActive=1, CreditScore=800
    assert out_df.loc[2, "BalanceSalaryRatio"] == pytest.approx(50000.0 / (0.0 + 1.0))
    assert out_df.loc[2, "ZeroBalance"] == 0
    assert out_df.loc[2, "ProductsPerTenure"] == pytest.approx(4 / (5 + 1.0))
    assert out_df.loc[2, "TenureByAge"] == pytest.approx(5 / 60.0)
    assert out_df.loc[2, "CreditScoreByAge"] == pytest.approx(800 / 60.0)
    assert out_df.loc[2, "ActiveCardHolder"] == 0  # 0 * 1
    assert out_df.loc[2, "HighProducts"] == 1  # 4 >= 3
    assert out_df.loc[2, "IsSenior"] == 1  # 60 >= 50


def test_no_nan_or_inf_produced(handmade_3row_df: pd.DataFrame) -> None:
    """Verify that transformation handles zeros and edge cases without emitting NaN or inf."""
    fe = FeatureEngineer()
    out = fe.transform(handmade_3row_df)

    for col in ALL_ENGINEERED_FEATURES:
        assert not out[col].isna().any(), f"NaN found in {col}"
        assert not np.isinf(out[col]).any(), f"Inf found in {col}"


def test_output_column_order_is_stable(handmade_3row_df: pd.DataFrame) -> None:
    """Verify column names and order are identical across repeated transforms."""
    fe = FeatureEngineer()
    out1 = fe.transform(handmade_3row_df)
    out2 = fe.transform(handmade_3row_df)
    assert list(out1.columns) == list(out2.columns)


def test_fitting_on_train_does_not_touch_test(handmade_3row_df: pd.DataFrame) -> None:
    """Verify preprocessor fit on training data does not mutate test data or leak statistics."""
    train_slice = handmade_3row_df.iloc[:2].copy()
    test_slice = handmade_3row_df.iloc[2:].copy()
    test_slice_orig = test_slice.copy()

    preprocessor = build_linear_preprocessor()
    fe = FeatureEngineer()

    X_train_eng = fe.transform(train_slice)
    X_test_eng = fe.transform(test_slice)

    preprocessor.fit(X_train_eng)
    transformed_test = preprocessor.transform(X_test_eng)

    # Verify original test input remained unaltered
    pd.testing.assert_frame_equal(test_slice, test_slice_orig)
    assert transformed_test.shape[0] == 1


@pytest.mark.parametrize(
    "model_name",
    ["dummy", "logistic_regression", "random_forest", "xgboost", "lightgbm"],
)
def test_build_pipeline_for_all_supported_models(
    model_name: str, handmade_3row_df: pd.DataFrame
) -> None:
    """Verify build_pipeline returns a working Pipeline for all supported models."""
    pipe = build_pipeline(model_name)
    assert hasattr(pipe, "fit")
    assert hasattr(pipe, "predict")

    # Smoke fit/predict test
    X = handmade_3row_df.copy()
    y = np.array([0, 1, 0])

    pipe.fit(X, y)
    preds = pipe.predict(X)
    probs = pipe.predict_proba(X)

    assert len(preds) == 3
    assert probs.shape == (3, 2)
    assert (probs >= 0.0).all() and (probs <= 1.0).all()


def test_build_pipeline_invalid_model() -> None:
    """Verify build_pipeline raises ValueError for unsupported model key."""
    with pytest.raises(ValueError, match="Unsupported model_name"):
        build_pipeline("unsupported_neural_net")

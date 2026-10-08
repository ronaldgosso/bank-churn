"""Tests for data ingestion, schema validation, and stratified splitting."""

import json
from pathlib import Path

import pandas as pd
import pandera.pandas as pa
import pytest

from churn.config import RAW_DATA_PATH, SEED, TARGET
from churn.data import load_raw, make_split
from churn.validation import (
    sanitize_portfolio_data,
    validate_inference_data,
    validate_processed_data,
    validate_raw_data,
)


@pytest.fixture
def sample_valid_raw_df() -> pd.DataFrame:
    """Fixture providing a sample valid raw dataframe."""
    return pd.DataFrame(
        {
            "RowNumber": [1, 2],
            "CustomerId": [15634602, 15647311],
            "Surname": ["Hargrave", "Hill"],
            "CreditScore": [619, 608],
            "Geography": ["France", "Spain"],
            "Gender": ["Female", "Female"],
            "Age": [42, 41],
            "Tenure": [2, 1],
            "Balance": [0.0, 83807.86],
            "NumOfProducts": [1, 1],
            "HasCrCard": [1, 0],
            "IsActiveMember": [1, 1],
            "EstimatedSalary": [101348.88, 112542.58],
            "Exited": [1, 0],
        }
    )


def test_schema_accepts_valid_raw_data(sample_valid_raw_df: pd.DataFrame) -> None:
    """Verify raw schema accepts well-formed data."""
    validated = validate_raw_data(sample_valid_raw_df)
    assert len(validated) == 2


@pytest.mark.parametrize(
    ("corrupt_col", "corrupt_val"),
    [
        ("CreditScore", 250),  # < 300
        ("CreditScore", 950),  # > 900
        ("Geography", "USA"),  # Not in France/Germany/Spain
        ("Gender", "Other"),  # Not in Female/Male
        ("Age", 16),  # < 18
        ("Age", 105),  # > 100
        ("Tenure", 15),  # > 10
        ("NumOfProducts", 5),  # > 4
        ("HasCrCard", 2),  # Not in [0, 1]
        ("IsActiveMember", -1),  # Not in [0, 1]
        ("Exited", 2),  # Target not in [0, 1]
        ("Balance", -10.0),  # Negative balance
    ],
)
def test_schema_rejects_corrupted_row(
    sample_valid_raw_df: pd.DataFrame, corrupt_col: str, corrupt_val: object
) -> None:
    """Verify raw schema raises SchemaError on invalid values."""
    corrupted_df = sample_valid_raw_df.copy()
    corrupted_df.loc[0, corrupt_col] = corrupt_val
    with pytest.raises((pa.errors.SchemaError, pa.errors.SchemaErrors)):
        validate_raw_data(corrupted_df)


def test_schema_rejects_nulls(sample_valid_raw_df: pd.DataFrame) -> None:
    """Verify schema raises SchemaError on missing values."""
    corrupted_df = sample_valid_raw_df.copy()
    corrupted_df.loc[0, "CreditScore"] = None
    with pytest.raises((pa.errors.SchemaError, pa.errors.SchemaErrors)):
        validate_raw_data(corrupted_df)


def test_processed_and_inference_schemas() -> None:
    """Verify processed and inference schemas validate properly."""
    feature_data = {
        "CreditScore": [650],
        "Geography": ["Germany"],
        "Gender": ["Male"],
        "Age": [35],
        "Tenure": [5],
        "Balance": [50000.0],
        "NumOfProducts": [2],
        "HasCrCard": [1],
        "IsActiveMember": [1],
        "EstimatedSalary": [75000.0],
    }
    inf_df = pd.DataFrame(feature_data)
    assert len(validate_inference_data(inf_df)) == 1

    proc_df = inf_df.copy()
    proc_df[TARGET] = [0]
    assert len(validate_processed_data(proc_df)) == 1


def test_make_split_execution_and_stratification(tmp_path: Path) -> None:
    """Verify make_split produces stratified, non-overlapping train and test splits."""
    if not RAW_DATA_PATH.exists():
        pytest.skip(f"Raw data file {RAW_DATA_PATH} not found.")

    train_path = tmp_path / "train.csv"
    test_path = tmp_path / "test.csv"
    summary_path = tmp_path / "data_summary.json"

    train_df, test_df = make_split(
        raw_path=RAW_DATA_PATH,
        train_path=train_path,
        test_path=test_path,
        summary_path=summary_path,
        test_size=0.20,
        random_state=SEED,
    )

    # Check sizes: 80% train (8,000) and 20% test (2,000)
    assert len(train_df) == 8000
    assert len(test_df) == 2000

    # Identifier columns must be dropped
    for col in ["RowNumber", "CustomerId", "Surname"]:
        assert col not in train_df.columns
        assert col not in test_df.columns

    # Verify stratification (churn rate difference < 0.5%)
    raw_churn_rate = load_raw()[TARGET].mean()
    train_churn_rate = train_df[TARGET].mean()
    test_churn_rate = test_df[TARGET].mean()

    assert abs(train_churn_rate - raw_churn_rate) < 0.005
    assert abs(test_churn_rate - raw_churn_rate) < 0.005

    # Verify files created on disk
    assert train_path.exists()
    assert test_path.exists()
    assert summary_path.exists()

    # Verify summary JSON content
    with open(summary_path, encoding="utf-8") as f:
        summary = json.load(f)

    assert summary["raw_total_rows"] == 10000
    assert summary["splits"]["train"]["rows"] == 8000
    assert summary["splits"]["test"]["rows"] == 2000
    assert "churn_rate" in summary["splits"]["train"]
    assert "duplicate_count" in summary


def test_train_test_no_overlap(tmp_path: Path) -> None:
    """Verify train and test splits contain no identical combined records."""
    if not RAW_DATA_PATH.exists():
        pytest.skip(f"Raw data file {RAW_DATA_PATH} not found.")

    train_path = tmp_path / "train.csv"
    test_path = tmp_path / "test.csv"
    summary_path = tmp_path / "data_summary.json"

    train_df, test_df = make_split(
        raw_path=RAW_DATA_PATH,
        train_path=train_path,
        test_path=test_path,
        summary_path=summary_path,
        test_size=0.20,
        random_state=SEED,
    )

    # Merge train and test on all columns to ensure intersection is empty
    merged = pd.merge(train_df, test_df, how="inner")
    assert len(merged) == 0


def test_sanitize_portfolio_valid_data(sample_valid_raw_df: pd.DataFrame) -> None:
    """Verify sanitization succeeds on well-formed raw portfolio data."""
    clean_df, report = sanitize_portfolio_data(sample_valid_raw_df)
    assert report["is_valid"] is True
    assert report["total_records"] == 2
    assert len(report["core_features_found"]) == 10
    assert "CustomerId" in report["identifier_columns"]
    assert "Surname" in report["identifier_columns"]
    assert clean_df["CreditScore"].dtype == "int64" or clean_df["CreditScore"].dtype == "int32"


def test_sanitize_portfolio_aliases_and_casing() -> None:
    """Verify column alias resolution and casing normalization."""
    messy_df = pd.DataFrame(
        {
            "credit_score": [720],
            "country": ["  germany "],
            "sex": ["female"],
            "age": [45],
            "tenure": [3],
            "balance": [50000.0],
            "num_of_products": [2],
            "has_credit_card": ["yes"],
            "is_active_member": ["true"],
            "salary": [80000.0],
            "customer_id": [99999],
        }
    )
    clean_df, report = sanitize_portfolio_data(messy_df)
    assert report["is_valid"] is True
    assert "CreditScore" in clean_df.columns
    assert clean_df["Geography"].iloc[0] == "Germany"
    assert clean_df["Gender"].iloc[0] == "Female"
    assert clean_df["HasCrCard"].iloc[0] == 1
    assert clean_df["IsActiveMember"].iloc[0] == 1
    assert "CustomerId" in clean_df.columns


def test_sanitize_portfolio_missing_required_features() -> None:
    """Verify incomplete dataset reports missing features without crashing."""
    incomplete_df = pd.DataFrame(
        {
            "CreditScore": [650],
            "Age": [40],
        }
    )
    _clean_df, report = sanitize_portfolio_data(incomplete_df)
    assert report["is_valid"] is False
    assert len(report["missing_required"]) > 0
    assert "Geography" in report["missing_required"]
    assert "Balance" in report["missing_required"]


def test_sanitize_portfolio_imputes_and_clamps() -> None:
    """Verify null imputation and boundary clamping."""
    outlier_df = pd.DataFrame(
        {
            "CreditScore": [1000],  # Clamped to 850
            "Geography": ["UnknownLand"],  # Mapped to France
            "Gender": ["Alien"],  # Mapped to Female
            "Age": [None],  # Imputed with 38
            "Tenure": [-5],  # Clamped to 0
            "Balance": [-100.0],  # Clamped to 0.0
            "NumOfProducts": [10],  # Clamped to 4
            "HasCrCard": [None],  # Imputed
            "IsActiveMember": [None],  # Imputed
            "EstimatedSalary": [None],  # Imputed
        }
    )
    clean_df, report = sanitize_portfolio_data(outlier_df)
    assert report["is_valid"] is True
    assert clean_df["CreditScore"].iloc[0] == 850
    assert clean_df["Geography"].iloc[0] == "France"
    assert clean_df["Gender"].iloc[0] == "Female"
    assert clean_df["Age"].iloc[0] == 38
    assert clean_df["Tenure"].iloc[0] == 0
    assert clean_df["Balance"].iloc[0] == 0.0
    assert clean_df["NumOfProducts"].iloc[0] == 4
    assert report["nulls_imputed"] > 0
    assert report["anomalies_coerced"] > 0

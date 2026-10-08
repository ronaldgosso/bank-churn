"""Data acquisition, ingestion, and stratified train/test split utilities."""

import json
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

from churn.config import (
    DATA_SUMMARY_PATH,
    DROP_COLS,
    RAW_DATA_PATH,
    SEED,
    TARGET,
    TEST_DATA_PATH,
    TRAIN_DATA_PATH,
)
from churn.validation import validate_processed_data, validate_raw_data


def load_raw(file_path: Path | str = RAW_DATA_PATH) -> pd.DataFrame:
    """Load raw bank customer churn dataset and validate schema.

    Parameters
    ----------
    file_path : Path | str, optional
        Path to raw CSV file, by default RAW_DATA_PATH.

    Returns
    -------
    pd.DataFrame
        Validated raw dataframe.

    Raises
    ------
    FileNotFoundError
        If raw CSV does not exist.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Raw dataset not found at {path}. Please place Churn_Modelling.csv in data/raw/."
        )

    df = pd.read_csv(path)
    return validate_raw_data(df)


def make_split(
    raw_path: Path | str = RAW_DATA_PATH,
    train_path: Path | str = TRAIN_DATA_PATH,
    test_path: Path | str = TEST_DATA_PATH,
    summary_path: Path | str = DATA_SUMMARY_PATH,
    test_size: float = 0.20,
    random_state: int = SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Perform a leak-free, stratified 80/20 train/test split.

    Drops identifier columns, validates data schemas, persists processed CSVs,
    and writes dataset summary metrics to `reports/data_summary.json`.

    Parameters
    ----------
    raw_path : Path | str, optional
        Path to raw CSV file.
    train_path : Path | str, optional
        Path to destination train CSV.
    test_path : Path | str, optional
        Path to destination test CSV.
    summary_path : Path | str, optional
        Path to output data summary JSON.
    test_size : float, optional
        Fraction of data reserved for holdout test set, default 0.20.
    random_state : int, optional
        Random seed for reproducibility, default SEED (42).

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        Tuple of (train_df, test_df).
    """
    df = load_raw(raw_path)

    # Calculate duplicate count across raw data
    duplicate_rows = int(df.duplicated().sum())

    # Drop identifier columns not used in modeling
    cols_to_drop = [c for c in DROP_COLS if c in df.columns]
    modeling_df = df.drop(columns=cols_to_drop)

    # Perform stratified split
    train_df, test_df = train_test_split(
        modeling_df,
        test_size=test_size,
        random_state=random_state,
        stratify=modeling_df[TARGET],
    )

    # Validate processed schemas
    train_df = validate_processed_data(train_df.reset_index(drop=True))
    test_df = validate_processed_data(test_df.reset_index(drop=True))

    # Ensure output directories exist
    train_dest = Path(train_path)
    test_dest = Path(test_path)
    summary_dest = Path(summary_path)

    train_dest.parent.mkdir(parents=True, exist_ok=True)
    summary_dest.parent.mkdir(parents=True, exist_ok=True)

    # Persist split datasets
    train_df.to_csv(train_dest, index=False)
    test_df.to_csv(test_dest, index=False)

    # Generate data summary report
    summary: dict[str, Any] = {
        "raw_total_rows": len(df),
        "raw_total_columns": df.shape[1],
        "duplicate_count": duplicate_rows,
        "splits": {
            "train": {
                "rows": len(train_df),
                "columns": train_df.shape[1],
                "churn_rate": float(train_df[TARGET].mean()),
                "churn_count": int(train_df[TARGET].sum()),
                "non_churn_count": int((train_df[TARGET] == 0).sum()),
            },
            "test": {
                "rows": len(test_df),
                "columns": test_df.shape[1],
                "churn_rate": float(test_df[TARGET].mean()),
                "churn_count": int(test_df[TARGET].sum()),
                "non_churn_count": int((test_df[TARGET] == 0).sum()),
            },
        },
        "random_state": random_state,
        "test_size": test_size,
    }

    with open(summary_dest, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return train_df, test_df


def load_train(train_path: Path | str = TRAIN_DATA_PATH) -> pd.DataFrame:
    """Load and validate training split.

    Parameters
    ----------
    train_path : Path | str, optional
        Path to processed train CSV.

    Returns
    -------
    pd.DataFrame
        Training dataframe.
    """
    path = Path(train_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Train dataset not found at {path}. Run make_split() first to generate it."
        )
    return validate_processed_data(pd.read_csv(path))


def load_test(test_path: Path | str = TEST_DATA_PATH) -> pd.DataFrame:
    """Load and validate holdout test split.

    NOTE: Per project specification, the test set must be touched exactly ONCE
    in Phase 6 (Final Evaluation). Do not call during EDA, feature engineering,
    or model tuning.

    Parameters
    ----------
    test_path : Path | str, optional
        Path to processed test CSV.

    Returns
    -------
    pd.DataFrame
        Holdout test dataframe.
    """
    path = Path(test_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Test dataset not found at {path}. Run make_split() first to generate it."
        )
    return validate_processed_data(pd.read_csv(path))

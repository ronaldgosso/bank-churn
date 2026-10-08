"""Pandera schema validation for bank churn datasets."""

import pandas as pd
from pandera.pandas import Check, Column, DataFrameSchema

from churn.config import (
    DROP_COLS,
    TARGET,
    VALID_GENDERS,
    VALID_GEOGRAPHIES,
)

# Base schema for core customer features
FEATURE_COLUMNS: dict[str, Column] = {
    "CreditScore": Column(int, Check.in_range(300, 900), nullable=False, coerce=True),
    "Geography": Column(str, Check.isin(VALID_GEOGRAPHIES), nullable=False),
    "Gender": Column(str, Check.isin(VALID_GENDERS), nullable=False),
    "Age": Column(int, Check.in_range(18, 100), nullable=False, coerce=True),
    "Tenure": Column(int, Check.in_range(0, 10), nullable=False, coerce=True),
    "Balance": Column(float, Check.greater_than_or_equal_to(0.0), nullable=False, coerce=True),
    "NumOfProducts": Column(int, Check.in_range(1, 4), nullable=False, coerce=True),
    "HasCrCard": Column(int, Check.isin([0, 1]), nullable=False, coerce=True),
    "IsActiveMember": Column(int, Check.isin([0, 1]), nullable=False, coerce=True),
    "EstimatedSalary": Column(
        float, Check.greater_than_or_equal_to(0.0), nullable=False, coerce=True
    ),
}

# Target column schema
TARGET_COLUMN: dict[str, Column] = {
    TARGET: Column(int, Check.isin([0, 1]), nullable=False, coerce=True),
}

# Raw identifier columns schema
RAW_IDENTIFIER_COLUMNS: dict[str, Column] = {
    DROP_COLS[0]: Column(int, Check.greater_than(0), nullable=False, coerce=True),  # RowNumber
    DROP_COLS[1]: Column(int, Check.greater_than(0), nullable=False, coerce=True),  # CustomerId
    DROP_COLS[2]: Column(str, nullable=False),  # Surname
}

# Full raw schema
raw_schema: DataFrameSchema = DataFrameSchema(
    columns={**RAW_IDENTIFIER_COLUMNS, **FEATURE_COLUMNS, **TARGET_COLUMN},
    strict=False,
    coerce=True,
)

# Processed train/test schema (without raw identifiers)
processed_schema: DataFrameSchema = DataFrameSchema(
    columns={**FEATURE_COLUMNS, **TARGET_COLUMN},
    strict=True,
    coerce=True,
)

# Inference feature schema (features only, without target or identifiers)
inference_schema: DataFrameSchema = DataFrameSchema(
    columns=FEATURE_COLUMNS,
    strict=True,
    coerce=True,
)


def validate_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    """Validate raw customer data against the raw schema.

    Parameters
    ----------
    df : pd.DataFrame
        Raw input dataframe.

    Returns
    -------
    pd.DataFrame
        Validated dataframe.

    Raises
    ------
    pa.errors.SchemaError
        If data fails schema checks.
    """
    return raw_schema.validate(df)


def validate_processed_data(df: pd.DataFrame) -> pd.DataFrame:
    """Validate processed train/test data against the processed schema.

    Parameters
    ----------
    df : pd.DataFrame
        Processed dataframe containing features and target.

    Returns
    -------
    pd.DataFrame
        Validated dataframe.
    """
    return processed_schema.validate(df)


def validate_inference_data(df: pd.DataFrame) -> pd.DataFrame:
    """Validate feature dataframe for inference against the inference schema.

    Parameters
    ----------
    df : pd.DataFrame
        Inference feature dataframe.

    Returns
    -------
    pd.DataFrame
        Validated dataframe.
    """
    return inference_schema.validate(df)

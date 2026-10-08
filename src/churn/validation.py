from typing import Any

import pandas as pd
from pandera.pandas import Check, Column, DataFrameSchema

from churn.config import (
    ALL_FEATURE_COLS,
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


# Canonical column alias dictionary mapping common variations to standard schema
COLUMN_ALIASES: dict[str, str] = {
    "creditscore": "CreditScore",
    "credit_score": "CreditScore",
    "geography": "Geography",
    "country": "Geography",
    "gender": "Gender",
    "sex": "Gender",
    "age": "Age",
    "tenure": "Tenure",
    "balance": "Balance",
    "numofproducts": "NumOfProducts",
    "num_of_products": "NumOfProducts",
    "hascrcard": "HasCrCard",
    "has_cr_card": "HasCrCard",
    "hascreditcard": "HasCrCard",
    "has_credit_card": "HasCrCard",
    "isactivemember": "IsActiveMember",
    "is_active_member": "IsActiveMember",
    "estimatedsalary": "EstimatedSalary",
    "estimated_salary": "EstimatedSalary",
    "salary": "EstimatedSalary",
    "customerid": "CustomerId",
    "customer_id": "CustomerId",
    "surname": "Surname",
    "last_name": "Surname",
    "name": "Surname",
    "rownumber": "RowNumber",
    "row_number": "RowNumber",
    "exited": "Exited",
    "churn": "Exited",
}


def sanitize_portfolio_data(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Sanitize and audit an uploaded customer portfolio DataFrame before scoring.

    Normalizes column aliases, coerces data types, handles missing values,
    clamps anomalous extremes, and preserves non-feature identifier columns.

    Parameters
    ----------
    df : pd.DataFrame
        Raw input customer portfolio DataFrame.

    Returns
    -------
    tuple[pd.DataFrame, dict[str, Any]]
        Tuple containing:
        - clean_df : pd.DataFrame with sanitized features and preserved identifiers.
        - report : Audit summary dictionary detailing checks, renames, and coercions.
    """
    clean_df = df.copy()
    audit_messages: list[str] = []
    renamed: dict[str, str] = {}
    nulls_imputed = 0
    anomalies_coerced = 0

    # 1. Column Name Normalization & Alias Resolution
    for col in list(clean_df.columns):
        normalized_name = str(col).strip().lower().replace(" ", "_")
        if normalized_name in COLUMN_ALIASES and COLUMN_ALIASES[normalized_name] != col:
            target_col = COLUMN_ALIASES[normalized_name]
            clean_df.rename(columns={col: target_col}, inplace=True)
            renamed[str(col)] = target_col

    if renamed:
        audit_messages.append(
            f"Mapped {len(renamed)} column(s) to canonical schema: {', '.join(f'`{k}` -> `{v}`' for k, v in renamed.items())}."
        )

    # 2. Check Core Modeling Feature Presence
    missing_required = [c for c in ALL_FEATURE_COLS if c not in clean_df.columns]
    found_core = [c for c in ALL_FEATURE_COLS if c in clean_df.columns]
    preserved_identifiers = [c for c in DROP_COLS if c in clean_df.columns]
    extra_cols = [
        c
        for c in clean_df.columns
        if c not in ALL_FEATURE_COLS and c not in DROP_COLS and c != TARGET
    ]

    if missing_required:
        return clean_df, {
            "is_valid": False,
            "total_records": len(clean_df),
            "core_features_found": found_core,
            "missing_required": missing_required,
            "renamed_columns": renamed,
            "identifier_columns": preserved_identifiers,
            "extra_columns": extra_cols,
            "nulls_imputed": 0,
            "anomalies_coerced": 0,
            "audit_messages": [
                f"Missing {len(missing_required)} required feature(s): {', '.join(missing_required)}."
            ],
        }

    # 3. Clean Categorical Fields
    # Geography
    clean_df["Geography"] = clean_df["Geography"].astype(str).str.strip().str.title()
    invalid_geo_mask = ~clean_df["Geography"].isin(VALID_GEOGRAPHIES)
    if invalid_geo_mask.any():
        bad_count = int(invalid_geo_mask.sum())
        anomalies_coerced += bad_count
        clean_df.loc[invalid_geo_mask, "Geography"] = "France"
        audit_messages.append(
            f"Normalized {bad_count} record(s) with invalid Geography to modal default ('France')."
        )

    # Gender
    clean_df["Gender"] = clean_df["Gender"].astype(str).str.strip().str.title()
    gender_map = {"M": "Male", "F": "Female"}
    clean_df["Gender"] = clean_df["Gender"].replace(gender_map)
    invalid_gender_mask = ~clean_df["Gender"].isin(VALID_GENDERS)
    if invalid_gender_mask.any():
        bad_count = int(invalid_gender_mask.sum())
        anomalies_coerced += bad_count
        clean_df.loc[invalid_gender_mask, "Gender"] = "Female"
        audit_messages.append(
            f"Normalized {bad_count} record(s) with unrecognized Gender to 'Female'."
        )

    # 4. Numeric & Binary Field Sanitization
    num_specs: dict[str, tuple[float, float, float]] = {
        "CreditScore": (300.0, 850.0, 650.0),
        "Age": (18.0, 100.0, 38.0),
        "Tenure": (0.0, 10.0, 5.0),
        "Balance": (0.0, 1_000_000.0, 0.0),
        "NumOfProducts": (1.0, 4.0, 1.0),
        "EstimatedSalary": (0.0, 2_000_000.0, 100_000.0),
    }

    for col, (min_v, max_v, default_val) in num_specs.items():
        clean_df[col] = pd.to_numeric(clean_df[col], errors="coerce")
        null_count = int(clean_df[col].isna().sum())
        if null_count > 0:
            nulls_imputed += null_count
            clean_df[col] = clean_df[col].fillna(default_val)

        # Clamping
        low_mask = clean_df[col] < min_v
        high_mask = clean_df[col] > max_v
        if low_mask.any() or high_mask.any():
            clamp_count = int(low_mask.sum() + high_mask.sum())
            anomalies_coerced += clamp_count
            clean_df[col] = clean_df[col].clip(lower=min_v, upper=max_v)

        if col in ["CreditScore", "Age", "Tenure", "NumOfProducts"]:
            clean_df[col] = clean_df[col].round().astype(int)
        else:
            clean_df[col] = clean_df[col].astype(float)

    # Binary flags (HasCrCard, IsActiveMember)
    bool_map: dict[str, int] = {
        "1": 1,
        "true": 1,
        "yes": 1,
        "y": 1,
        "0": 0,
        "false": 0,
        "no": 0,
        "n": 0,
    }
    for b_col, def_val in [("HasCrCard", 1), ("IsActiveMember", 0)]:
        raw_vals = clean_df[b_col]
        if pd.api.types.is_numeric_dtype(raw_vals):
            clean_df[b_col] = (
                pd.to_numeric(raw_vals, errors="coerce").fillna(def_val).clip(0, 1).astype(int)
            )
        else:
            mapped_bool = (
                raw_vals.astype(str)
                .str.strip()
                .str.lower()
                .map(bool_map)
                .fillna(def_val)
                .astype(int)
            )
            clean_df[b_col] = mapped_bool

    if nulls_imputed > 0:
        audit_messages.append(
            f"Imputed {nulls_imputed} missing cell(s) with domain-specific median defaults."
        )
    if anomalies_coerced > 0:
        audit_messages.append(
            f"Clamped {anomalies_coerced} outlier boundary value(s) within valid physical ranges."
        )
    if preserved_identifiers:
        audit_messages.append(
            f"Preserved {len(preserved_identifiers)} identifier column(s) for export: {', '.join(f'`{c}`' for c in preserved_identifiers)}."
        )

    return clean_df, {
        "is_valid": True,
        "total_records": len(clean_df),
        "core_features_found": found_core,
        "missing_required": [],
        "renamed_columns": renamed,
        "identifier_columns": preserved_identifiers,
        "extra_columns": extra_cols,
        "nulls_imputed": nulls_imputed,
        "anomalies_coerced": anomalies_coerced,
        "audit_messages": audit_messages,
    }

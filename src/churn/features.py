"""Feature engineering and preprocessing pipeline builders."""

from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from churn.config import BINARY, CATEGORICAL, NUMERIC, SEED

ENGINEERED_NUMERIC: list[str] = [
    "BalanceSalaryRatio",
    "ProductsPerTenure",
    "TenureByAge",
    "CreditScoreByAge",
]

ENGINEERED_BINARY: list[str] = [
    "ZeroBalance",
    "ActiveCardHolder",
    "HighProducts",
    "IsSenior",
]

ALL_ENGINEERED_FEATURES: list[str] = ENGINEERED_NUMERIC + ENGINEERED_BINARY


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Domain-specific feature engineer for bank churn prediction.

    Engineers 8 behavioral and financial interaction features:
    - BalanceSalaryRatio: Balance / (EstimatedSalary + 1)
    - ZeroBalance: 1 if Balance == 0 else 0
    - ProductsPerTenure: NumOfProducts / (Tenure + 1)
    - TenureByAge: Tenure / Age
    - CreditScoreByAge: CreditScore / Age
    - ActiveCardHolder: HasCrCard * IsActiveMember
    - HighProducts: 1 if NumOfProducts >= 3 else 0
    - IsSenior: 1 if Age >= 50 else 0
    """

    def __init__(self, include_features: bool = True) -> None:
        """Initialize FeatureEngineer.

        Parameters
        ----------
        include_features : bool, optional
            Whether to add engineered features. If False, returns input untouched.
        """
        self.include_features = include_features

    def fit(self, X: pd.DataFrame, y: Any = None) -> "FeatureEngineer":
        """Fit transformer (stateless).

        Parameters
        ----------
        X : pd.DataFrame
            Input features dataframe.
        y : Any, optional
            Ignored.

        Returns
        -------
        FeatureEngineer
            Self instance.
        """
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Add engineered features to input dataframe.

        Parameters
        ----------
        X : pd.DataFrame
            Input dataframe.

        Returns
        -------
        pd.DataFrame
            Dataframe with engineered features appended in deterministic column order.
        """
        if not self.include_features:
            return X.copy()

        df = X.copy()

        # Engineered features per specification
        df["BalanceSalaryRatio"] = df["Balance"] / (df["EstimatedSalary"] + 1.0)
        df["ZeroBalance"] = (df["Balance"] == 0.0).astype(int)
        df["ProductsPerTenure"] = df["NumOfProducts"] / (df["Tenure"] + 1.0)
        df["TenureByAge"] = df["Tenure"] / df["Age"]
        df["CreditScoreByAge"] = df["CreditScore"] / df["Age"]
        df["ActiveCardHolder"] = (df["HasCrCard"] * df["IsActiveMember"]).astype(int)
        df["HighProducts"] = (df["NumOfProducts"] >= 3).astype(int)
        df["IsSenior"] = (df["Age"] >= 50).astype(int)

        # Replace any potential inf or NaN with 0.0 (Windows safety & edge case stability)
        for col in ALL_ENGINEERED_FEATURES:
            df[col] = df[col].replace([np.inf, -np.inf], 0.0).fillna(0.0)

        return df


def build_linear_preprocessor(include_engineered: bool = True) -> ColumnTransformer:
    """Build ColumnTransformer for linear models (OneHot + StandardScaler).

    Scales all numeric features to zero mean / unit variance and one-hot encodes
    categorical variables. Binaries are passed through.

    Parameters
    ----------
    include_engineered : bool, optional
        Whether engineered features are included in input, default True.

    Returns
    -------
    ColumnTransformer
        Preconfigured transformer for linear pipelines.
    """
    numeric_cols = list(NUMERIC)
    binary_cols = list(BINARY)

    if include_engineered:
        numeric_cols.extend(ENGINEERED_NUMERIC)
        binary_cols.extend(ENGINEERED_BINARY)

    return ColumnTransformer(
        transformers=[
            (
                "num",
                StandardScaler(),
                numeric_cols,
            ),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False, drop="first"),
                CATEGORICAL,
            ),
            (
                "bin",
                "passthrough",
                binary_cols,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_tree_preprocessor(include_engineered: bool = True) -> ColumnTransformer:
    """Build ColumnTransformer for tree-based models (OneHot only, no scaling).

    Parameters
    ----------
    include_engineered : bool, optional
        Whether engineered features are included in input, default True.

    Returns
    -------
    ColumnTransformer
        Preconfigured transformer for tree pipelines.
    """
    numeric_cols = list(NUMERIC)
    binary_cols = list(BINARY)

    if include_engineered:
        numeric_cols.extend(ENGINEERED_NUMERIC)
        binary_cols.extend(ENGINEERED_BINARY)

    passthrough_cols = numeric_cols + binary_cols

    return ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL,
            ),
            (
                "pass",
                "passthrough",
                passthrough_cols,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_pipeline(
    model_name: str,
    include_engineered: bool = True,
    random_state: int = SEED,
    **model_kwargs: Any,
) -> Pipeline:
    """Build a complete, leak-free scikit-learn Pipeline for any supported model.

    Supported model names:
    - 'dummy': Baseline DummyClassifier
    - 'logistic_regression' / 'logreg': LogisticRegression with scaling
    - 'random_forest' / 'rf': RandomForestClassifier
    - 'xgboost' / 'xgb': XGBClassifier
    - 'lightgbm' / 'lgbm': LGBMClassifier

    Parameters
    ----------
    model_name : str
        Key identifying the model architecture.
    include_engineered : bool, optional
        Whether to include domain feature engineering, default True.
    random_state : int, optional
        Random seed, default SEED (42).
    **model_kwargs : Any
        Hyperparameters passed directly to the estimator.

    Returns
    -------
    Pipeline
        Scikit-learn Pipeline with feature engineering, preprocessing, and classifier.

    Raises
    ------
    ValueError
        If model_name is not supported.
    """
    normalized_name = model_name.lower().strip()

    if normalized_name in ["dummy", "baseline"]:
        preprocessor = build_tree_preprocessor(include_engineered=include_engineered)
        clf = DummyClassifier(strategy="most_frequent", **model_kwargs)

    elif normalized_name in ["logistic_regression", "logreg", "lr"]:
        preprocessor = build_linear_preprocessor(include_engineered=include_engineered)
        defaults: dict[str, Any] = {
            "random_state": random_state,
            "max_iter": 1000,
            "class_weight": "balanced",
        }
        defaults.update(model_kwargs)
        clf = LogisticRegression(**defaults)

    elif normalized_name in ["random_forest", "rf"]:
        preprocessor = build_tree_preprocessor(include_engineered=include_engineered)
        defaults = {
            "random_state": random_state,
            "class_weight": "balanced",
            "n_estimators": 100,
        }
        defaults.update(model_kwargs)
        clf = RandomForestClassifier(**defaults)

    elif normalized_name in ["xgboost", "xgb"]:
        preprocessor = build_tree_preprocessor(include_engineered=include_engineered)
        defaults = {
            "random_state": random_state,
            "eval_metric": "logloss",
        }
        defaults.update(model_kwargs)
        clf = XGBClassifier(**defaults)

    elif normalized_name in ["lightgbm", "lgbm"]:
        preprocessor = build_tree_preprocessor(include_engineered=include_engineered)
        defaults = {
            "random_state": random_state,
            "class_weight": "balanced",
            "verbose": -1,
        }
        defaults.update(model_kwargs)
        clf = LGBMClassifier(**defaults)

    else:
        supported = ["dummy", "logistic_regression", "random_forest", "xgboost", "lightgbm"]
        raise ValueError(
            f"Unsupported model_name '{model_name}'. Choose from: {supported}"
        )

    return Pipeline(
        steps=[
            ("feature_engineer", FeatureEngineer(include_features=include_engineered)),
            ("preprocessor", preprocessor),
            ("classifier", clf),
        ]
    )

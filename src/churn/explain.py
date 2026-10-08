"""Explainability helpers using SHAP for global feature importance and individual customer explanations."""

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.pipeline import Pipeline


def get_preprocessed_features(
    pipeline: Pipeline,
    X: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    """Transform raw feature DataFrame through pipeline preprocessors into a named DataFrame.

    Parameters
    ----------
    pipeline : Pipeline
        Fitted model pipeline containing 'feature_engineer' and 'preprocessor' steps.
    X : pd.DataFrame
        Input raw customer features.

    Returns
    -------
    tuple[pd.DataFrame, list[str]]
        Tuple of (preprocessed_feature_dataframe, feature_names_list).
    """
    fe = pipeline.named_steps["feature_engineer"]
    prep = pipeline.named_steps["preprocessor"]

    X_eng = fe.transform(X)
    X_prep = prep.transform(X_eng)
    feature_names = [str(n) for n in prep.get_feature_names_out()]

    X_prep_df = pd.DataFrame(X_prep, columns=feature_names, index=X.index)
    return X_prep_df, feature_names


def create_shap_explainer(pipeline: Pipeline) -> shap.TreeExplainer:
    """Initialize a TreeExplainer on the fitted classifier within the pipeline.

    Parameters
    ----------
    pipeline : Pipeline
        Fitted pipeline.

    Returns
    -------
    shap.TreeExplainer
        Initialized TreeExplainer.
    """
    clf = pipeline.named_steps["classifier"]
    return shap.TreeExplainer(clf)


def compute_shap_values(
    explainer: shap.TreeExplainer,
    X_preprocessed_df: pd.DataFrame,
) -> shap.Explanation:
    """Compute SHAP values explanation object for preprocessed DataFrame.

    Parameters
    ----------
    explainer : shap.TreeExplainer
        SHAP TreeExplainer.
    X_preprocessed_df : pd.DataFrame
        Preprocessed features DataFrame with named columns.

    Returns
    -------
    shap.Explanation
        SHAP explanation object with values, data, and feature names.
    """
    return explainer(X_preprocessed_df)


def get_top_drivers(
    shap_values_row: np.ndarray | shap.Explanation,
    feature_names: list[str],
    raw_feature_values: dict[str, Any] | pd.Series | None = None,
    top_n: int = 5,
) -> list[str]:
    """Extract top N churn drivers formatted as plain-English explanatory strings.

    Parameters
    ----------
    shap_values_row : np.ndarray | shap.Explanation
        SHAP values array or 1D Explanation for a single customer.
    feature_names : list[str]
        List of feature names matching the SHAP values length.
    raw_feature_values : dict[str, Any] | pd.Series | None, optional
        Original unencoded customer feature values for context, optional.
    top_n : int, optional
        Number of top drivers to return, default 5.

    Returns
    -------
    list[str]
        List of plain-English explanation strings.
    """
    if hasattr(shap_values_row, "values"):
        s_vals = np.asarray(shap_values_row.values).flatten()
    else:
        s_vals = np.asarray(shap_values_row).flatten()

    # Sort indices by absolute impact
    ranked_indices = np.argsort(np.abs(s_vals))[::-1][:top_n]

    drivers: list[str] = []
    for idx in ranked_indices:
        feat = feature_names[idx]
        val = s_vals[idx]
        direction = "increases churn risk" if val > 0 else "lowers churn risk"

        # Look up friendly raw value if available
        context_str = ""
        if raw_feature_values is not None:
            raw_val = None
            if isinstance(raw_feature_values, dict):
                raw_val = raw_feature_values.get(feat)
            elif hasattr(raw_feature_values, "get"):
                raw_val = raw_feature_values.get(feat)

            if raw_val is not None:
                context_str = f" ({raw_val})"

        clean_name = feat.replace("cat__", "").replace("pass__", "")
        driver_str = f"{clean_name}{context_str}: {direction} ({val:+.2f} log-odds)"
        drivers.append(driver_str)

    return drivers


def plot_shap_summary(
    shap_explanation: shap.Explanation,
    save_path: Path | str | None = None,
    max_display: int = 12,
) -> None:
    """Generate and save SHAP beeswarm summary plot."""
    fig = plt.figure(figsize=(9, 6))
    shap.plots.beeswarm(shap_explanation, max_display=max_display, show=False)
    plt.title("SHAP Beeswarm Summary Plot", fontsize=12, fontweight="bold", pad=15)
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_shap_bar(
    shap_explanation: shap.Explanation,
    save_path: Path | str | None = None,
    max_display: int = 12,
) -> None:
    """Generate and save SHAP global mean absolute importance bar plot."""
    fig = plt.figure(figsize=(8, 5))
    shap.plots.bar(shap_explanation, max_display=max_display, show=False)
    plt.title("Global Feature Importance (Mean |SHAP|)", fontsize=12, fontweight="bold", pad=15)
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_shap_dependence(
    shap_explanation: shap.Explanation,
    feature_name: str,
    interaction_feature: str | None = None,
    save_path: Path | str | None = None,
) -> None:
    """Generate and save SHAP scatter/dependence plot for a feature."""
    fig = plt.figure(figsize=(7, 4.5))
    shap.plots.scatter(
        shap_explanation[:, feature_name],
        color=shap_explanation[:, interaction_feature] if interaction_feature else None,
        show=False,
    )
    plt.title(f"SHAP Dependence: {feature_name}", fontsize=11, fontweight="bold", pad=12)
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_shap_waterfall(
    shap_explanation_row: shap.Explanation,
    customer_title: str = "Customer",
    save_path: Path | str | None = None,
    max_display: int = 8,
) -> None:
    """Generate and save individual customer SHAP waterfall explanation."""
    fig = plt.figure(figsize=(8, 5))
    shap.plots.waterfall(shap_explanation_row, max_display=max_display, show=False)
    plt.title(f"Decision Waterfall: {customer_title}", fontsize=11, fontweight="bold", pad=12)
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

"""Fairness assessment and group parity metrics across demographic slices."""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, roc_auc_score


def evaluate_subgroup_metrics(
    df: pd.DataFrame,
    group_col: str,
    target_col: str = "Exited",
    prob_col: str = "probability",
    pred_col: str = "predicted_label",
) -> pd.DataFrame:
    """Calculate classification performance and selection metrics across demographic slices.

    Metrics reported per subgroup:
    - Sample size (count and percentage)
    - Base churn rate
    - Subgroup ROC-AUC
    - Selection Rate (P(Pred=1))
    - Recall (True Positive Rate)
    - Precision (Positive Predictive Value)

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing demographic attributes, actual target, probabilities, and binary predictions.
    group_col : str
        Demographic slicing column (e.g. 'Gender', 'Geography').
    target_col : str, optional
        Target column name, default 'Exited'.
    prob_col : str, optional
        Predicted probability column, default 'probability'.
    pred_col : str, optional
        Discrete predicted label column, default 'predicted_label'.

    Returns
    -------
    pd.DataFrame
        Subgroup fairness metrics table.
    """
    records: list[dict[str, Any]] = []
    total_len = len(df)

    for group_val, slice_df in df.groupby(group_col, observed=False):
        n = len(slice_df)
        y_true = slice_df[target_col].to_numpy().astype(int)
        y_prob = slice_df[prob_col].to_numpy().astype(float)
        y_pred = slice_df[pred_col].to_numpy().astype(int)

        churn_count = int(y_true.sum())
        base_rate = float(churn_count / n) if n > 0 else 0.0

        # Subgroup ROC-AUC
        try:
            auc = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else np.nan
        except ValueError:
            auc = np.nan

        selection_rate = float(y_pred.mean()) if n > 0 else 0.0
        rec = float(recall_score(y_true, y_pred, zero_division=0))
        prec = float(precision_score(y_true, y_pred, zero_division=0))

        records.append({
            group_col: str(group_val),
            "sample_count": n,
            "sample_pct": round(n / total_len, 4),
            "churn_count": churn_count,
            "churn_rate": round(base_rate, 4),
            "roc_auc": round(auc, 4) if not np.isnan(auc) else None,
            "selection_rate": round(selection_rate, 4),
            "recall": round(rec, 4),
            "precision": round(prec, 4),
        })

    return pd.DataFrame(records)


def compute_fairness_disparities(
    subgroup_df: pd.DataFrame,
) -> dict[str, float]:
    """Compute selection rate ratio and maximum recall gap across subgroups.

    Parameters
    ----------
    subgroup_df : pd.DataFrame
        Subgroup metrics table from `evaluate_subgroup_metrics`.

    Returns
    -------
    dict[str, float]
        Dictionary with:
        - selection_rate_ratio: min(selection_rate) / max(selection_rate) (Four-fifths rule threshold: 0.80)
        - recall_gap: max(recall) - min(recall)
        - precision_gap: max(precision) - min(precision)
    """
    sel_rates = subgroup_df["selection_rate"].dropna()
    recalls = subgroup_df["recall"].dropna()
    precisions = subgroup_df["precision"].dropna()

    min_sel = float(sel_rates.min())
    max_sel = float(sel_rates.max())
    sel_ratio = round(min_sel / max_sel, 4) if max_sel > 0 else 1.0

    rec_gap = round(float(recalls.max() - recalls.min()), 4)
    prec_gap = round(float(precisions.max() - precisions.min()), 4)

    return {
        "selection_rate_ratio": sel_ratio,
        "recall_gap": rec_gap,
        "precision_gap": prec_gap,
    }

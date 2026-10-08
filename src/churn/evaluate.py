"""Model evaluation metrics, cross-validation scoring, and performance curve generators."""

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold

from churn.config import SEED


def calculate_metrics(
    y_true: np.ndarray | pd.Series,
    y_prob: np.ndarray | pd.Series,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Calculate classification and calibration metrics at a given decision threshold.

    Parameters
    ----------
    y_true : np.ndarray | pd.Series
        Ground truth binary labels (0 or 1).
    y_prob : np.ndarray | pd.Series
        Predicted probabilities for positive class (1).
    threshold : float, optional
        Decision threshold for class prediction, default 0.5.

    Returns
    -------
    dict[str, float]
        Dictionary containing roc_auc, pr_auc, recall, precision, f1, and brier_score.
    """
    y_true_arr = np.asarray(y_true).astype(int)
    y_prob_arr = np.asarray(y_prob).astype(float)
    y_pred_arr = (y_prob_arr >= threshold).astype(int)

    # When all predictions are identical or probabilities constant (e.g. DummyClassifier)
    try:
        roc_auc = float(roc_auc_score(y_true_arr, y_prob_arr))
    except ValueError:
        roc_auc = 0.5

    try:
        pr_auc = float(average_precision_score(y_true_arr, y_prob_arr))
    except ValueError:
        pr_auc = float(y_true_arr.mean())

    return {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "recall": float(recall_score(y_true_arr, y_pred_arr, zero_division=0)),
        "precision": float(precision_score(y_true_arr, y_pred_arr, zero_division=0)),
        "f1": float(f1_score(y_true_arr, y_pred_arr, zero_division=0)),
        "brier_score": float(brier_score_loss(y_true_arr, y_prob_arr)),
    }


def evaluate_cv(
    pipeline: Any,
    X: pd.DataFrame,
    y: pd.Series | np.ndarray,
    n_splits: int = 5,
    random_state: int = SEED,
    threshold: float = 0.5,
) -> tuple[dict[str, float], dict[str, float], pd.DataFrame]:
    """Evaluate pipeline using leak-free StratifiedKFold cross-validation.

    Computes per-fold and aggregated mean and standard deviation for:
    ROC-AUC, PR-AUC, Recall, Precision, F1 at threshold, and Brier score.

    Parameters
    ----------
    pipeline : Any
        Sklearn or imblearn Pipeline.
    X : pd.DataFrame
        Training features dataframe.
    y : pd.Series | np.ndarray
        Training labels.
    n_splits : int, optional
        Number of cross-validation folds, default 5.
    random_state : int, optional
        Random seed, default SEED (42).
    threshold : float, optional
        Threshold for discrete predictions, default 0.5.

    Returns
    -------
    tuple[dict[str, float], dict[str, float], pd.DataFrame]
        Tuple of (mean_metrics, std_metrics, fold_results_df).
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    y_arr = np.asarray(y).astype(int)

    fold_records: list[dict[str, Any]] = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_arr), start=1):
        X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
        y_train_fold, y_val_fold = y_arr[train_idx], y_arr[val_idx]

        model = clone(pipeline)
        model.fit(X_train_fold, y_train_fold)

        if hasattr(model, "predict_proba"):
            y_prob_fold = model.predict_proba(X_val_fold)[:, 1]
        else:
            y_prob_fold = model.decision_function(X_val_fold)

        metrics = calculate_metrics(y_val_fold, y_prob_fold, threshold=threshold)
        metrics["fold"] = fold
        fold_records.append(metrics)

    fold_df = pd.DataFrame(fold_records)
    metric_cols = ["roc_auc", "pr_auc", "recall", "precision", "f1", "brier_score"]

    mean_metrics = {m: float(fold_df[m].mean()) for m in metric_cols}
    std_metrics = {m: float(fold_df[m].std()) for m in metric_cols}

    return mean_metrics, std_metrics, fold_df


def get_oof_predictions(
    pipeline: Any,
    X: pd.DataFrame,
    y: pd.Series | np.ndarray,
    n_splits: int = 5,
    random_state: int = SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate out-of-fold probability predictions via StratifiedKFold.

    Parameters
    ----------
    pipeline : Any
        Sklearn or imblearn Pipeline.
    X : pd.DataFrame
        Features dataframe.
    y : pd.Series | np.ndarray
        Ground truth labels.
    n_splits : int, optional
        Number of folds, default 5.
    random_state : int, optional
        Random seed, default SEED (42).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Tuple of (oof_probabilities, y_true).
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    y_arr = np.asarray(y).astype(int)
    oof_probs = np.zeros(len(y_arr), dtype=float)

    for train_idx, val_idx in skf.split(X, y_arr):
        X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
        y_train_fold = y_arr[train_idx]

        model = clone(pipeline)
        model.fit(X_train_fold, y_train_fold)

        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_val_fold)[:, 1]
        else:
            probs = model.decision_function(X_val_fold)

        oof_probs[val_idx] = probs

    return oof_probs, y_arr


def get_roc_curve_data(
    y_true: np.ndarray, y_prob: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return false positive rates, true positive rates, and thresholds for ROC curve."""
    return roc_curve(y_true, y_prob)


def get_pr_curve_data(
    y_true: np.ndarray, y_prob: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return precision, recall, and thresholds for Precision-Recall curve."""
    return precision_recall_curve(y_true, y_prob)


def get_confusion_matrix(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5
) -> np.ndarray:
    """Return 2x2 confusion matrix [[TN, FP], [FN, TP]] at threshold."""
    y_pred = (y_prob >= threshold).astype(int)
    return confusion_matrix(y_true, y_pred)


def plot_calibration_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
    save_path: Any = None,
) -> None:
    """Plot reliability diagram (calibration curve) comparing predicted vs observed frequency.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth binary labels.
    y_prob : np.ndarray
        Predicted probabilities.
    n_bins : int, optional
        Number of probability bins, default 10.
    save_path : Any, optional
        Path to save plot.
    """
    from sklearn.calibration import calibration_curve

    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="uniform")

    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.plot(prob_pred, prob_true, marker="s", linewidth=2.5, color="#386cb0", label="Model Calibration")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect Calibration")

    brier = brier_score_loss(y_true, y_prob)
    ax.set_title(f"Reliability Diagram (Brier Score = {brier:.4f})", fontsize=11, fontweight="bold")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives (Observed)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right")

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
    plt.close(fig)

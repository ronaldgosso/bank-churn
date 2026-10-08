"""Business metrics: expected profit calculation, threshold optimization, gain and lift tables."""

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score

from churn.config import (
    CLV_RETAINED,
    OFFER_COST,
    OFFER_SUCCESS_RATE,
)


def optimal_threshold(
    clv: float = CLV_RETAINED,
    cost: float = OFFER_COST,
    success_rate: float = OFFER_SUCCESS_RATE,
) -> float:
    """Compute theoretically optimal retention decision threshold.

    Derivation:
    Expected profit of contacting customer i with predicted churn probability p_i:
        E[Profit_i] = p_i * success_rate * clv - cost

    Contacting is financially positive when E[Profit_i] > 0:
        p_i * success_rate * clv > cost
        p_i > cost / (success_rate * clv)

    Parameters
    ----------
    clv : float, optional
        Customer Lifetime Value of a retained customer, default CLV_RETAINED (500).
    cost : float, optional
        Cost of retention offer per contacted customer, default OFFER_COST (50).
    success_rate : float, optional
        Offer acceptance/retention success rate, default OFFER_SUCCESS_RATE (0.30).

    Returns
    -------
    float
        Optimal probability threshold (e.g. 0.3333 with defaults).
    """
    denominator = success_rate * clv
    if denominator <= 0:
        raise ValueError("Product of success_rate and clv must be strictly positive.")
    return float(min(max(cost / denominator, 0.0), 1.0))


def expected_profit(
    y_prob: np.ndarray | pd.Series,
    threshold: float | None = None,
    clv: float = CLV_RETAINED,
    cost: float = OFFER_COST,
    success_rate: float = OFFER_SUCCESS_RATE,
) -> tuple[float, int]:
    """Calculate total expected retention profit over scored customers.

    Parameters
    ----------
    y_prob : np.ndarray | pd.Series
        Predicted churn probabilities.
    threshold : float | None, optional
        Decision threshold. If None, uses theoretical optimal threshold.
    clv : float, optional
        Customer Lifetime Value.
    cost : float, optional
        Offer contact cost.
    success_rate : float, optional
        Retention offer success rate.

    Returns
    -------
    tuple[float, int]
        Tuple of (total_expected_profit, total_contacted_count).
    """
    probs = np.asarray(y_prob, dtype=float)
    if threshold is None:
        threshold = optimal_threshold(clv=clv, cost=cost, success_rate=success_rate)

    contact_mask = probs >= threshold
    contacted_count = int(contact_mask.sum())

    if contacted_count == 0:
        return 0.0, 0

    contacted_probs = probs[contact_mask]
    total_profit = float(np.sum(contacted_probs * success_rate * clv - cost))
    return round(total_profit, 2), contacted_count


def actual_retention_profit(
    y_true: np.ndarray | pd.Series,
    y_prob: np.ndarray | pd.Series,
    threshold: float,
    clv: float = CLV_RETAINED,
    cost: float = OFFER_COST,
    success_rate: float = OFFER_SUCCESS_RATE,
) -> dict[str, float]:
    """Compute realized retention metrics against ground truth labels at threshold.

    Parameters
    ----------
    y_true : np.ndarray | pd.Series
        Ground truth binary churn labels (0 or 1).
    y_prob : np.ndarray | pd.Series
        Predicted churn probabilities.
    threshold : float
        Decision threshold.
    clv : float, optional
        Customer lifetime value.
    cost : float, optional
        Contact cost.
    success_rate : float, optional
        Offer success rate.

    Returns
    -------
    dict[str, float]
        Dictionary with contact count, TP count, total cost, gross value, and net profit.
    """
    y_t = np.asarray(y_true, dtype=int)
    y_p = np.asarray(y_prob, dtype=float)
    contacted = y_p >= threshold

    n_contacted = int(contacted.sum())
    true_positives = int((contacted & (y_t == 1)).sum())

    total_cost = float(n_contacted * cost)
    retained_customers = float(true_positives * success_rate)
    gross_value = float(retained_customers * clv)
    net_profit = float(gross_value - total_cost)

    return {
        "threshold": threshold,
        "n_contacted": float(n_contacted),
        "contact_rate": float(n_contacted / len(y_t)) if len(y_t) > 0 else 0.0,
        "true_positives_captured": float(true_positives),
        "capture_rate": float(true_positives / y_t.sum()) if y_t.sum() > 0 else 0.0,
        "total_campaign_cost": total_cost,
        "gross_value_retained": gross_value,
        "net_profit": net_profit,
    }


def gain_table(
    y_true: np.ndarray | pd.Series,
    y_prob: np.ndarray | pd.Series,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Generate decile cumulative gains and lift analysis table.

    Customers are sorted in descending order of predicted churn risk and partitioned
    into n_bins equal-sized deciles.

    Parameters
    ----------
    y_true : np.ndarray | pd.Series
        Ground truth churn labels.
    y_prob : np.ndarray | pd.Series
        Predicted churn probabilities.
    n_bins : int, optional
        Number of quantile bins, default 10 (deciles).

    Returns
    -------
    pd.DataFrame
        Gains table containing counts, decile churn rates, cumulative gains, and lift.
    """
    df = pd.DataFrame({
        "y_true": np.asarray(y_true, dtype=int),
        "y_prob": np.asarray(y_prob, dtype=float),
    }).sort_values(by="y_prob", ascending=False).reset_index(drop=True)

    total_customers = len(df)
    total_churners = int(df["y_true"].sum())
    base_churn_rate = float(total_churners / total_customers)

    # Bin into deciles
    df["decile"] = pd.qcut(
        df.index,
        q=n_bins,
        labels=[f"D{i}" for i in range(1, n_bins + 1)],
    )

    records: list[dict[str, Any]] = []
    cum_customers = 0
    cum_churners = 0

    for decile_label, group in df.groupby("decile", observed=False):
        n_cust = len(group)
        n_churn = int(group["y_true"].sum())
        decile_rate = float(n_churn / n_cust) if n_cust > 0 else 0.0

        cum_customers += n_cust
        cum_churners += n_churn

        cum_cust_pct = float(cum_customers / total_customers)
        cum_gain_pct = float(cum_churners / total_churners) if total_churners > 0 else 0.0
        decile_lift = float(decile_rate / base_churn_rate) if base_churn_rate > 0 else 1.0
        cum_lift = float(cum_gain_pct / cum_cust_pct) if cum_cust_pct > 0 else 1.0

        records.append({
            "decile": decile_label,
            "customers": n_cust,
            "churners": n_churn,
            "decile_churn_rate": round(decile_rate, 4),
            "cum_customers": cum_customers,
            "cum_customer_pct": round(cum_cust_pct, 4),
            "cum_churners": cum_churners,
            "cum_gain_pct": round(cum_gain_pct, 4),
            "decile_lift": round(decile_lift, 2),
            "cum_lift": round(cum_lift, 2),
        })

    return pd.DataFrame(records)


def lift_at_k(
    y_true: np.ndarray | pd.Series,
    y_prob: np.ndarray | pd.Series,
    k: float = 0.20,
    clv: float = CLV_RETAINED,
    cost: float = OFFER_COST,
    success_rate: float = OFFER_SUCCESS_RATE,
) -> tuple[float, float, float]:
    """Calculate cumulative lift, churner capture percentage, and estimated profit for top k% customers.

    Parameters
    ----------
    y_true : np.ndarray | pd.Series
        Ground truth churn labels.
    y_prob : np.ndarray | pd.Series
        Predicted probabilities.
    k : float, optional
        Top fraction of customers by risk (e.g. 0.10, 0.20, 0.30), default 0.20.
    clv : float, optional
        CLV retained.
    cost : float, optional
        Offer cost.
    success_rate : float, optional
        Offer success rate.

    Returns
    -------
    tuple[float, float, float]
        Tuple of (lift_multiplier, capture_rate_pct, estimated_net_profit).
    """
    df = pd.DataFrame({
        "y_true": np.asarray(y_true, dtype=int),
        "y_prob": np.asarray(y_prob, dtype=float),
    }).sort_values(by="y_prob", ascending=False).reset_index(drop=True)

    n_top = int(np.ceil(k * len(df)))
    top_slice = df.iloc[:n_top]

    total_churners = int(df["y_true"].sum())
    top_churners = int(top_slice["y_true"].sum())

    capture_rate = float(top_churners / total_churners) if total_churners > 0 else 0.0
    lift = float(capture_rate / k) if k > 0 else 1.0

    # Realized net profit on this top k segment
    total_campaign_cost = float(n_top * cost)
    gross_value = float(top_churners * success_rate * clv)
    net_profit = float(gross_value - total_campaign_cost)

    return round(lift, 2), round(capture_rate, 4), round(net_profit, 2)


def find_f1_optimal_threshold(
    y_true: np.ndarray | pd.Series,
    y_prob: np.ndarray | pd.Series,
) -> tuple[float, float]:
    """Find threshold that maximizes discrete F1 score via out-of-fold predictions.

    Parameters
    ----------
    y_true : np.ndarray | pd.Series
        Ground truth labels.
    y_prob : np.ndarray | pd.Series
        Predicted probabilities.

    Returns
    -------
    tuple[float, float]
        Tuple of (best_threshold, best_f1_score).
    """
    y_t = np.asarray(y_true, dtype=int)
    y_p = np.asarray(y_prob, dtype=float)

    thresholds = np.linspace(0.05, 0.95, 181)
    best_t = 0.5
    best_f1 = -1.0

    for t in thresholds:
        pred = (y_p >= t).astype(int)
        score = float(f1_score(y_t, pred, zero_division=0))
        if score > best_f1:
            best_f1 = score
            best_t = float(t)

    return round(best_t, 3), round(best_f1, 4)


def plot_gain_and_lift_charts(
    gains_df: pd.DataFrame,
    save_path: Path | str | None = None,
) -> None:
    """Plot cumulative gain and lift charts side-by-side."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))

    x = [0.0] + gains_df["cum_customer_pct"].tolist()
    y_gain = [0.0] + gains_df["cum_gain_pct"].tolist()

    # Cumulative Gain Chart
    ax1.plot(x, y_gain, marker="o", color="#386cb0", linewidth=2.5, label="Model Gain Curve")
    ax1.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Baseline (Random)")
    ax1.set_title("Cumulative Gains Chart", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Fraction of Customer Base Contacted")
    ax1.set_ylabel("Fraction of Total Churners Captured")
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1.05)
    ax1.legend(loc="lower right")

    # Decile Lift Chart
    deciles = gains_df["decile"].tolist()
    lifts = gains_df["decile_lift"].tolist()
    bars = ax2.bar(deciles, lifts, color="#7fc97f", edgecolor="#444444", width=0.5)
    ax2.axhline(1.0, color="gray", linestyle="--", label="Baseline Lift (1.0x)")
    ax2.set_title("Decile Lift Chart", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Risk Decile (D1 = Highest Risk)")
    ax2.set_ylabel("Lift Multiplier")
    ax2.legend()

    for bar in bars:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, yval + 0.08, f"{yval:.1f}x", ha="center", fontsize=9, fontweight="bold")

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_profit_vs_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    optimal_t: float,
    f1_t: float,
    clv: float = CLV_RETAINED,
    cost: float = OFFER_COST,
    success_rate: float = OFFER_SUCCESS_RATE,
    save_path: Path | str | None = None,
) -> None:
    """Plot net profit curve across thresholds, highlighting default 0.5, optimal, and F1-optimal."""
    thresholds = np.linspace(0.05, 0.90, 100)
    profits = [
        actual_retention_profit(y_true, y_prob, t, clv=clv, cost=cost, success_rate=success_rate)["net_profit"]
        for t in thresholds
    ]

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(thresholds, profits, color="#2b5c8f", linewidth=2.5, label="Net Campaign Profit ($)")

    # Key threshold markers
    for t, label, color in [
        (optimal_t, f"Business Optimal ({optimal_t:.3f})", "green"),
        (0.5, "Default 0.50", "gray"),
        (f1_t, f"F1-Optimal ({f1_t:.3f})", "purple"),
    ]:
        p = actual_retention_profit(y_true, y_prob, t, clv=clv, cost=cost, success_rate=success_rate)["net_profit"]
        ax.scatter([t], [p], color=color, s=80, zorder=5)
        ax.axvline(t, color=color, linestyle="--", alpha=0.7, label=f"{label}: ${p:,.0f}")

    ax.set_title("Retention Campaign Net Profit vs. Decision Threshold", fontsize=12, fontweight="bold")
    ax.set_xlabel("Probability Decision Threshold")
    ax.set_ylabel("Net Campaign Profit ($)")
    ax.legend()

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_threshold_confusion_matrix(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    save_path: Path | str | None = None,
) -> None:
    """Plot confusion matrix at chosen optimal threshold."""
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax)

    classes = ["Retained (0)", "Exited (1)"]
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(classes)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes)

    # Text annotations
    thresh_val = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                f"{cm[i, j]:,}",
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh_val else "black",
                fontsize=11,
                fontweight="bold",
            )

    ax.set_title(f"Confusion Matrix @ Threshold = {threshold:.3f}", fontsize=11, fontweight="bold")
    ax.set_ylabel("Actual Label")
    ax.set_xlabel("Predicted Label")

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
    plt.close(fig)

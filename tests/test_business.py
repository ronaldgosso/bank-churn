"""Unit tests for business metrics, expected profit, thresholds, and gain/lift tables."""

import numpy as np
import pytest

from churn.business import (
    actual_retention_profit,
    expected_profit,
    find_f1_optimal_threshold,
    gain_table,
    lift_at_k,
    optimal_threshold,
)


def test_optimal_threshold_calculation() -> None:
    """Verify optimal threshold formula: cost / (success_rate * clv)."""
    # Defaults: 50 / (0.30 * 500) = 50 / 150 = 0.33333...
    opt_t = optimal_threshold(clv=500.0, cost=50.0, success_rate=0.30)
    assert opt_t == pytest.approx(1.0 / 3.0, rel=1e-3)

    # Edge cases
    assert optimal_threshold(clv=100.0, cost=100.0, success_rate=1.0) == 1.0
    with pytest.raises(ValueError, match="strictly positive"):
        optimal_threshold(clv=0.0)


def test_expected_profit_calculation() -> None:
    """Verify expected profit calculation across probability array."""
    # Two customers: p1 = 0.8, p2 = 0.1
    # Profit 1: 0.8 * 0.3 * 500 - 50 = 120 - 50 = +70
    # Profit 2: 0.1 * 0.3 * 500 - 50 = 15 - 50 = -35 (should not be contacted at threshold 0.333)
    probs = np.array([0.8, 0.1])
    prof, n_contacted = expected_profit(probs, threshold=0.333, clv=500, cost=50, success_rate=0.3)
    assert n_contacted == 1
    assert prof == pytest.approx(70.0)

    # If nobody contacted
    prof_zero, n_zero = expected_profit(probs, threshold=0.9)
    assert n_zero == 0
    assert prof_zero == 0.0


def test_actual_retention_profit() -> None:
    """Verify actual profit realized against ground truth labels."""
    y_true = np.array([1, 0, 1, 0])
    y_prob = np.array([0.9, 0.8, 0.2, 0.1])  # At 0.5: contact indices 0, 1. Index 0 is TP, 1 is FP.
    # Cost = 2 * 50 = 100
    # Value = 1 TP * 0.3 * 500 = 150
    # Net profit = 150 - 100 = 50
    result = actual_retention_profit(y_true, y_prob, threshold=0.5, clv=500, cost=50, success_rate=0.3)
    assert result["n_contacted"] == 2
    assert result["true_positives_captured"] == 1
    assert result["total_campaign_cost"] == 100.0
    assert result["gross_value_retained"] == 150.0
    assert result["net_profit"] == 50.0


def test_gain_table_structure() -> None:
    """Verify gain table produces valid decile cumulative metrics and monotonicity."""
    # 100 samples with 20 churners placed mostly in top probabilities
    np.random.seed(42)
    y_prob = np.linspace(0.99, 0.01, 100)
    y_true = np.zeros(100, dtype=int)
    y_true[:18] = 1  # 18 churners in top deciles
    y_true[40] = 1
    y_true[80] = 1

    df_gain = gain_table(y_true, y_prob, n_bins=10)
    assert len(df_gain) == 10
    assert "decile_lift" in df_gain.columns
    assert "cum_gain_pct" in df_gain.columns

    # Final cumulative gain must be 1.0 (100%)
    assert df_gain.iloc[-1]["cum_gain_pct"] == pytest.approx(1.0)
    assert df_gain.iloc[0]["cum_gain_pct"] > 0.40  # Top decile captured large majority


def test_lift_at_k() -> None:
    """Verify lift_at_k computes lift multiplier and profit for budget constraints."""
    y_prob = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05])
    y_true = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])  # 2 churners out of 10

    # Top 20% = top 2 customers (both are churners)
    lift, capture_rate, profit = lift_at_k(y_true, y_prob, k=0.20, clv=500, cost=50, success_rate=0.3)
    assert capture_rate == 1.0  # Captured 2 / 2
    assert lift == 5.0  # 1.0 / 0.20 = 5.0x lift
    # Cost = 2 * 50 = 100, Value = 2 * 0.3 * 500 = 300, Profit = 200
    assert profit == 200.0


def test_find_f1_optimal_threshold() -> None:
    """Verify threshold optimization finds valid threshold maximizing F1."""
    y_true = np.array([0, 0, 1, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.6, 0.7, 0.8])

    best_t, best_f1 = find_f1_optimal_threshold(y_true, y_prob)
    assert 0.2 < best_t < 0.6
    assert best_f1 == 1.0

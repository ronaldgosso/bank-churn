"""Unit tests for Streamlit application utilities and page data dependencies."""

import py_compile
from pathlib import Path

from utils import (
    COLOR_HIGH,
    COLOR_LOW,
    COLOR_MED,
    get_cached_business_summary,
    get_cached_fairness_summary,
    get_cached_metadata,
    get_cached_metrics,
    get_cached_model,
    get_cached_model_comparison,
    get_sample_portfolio,
)

from churn.config import ALL_FEATURE_COLS


def test_color_palette_constants() -> None:
    """Verify that required consistent color palette is configured."""
    assert COLOR_LOW.startswith("#")
    assert COLOR_MED.startswith("#")
    assert COLOR_HIGH.startswith("#")
    assert COLOR_LOW != COLOR_HIGH


def test_cached_loaders_return_expected_structures() -> None:
    """Verify that cached data loaders load genuine project artifacts."""
    metrics = get_cached_metrics()
    assert "roc_auc" in metrics
    assert "pr_auc" in metrics
    assert "decision_threshold" in metrics
    assert metrics["roc_auc"] >= 0.85

    biz = get_cached_business_summary()
    assert "optimal_threshold" in biz

    fairness = get_cached_fairness_summary()
    assert "gender_metrics" in fairness
    assert "geography_metrics" in fairness

    metadata = get_cached_metadata()
    assert "model_architecture" in metadata

    comp_df = get_cached_model_comparison()
    assert not comp_df.empty
    assert "model" in comp_df.columns
    assert "roc_auc_mean" in comp_df.columns


def test_sample_portfolio_schema() -> None:
    """Verify that template portfolio matches required feature schema."""
    sample_df = get_sample_portfolio()
    assert not sample_df.empty
    for col in ALL_FEATURE_COLS:
        assert col in sample_df.columns
    assert "Exited" not in sample_df.columns


def test_cached_model_inference() -> None:
    """Verify that cached model loads and executes prediction."""
    model, metadata = get_cached_model()
    assert model is not None
    assert "decision_threshold" in metadata

    sample_df = get_sample_portfolio().head(3)
    probs = model.predict_proba(sample_df)[:, 1]
    assert len(probs) == 3
    assert all(0.0 <= p <= 1.0 for p in probs)


def test_app_pages_syntax_compilation() -> None:
    """Verify that all Streamlit app scripts compile with clean Python syntax."""
    app_root = Path(__file__).resolve().parent.parent / "app"
    py_files = [app_root / "Home.py"] + list((app_root / "pages").glob("*.py"))

    assert len(py_files) >= 6  # Home + 5 pages

    for py_file in py_files:
        compiled = py_compile.compile(str(py_file), doraise=True)
        assert compiled is not None

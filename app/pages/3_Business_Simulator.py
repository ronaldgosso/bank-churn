"""Business Simulator: Interactive ROI simulation, threshold optimization, and decile gain curves."""

import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st

# Setup sys.path
APP_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = APP_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from utils import (
    COLOR_HIGH,
    COLOR_LOW,
    COLOR_PRIMARY,
    apply_custom_styles,
    get_cached_business_summary,
    get_cached_holdout_predictions,
    get_cached_model,
    get_cached_test_data,
)

from churn.business import (
    actual_retention_profit,
    expected_profit,
    gain_table,
    lift_at_k,
    optimal_threshold,
)

st.set_page_config(
    page_title="Business Simulator · Bank Churn",
    page_icon=":material/query_stats:",
    layout="wide",
)

apply_custom_styles()

st.title("Retention Campaign Business Simulator")
st.markdown(
    "Simulate the financial return of retention interventions. Adjust unit economics and campaign budget constraints "
    "to compute theoretical optimal decision boundaries, net retention profit, and cumulative capture curves."
)

st.divider()

# Load test dataset and model for interactive calculation
test_df = get_cached_test_data()
model, _ = get_cached_model()
biz_summary = get_cached_business_summary()

# Extract ground truth and predicted probabilities seamlessly
y_true: np.ndarray | None = None
y_prob: np.ndarray | None = None

if not test_df.empty and "Exited" in test_df.columns:
    y_true = test_df["Exited"].to_numpy().astype(int)
    if "churn_probability" in test_df.columns:
        y_prob = test_df["churn_probability"].to_numpy().astype(float)
    elif model is not None:
        try:
            X_test = test_df.drop(columns=["Exited"])
            y_prob = model.predict_proba(X_test)[:, 1].astype(float)
        except Exception:
            y_prob = None

if y_true is None or y_prob is None or len(y_true) == 0:
    y_true, y_prob = get_cached_holdout_predictions()

# Interactive Controls
st.subheader("1. Campaign Economics & Budget Parameters")

c1, c2, c3, c4 = st.columns(4)

with c1:
    sim_clv = st.slider(
        "Customer Lifetime Value (CLV $)",
        min_value=100.0,
        max_value=2000.0,
        value=500.0,
        step=50.0,
        help="Gross monetary value preserved when an at-risk customer is successfully saved.",
    )

with c2:
    sim_cost = st.slider(
        "Retention Offer Cost ($)",
        min_value=10.0,
        max_value=200.0,
        value=50.0,
        step=5.0,
        help="Direct operational cost of making the retention intervention per customer.",
    )

with c3:
    sim_success_rate = st.slider(
        "Offer Save Rate (%)",
        min_value=5.0,
        max_value=80.0,
        value=30.0,
        step=5.0,
        help="Probability that an at-risk customer accepts the offer and remains with the bank.",
    ) / 100.0

with c4:
    sim_budget_k = st.slider(
        "Budget Limit (Top K% Risk)",
        min_value=5,
        max_value=50,
        value=20,
        step=5,
        help="Operational constraint: Contact only the top K percent highest-risk customers.",
    )

# Compute live figures
sim_opt_thresh = optimal_threshold(clv=sim_clv, cost=sim_cost, success_rate=sim_success_rate)

# Profit at optimal threshold
opt_profit, opt_contacted = expected_profit(
    y_prob, threshold=sim_opt_thresh, clv=sim_clv, cost=sim_cost, success_rate=sim_success_rate
)
opt_actual_profit, opt_cm, opt_capture = actual_retention_profit(
    y_true, y_prob, threshold=sim_opt_thresh, clv=sim_clv, cost=sim_cost, success_rate=sim_success_rate
)

# Profit at default 0.5 threshold
def_profit, def_contacted = expected_profit(
    y_prob, threshold=0.5, clv=sim_clv, cost=sim_cost, success_rate=sim_success_rate
)
def_actual_profit, def_cm, def_capture = actual_retention_profit(
    y_true, y_prob, threshold=0.5, clv=sim_clv, cost=sim_cost, success_rate=sim_success_rate
)

# Top K% calculation
budget_k_frac = sim_budget_k / 100.0
k_threshold_val = float(np.quantile(y_prob, 1.0 - budget_k_frac))
budget_actual_profit, budget_cm, budget_capture = actual_retention_profit(
    y_true,
    y_prob,
    threshold=k_threshold_val,
    clv=sim_clv,
    cost=sim_cost,
    success_rate=sim_success_rate,
)
budget_lift, budget_captured_count = lift_at_k(y_true, y_prob, k_percent=sim_budget_k)

st.divider()
st.subheader("2. Financial Performance Comparison")

kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)

with kpi_col1:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">Optimal Threshold</div>
            <div class="metric-value">{sim_opt_thresh:.3f}</div>
            <div class="metric-subtitle">Cost / (Save Rate &times; CLV)</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with kpi_col2:
    profit_delta = opt_actual_profit - def_actual_profit
    delta_str = f"+${profit_delta:,.0f} vs 0.5 default" if profit_delta >= 0 else f"-${abs(profit_delta):,.0f}"
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">Optimal Threshold Net Profit</div>
            <div class="metric-value">${opt_actual_profit:,.0f}</div>
            <div class="metric-subtitle">{delta_str}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with kpi_col3:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">Default 0.5 Threshold Profit</div>
            <div class="metric-value">${def_actual_profit:,.0f}</div>
            <div class="metric-subtitle">Misses {opt_capture - def_capture:.1%} of churners</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with kpi_col4:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">Budget (Top {sim_budget_k}%) Profit</div>
            <div class="metric-value">${budget_actual_profit:,.0f}</div>
            <div class="metric-subtitle">Lift: {budget_lift:.2f}x · Capture: {budget_capture:.1%}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Dynamic Acceptance Statement Banner
st.success(
    f"**Decision Policy Statement:** Contacting the top **{sim_budget_k}%** of customers captures "
    f"**{budget_capture*100:.1f}%** of churners (lift **{budget_lift:.2f}x**) and yields an estimated profit of **${budget_actual_profit:,.2f}** "
    f"on the holdout portfolio."
)

econ_img = PROJECT_ROOT / "public" / "retention_economics_concept.png"
if econ_img.exists():
    with st.expander("Cost-Sensitive Retention Economics & Optimization Formula", expanded=False):
        st.image(
            str(econ_img),
            caption="Mathematical Trade-off: CLV vs. Intervention Offer Cost & Empirical Churn Capture",
            use_container_width=True,
        )

st.divider()
st.subheader("3. Interactive Sensitivity & Gain Analytics")

chart_left, chart_right = st.columns([1, 1])

with chart_left:
    # Profit vs Threshold Curve
    threshold_range = np.linspace(0.05, 0.95, 91)
    profits_curve = [
        actual_retention_profit(
            y_true,
            y_prob,
            threshold=float(t),
            clv=sim_clv,
            cost=sim_cost,
            success_rate=sim_success_rate,
        )[0]
        for t in threshold_range
    ]

    p_fig = go.Figure()
    p_fig.add_trace(
        go.Scatter(
            x=threshold_range,
            y=profits_curve,
            mode="lines",
            name="Net Campaign Profit",
            line={"color": COLOR_PRIMARY, "width": 3},
        )
    )
    p_fig.add_vline(
        x=sim_opt_thresh,
        line_dash="dash",
        line_color=COLOR_HIGH,
        annotation_text=f"Optimal ({sim_opt_thresh:.3f})",
        annotation_position="top left",
    )
    p_fig.add_vline(
        x=0.5,
        line_dash="dot",
        line_color="#64748b",
        annotation_text="Default (0.500)",
        annotation_position="top right",
    )
    p_fig.update_layout(
        title="Net Profit vs. Decision Threshold Curve",
        xaxis_title="Classification Threshold",
        yaxis_title="Net Profit ($ USD)",
        template="plotly_white",
        height=380,
    )
    st.plotly_chart(p_fig, use_container_width=True)

with chart_right:
    # Cumulative Gain Curve
    g_table = gain_table(y_true, y_prob, n_bins=10)
    gain_x = [0.0] + list(g_table["cumulative_population_pct"])
    gain_y = [0.0] + list(g_table["cumulative_churn_captured_pct"])

    g_fig = go.Figure()
    g_fig.add_trace(
        go.Scatter(
            x=gain_x,
            y=gain_y,
            mode="lines+markers",
            name="Model Capture",
            line={"color": COLOR_LOW, "width": 3},
        )
    )
    g_fig.add_trace(
        go.Scatter(
            x=[0.0, 1.0],
            y=[0.0, 1.0],
            mode="lines",
            name="Random Baseline",
            line={"color": "#94a3b8", "dash": "dash"},
        )
    )
    # Highlight user chosen top K%
    g_fig.add_vline(
        x=sim_budget_k / 100.0,
        line_dash="dot",
        line_color=COLOR_HIGH,
        annotation_text=f"Top {sim_budget_k}% ({budget_capture:.1%})",
        annotation_position="bottom right",
    )
    g_fig.update_layout(
        title="Cumulative Gain Chart (Churn Capture)",
        xaxis_title="Proportion of Customers Contacted",
        yaxis_title="Cumulative Churners Captured",
        template="plotly_white",
        height=380,
    )
    st.plotly_chart(g_fig, use_container_width=True)

# Gain and Decile Table
st.markdown("### Decile Gain & Lift Breakdown")
st.dataframe(
    g_table.style.format(
        {
            "churn_rate": "{:.2%}",
            "lift": "{:.2f}x",
            "cumulative_lift": "{:.2f}x",
            "cumulative_population_pct": "{:.1%}",
            "cumulative_churn_captured_pct": "{:.1%}",
        }
    ),
    use_container_width=True,
)

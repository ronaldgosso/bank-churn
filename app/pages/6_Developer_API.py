"""Developer Hub & REST API Console: Microservice documentation, schema references, and live test runner."""

import json
import sys
import time
from pathlib import Path

import pandas as pd
import streamlit as st

# Setup sys.path
APP_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = APP_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from utils import (
    apply_custom_styles,
    get_cached_metadata,
    get_cached_model,
)

from churn.config import ALL_FEATURE_COLS
from churn.predict import (
    compute_actionable_recourse,
    predict_batch,
    predict_one,
)

st.set_page_config(
    page_title="Developer API · Bank Churn",
    page_icon=":material/terminal:",
    layout="wide",
)

apply_custom_styles()

st.title("Developer Hub & REST API Console")
st.markdown(
    "Interactive API documentation, request payload builder, live in-browser test runner, "
    "and deployment specifications for the production churn inference microservice."
)

st.divider()

# Load model metadata
metadata = get_cached_metadata()
model, _ = get_cached_model()
threshold = float(metadata.get("decision_threshold", 0.3333))

# Architecture Callout
with st.expander("Service Architecture: Streamlit UI vs. FastAPI Microservice", expanded=True):
    st.markdown(
        """
        ### Understanding the Dual-Layer Architecture

        This repository contains **two complementary production interfaces**:

        | Interface | Hosting Layer | Protocol & Port | Primary Purpose |
        | :--- | :--- | :--- | :--- |
        | **Streamlit Decision Dashboard** | Streamlit Community Cloud (`*.streamlit.app`) | WebSockets / HTTP (8501) | Interactive visual decision-support for bankers & campaign managers |
        | **FastAPI REST Microservice** | Docker Container (`churn.api:app`) | REST / JSON (8000) | Low-latency programmatic inference for core banking systems & mobile apps |

        > **Architecture Note:**
        > Navigating to an arbitrary path like `https://<app>.streamlit.app/api/v2/user/details` returns an **HTTP 404**
        > because Streamlit Community Cloud's edge router is exclusively configured to serve the WebSocket frontend.
        > To consume the REST API programmatically, deploy the provided [`Dockerfile`](https://github.com/ronaldgosso/bank-churn/blob/main/Dockerfile)
        > on a container host (Render, Railway, Fly.io, AWS ECS, GCP Cloud Run) or launch it locally using Uvicorn.
        """
    )

st.markdown("### 1. Interactive REST API Console & Payload Simulator")

tab_single, tab_recourse, tab_batch, tab_health = st.tabs(
    ["POST /predict (Single Customer)", "POST /predict/recourse (Prescriptive)", "POST /predict/batch", "GET /health & /meta"]
)

# Representative Customer Presets
presets = {
    "High Risk Archetype": {
        "CreditScore": 620,
        "Geography": "Germany",
        "Gender": "Female",
        "Age": 52,
        "Tenure": 2,
        "Balance": 115000.0,
        "NumOfProducts": 3,
        "HasCrCard": 1,
        "IsActiveMember": 0,
        "EstimatedSalary": 75000.0,
    },
    "Borderline Archetype": {
        "CreditScore": 680,
        "Geography": "France",
        "Gender": "Female",
        "Age": 42,
        "Tenure": 4,
        "Balance": 92000.0,
        "NumOfProducts": 1,
        "HasCrCard": 1,
        "IsActiveMember": 0,
        "EstimatedSalary": 82000.0,
    },
    "Low Risk Archetype": {
        "CreditScore": 750,
        "Geography": "France",
        "Gender": "Male",
        "Age": 32,
        "Tenure": 7,
        "Balance": 45000.0,
        "NumOfProducts": 2,
        "HasCrCard": 1,
        "IsActiveMember": 1,
        "EstimatedSalary": 65000.0,
    },
}

with tab_single:
    col_req, col_res = st.columns([1, 1])

    with col_req:
        st.markdown("#### Request Configuration")
        selected_preset = st.selectbox(
            "Load Sample Customer Payload:",
            options=list(presets.keys()),
            key="preset_single",
        )
        sample_payload = presets[selected_preset]

        payload_text = st.text_area(
            "JSON Request Body:",
            value=json.dumps(sample_payload, indent=2),
            height=260,
            key="body_single",
        )

        send_btn = st.button("Send Request to /predict", type="primary", key="send_single")

    with col_res:
        st.markdown("#### Response Inspector")
        if send_btn or "last_single_res" in st.session_state:
            try:
                parsed_json = json.loads(payload_text)
                t0 = time.perf_counter()
                res_data = predict_one(parsed_json, model=model, threshold=threshold)
                latency_ms = (time.perf_counter() - t0) * 1000

                st.markdown(
                    f"**Status:** `200 OK` · **Latency:** `{latency_ms:.1f} ms` · "
                    f"**Content-Type:** `application/json`"
                )
                st.json(res_data)
                st.session_state["last_single_res"] = res_data
            except Exception as e:
                st.error(f"Inference error: {e}")
        else:
            st.info("Click **Send Request to /predict** to trigger real-time inference execution.")

with tab_recourse:
    c_r_left, c_r_right = st.columns([1, 1])
    with c_r_left:
        st.markdown("#### Prescriptive Recourse Request")
        rec_preset = st.selectbox(
            "Load Customer Profile:",
            options=list(presets.keys()),
            index=0,
            key="preset_recourse",
        )
        rec_cust = presets[rec_preset]

        rec_clv = st.number_input("Customer Lifetime Value (CLV $):", value=500.0, step=50.0)
        rec_save = st.slider("Offer Success Rate (%):", 5, 80, 30) / 100.0

        full_rec_payload = {
            "customer": rec_cust,
            "clv": rec_clv,
            "save_rate": rec_save,
        }

        rec_text = st.text_area(
            "JSON Body:",
            value=json.dumps(full_rec_payload, indent=2),
            height=230,
            key="body_recourse",
        )

        send_rec_btn = st.button("Send Request to /predict/recourse", type="primary", key="send_recourse")

    with c_r_right:
        st.markdown("#### Prescriptive Response Inspector")
        if send_rec_btn or "last_rec_res" in st.session_state:
            try:
                parsed_rec = json.loads(rec_text)
                t0 = time.perf_counter()
                rec_out = compute_actionable_recourse(
                    customer=parsed_rec["customer"],
                    model=model,
                    threshold=threshold,
                    clv=parsed_rec.get("clv", 500.0),
                    save_rate=parsed_rec.get("save_rate", 0.30),
                )
                latency_ms = (time.perf_counter() - t0) * 1000

                st.markdown(
                    f"**Status:** `200 OK` · **Latency:** `{latency_ms:.1f} ms` · "
                    f"**Content-Type:** `application/json`"
                )
                st.json(rec_out)
                st.session_state["last_rec_res"] = rec_out
            except Exception as e:
                st.error(f"Recourse calculation error: {e}")
        else:
            st.info("Click **Send Request to /predict/recourse** to test counterfactual generation.")

with tab_batch:
    st.markdown("#### Batch Prediction (/predict/batch)")
    batch_records = [presets["High Risk Archetype"], presets["Borderline Archetype"], presets["Low Risk Archetype"]]
    st.markdown("Array of customer records submitted in a single HTTP POST:")
    st.code(json.dumps(batch_records, indent=2), language="json")

    if st.button("Execute Batch Prediction (3 records)", type="primary"):
        batch_df = pd.DataFrame(batch_records)
        scored_batch = predict_batch(batch_df, model=model, threshold=threshold)
        b_col_cfg = {
            "churn_probability": st.column_config.ProgressColumn(
                "Churn Risk",
                format="%.2f",
                min_value=0.0,
                max_value=1.0,
            ),
            "Balance": st.column_config.NumberColumn(
                "Account Balance",
                format="$%.2f",
            ),
            "contact_recommendation": st.column_config.CheckboxColumn("Contact"),
        }
        st.dataframe(
            scored_batch[
                ["churn_probability", "risk_tier", "contact_recommendation", "CreditScore", "Geography", "Age", "Balance"]
            ],
            column_config=b_col_cfg,
            use_container_width=True,
        )

with tab_health:
    st.markdown("#### System Health & Metadata Endpoints")
    h1, h2 = st.columns(2)
    with h1:
        st.markdown("**`GET /health`** (Microservice status check)")
        health_resp = {
            "status": "ok",
            "model_name": metadata.get("model_name", "xgboost"),
            "decision_threshold": threshold,
            "training_timestamp": metadata.get("training_timestamp_utc", "2026-10-08"),
        }
        st.json(health_resp)
    with h2:
        st.markdown("**`GET /meta`** (Feature columns & business defaults)")
        meta_resp = {
            "model_metadata": metadata,
            "feature_columns": ALL_FEATURE_COLS,
            "business_defaults": {"clv_retained": 500.0, "offer_cost": 50.0, "save_rate": 0.30},
        }
        st.json(meta_resp)

st.divider()

# Multi-Language Code Snippets
st.markdown("### 2. Multi-Language Code Integration")
lang_curl, lang_py, lang_ts = st.tabs(["cURL (Bash)", "Python (Requests)", "JavaScript / TypeScript (Fetch)"])

with lang_curl:
    st.code(
        """# 1. Probe microservice health
curl -X GET "http://localhost:8000/health"

# 2. Score single customer churn risk
curl -X POST "http://localhost:8000/predict" \\
     -H "Content-Type: application/json" \\
     -d '{
       "CreditScore": 650,
       "Geography": "Germany",
       "Gender": "Female",
       "Age": 48,
       "Tenure": 3,
       "Balance": 85000.0,
       "NumOfProducts": 3,
       "HasCrCard": 1,
       "IsActiveMember": 0,
       "EstimatedSalary": 60000.0
     }'

# 3. Compute prescriptive counterfactual recourse
curl -X POST "http://localhost:8000/predict/recourse" \\
     -H "Content-Type: application/json" \\
     -d '{
       "customer": {
         "CreditScore": 650,
         "Geography": "Germany",
         "Gender": "Female",
         "Age": 48,
         "Tenure": 3,
         "Balance": 85000.0,
         "NumOfProducts": 3,
         "HasCrCard": 1,
         "IsActiveMember": 0,
         "EstimatedSalary": 60000.0
       },
       "clv": 500.0,
       "save_rate": 0.30
     }'
        """,
        language="bash",
    )

with lang_py:
    st.code(
        """import requests

API_BASE = "http://localhost:8000"

customer_payload = {
    "CreditScore": 650,
    "Geography": "Germany",
    "Gender": "Female",
    "Age": 48,
    "Tenure": 3,
    "Balance": 85000.0,
    "NumOfProducts": 3,
    "HasCrCard": 1,
    "IsActiveMember": 0,
    "EstimatedSalary": 60000.0,
}

# Real-time scoring
response = requests.post(f"{API_BASE}/predict", json=customer_payload)
result = response.json()

print(f"Churn Probability: {result['probability']:.1%}")
print(f"Risk Tier: {result['risk_tier']}")
print(f"Contact Recommended: {result['contact']}")
print("Top Drivers:", result["top_drivers"])
        """,
        language="python",
    )

with lang_ts:
    st.code(
        """// TypeScript / ES6 Fetch Example
interface ChurnScoreResponse {
  probability: number;
  risk_tier: 'High' | 'Medium' | 'Low';
  contact: boolean;
  top_drivers: string[];
}

async function evaluateCustomerRisk(customer: Record<string, unknown>): Promise<ChurnScoreResponse> {
  const res = await fetch("http://localhost:8000/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(customer),
  });

  if (!res.ok) {
    throw new Error(`API returned HTTP ${res.status}`);
  }

  return await res.json();
}
        """,
        language="typescript",
    )

st.divider()

# Deployment & Docker Instructions
st.markdown("### 3. Container Deployment & Microservice Operations")
d_col1, d_col2 = st.columns([1, 1])

with d_col1:
    st.markdown("#### Build and Run Locally with Docker")
    st.code(
        """# Build the container image
docker build -t bank-churn-api:latest .

# Run the FastAPI microservice on port 8000
docker run -d -p 8000:8000 --name bank-churn-service bank-churn-api:latest

# Verify live health status
curl http://localhost:8000/health
        """,
        language="bash",
    )

with d_col2:
    st.markdown("#### 1-Click Cloud Hosting Options")
    st.markdown(
        """
        - **Render / Railway / Fly.io:** Connect repository, select Docker runtime, set port `8000`.
        - **AWS ECS / Google Cloud Run:** Push Docker image to AWS ECR or GCP Artifact Registry, deploy as serverless container.
        - **Hugging Face Spaces:** Create Space with SDK `Docker` to host both UI and REST services.
        """
    )

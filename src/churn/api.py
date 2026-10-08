"""FastAPI microservice for real-time customer churn inference, explainability, and prescriptive recourse."""

from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from churn.config import (
    ALL_FEATURE_COLS,
    CLV_RETAINED,
    OFFER_COST,
    OFFER_SUCCESS_RATE,
)
from churn.predict import (
    CustomerInput,
    compute_actionable_recourse,
    load_production_model,
    predict_batch,
    predict_one,
)

app = FastAPI(
    title="Bank Customer Churn Prediction API",
    description=(
        "Production-grade REST microservice serving real-time churn risk probabilities, "
        "calibrated risk tiers, SHAP explanatory drivers, and prescriptive retention recourse."
    ),
    version="1.1.0",
)


class HealthResponse(BaseModel):
    """Health check response schema."""

    status: str
    model_name: str
    decision_threshold: float
    training_timestamp: str


class RecourseRequest(BaseModel):
    """Recourse simulation request schema."""

    customer: CustomerInput
    clv: float = Field(default=CLV_RETAINED, ge=0.0, description="Customer Lifetime Value ($)")
    save_rate: float = Field(
        default=OFFER_SUCCESS_RATE, ge=0.0, le=1.0, description="Retention offer acceptance probability"
    )


@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
def health() -> HealthResponse:
    """Return health status and loaded model metadata."""
    try:
        _, metadata = load_production_model()
        return HealthResponse(
            status="ok",
            model_name=str(metadata.get("model_name", "xgboost")),
            decision_threshold=float(metadata.get("decision_threshold", 0.3333)),
            training_timestamp=str(metadata.get("training_timestamp_utc", "unknown")),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/meta", tags=["Metadata"])
def metadata() -> dict[str, Any]:
    """Return model hyperparameters, threshold economics, and schema specification."""
    try:
        _, meta = load_production_model()
        return {
            "model_metadata": meta,
            "feature_columns": ALL_FEATURE_COLS,
            "business_defaults": {
                "clv_retained": CLV_RETAINED,
                "offer_cost": OFFER_COST,
                "save_rate": OFFER_SUCCESS_RATE,
            },
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/predict", tags=["Inference"])
def predict_single_customer(customer: CustomerInput) -> dict[str, Any]:
    """Score a single customer profile and return risk tier with top SHAP drivers."""
    try:
        return predict_one(customer)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@app.post("/predict/recourse", tags=["Prescriptive Analytics"])
def predict_counterfactual_recourse(payload: RecourseRequest) -> dict[str, Any]:
    """Compute actionable counterfactual retention interventions and ROI for an individual account."""
    try:
        return compute_actionable_recourse(
            customer=payload.customer,
            clv=payload.clv,
            save_rate=payload.save_rate,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@app.post("/predict/batch", tags=["Inference"])
def predict_batch_customers(customers: list[CustomerInput]) -> list[dict[str, Any]]:
    """Score a batch of customer profiles."""
    try:
        import pandas as pd

        records = [c.model_dump() for c in customers]
        df = pd.DataFrame(records)
        scored_df = predict_batch(df)
        return scored_df.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

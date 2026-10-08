"""FastAPI microservice for real-time customer churn inference."""

from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from churn.predict import CustomerInput, load_production_model, predict_batch, predict_one

app = FastAPI(
    title="Bank Customer Churn Prediction API",
    description="REST API serving real-time churn risk probabilities, tiers, and SHAP drivers.",
    version="1.0.0",
)


class HealthResponse(BaseModel):
    """Health check response schema."""

    status: str
    model_name: str
    decision_threshold: float
    training_timestamp: str


@app.get("/health", response_model=HealthResponse)
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


@app.post("/predict")
def predict_single_customer(customer: CustomerInput) -> dict[str, Any]:
    """Score a single customer profile and return risk tier with top SHAP drivers."""
    try:
        return predict_one(customer)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@app.post("/predict/batch")
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

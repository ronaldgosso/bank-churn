# Quickstart Guide

This guide will get you up and running with the **Bank Customer Churn Prediction and Decision Support System** in under 3 minutes.

[![Deploy to Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/deploy?repository=ronaldgosso/bank-churn&branch=main&mainModule=app/Home.py)
*(Prefer 1-Click Cloud Deploy? Click the badge above or see the [Deployment Guide](DEPLOYMENT.md)).*

---

## ⚡ 1. Prerequisites

- **Python:** 3.11, 3.12, 3.13, or 3.14
- **Git:** Installed on your system
- **Virtual Environment:** Recommended (`venv`)

---

## 🚀 2. Fast Setup (3 Commands)

### 1. Clone the repository
```bash
git clone https://github.com/ronaldgosso/bank-churn.git
cd bank-churn
```

### 2. Create and activate a virtual environment
**On Windows (PowerShell):**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```
*(If PowerShell blocks script execution, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned` first).*

**On Linux / macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies and the local package
```bash
python -m pip install --upgrade pip
pip install -r requirements-dev.txt
pip install -e .
```

---

## 🖥️ 3. Launch Applications

### Option A: Interactive Streamlit Decision App
Launch the multi-page dashboard to score customers, explore portfolios, and simulate retention campaigns:

```bash
streamlit run app/Home.py
```
👉 Open your browser at **http://localhost:8501**

**What you can do:**
- **Customer Risk Scorer:** Input or load customer profiles, view calibrated churn probability, risk tier badge, top 5 SHAP factors, and individual decision waterfall.
- **Portfolio Explorer:** Upload batch CSV files, filter by risk cohort, and export targeted retention contact lists.
- **Business Simulator:** Adjust CLV ($500), Offer Cost ($50), and Save Rate (30%) to simulate campaign profit and examine cumulative decile gain curves.
- **Model Insights:** Inspect cross-validation benchmarks, ROC/PR curves, reliability diagrams, and fairness parity tables.

---

### Option B: FastAPI REST Microservice
Serve high-throughput real-time predictions via REST API:

```bash
uvicorn churn.api:app --reload --port 8000
```
👉 Interactive API documentation (Swagger UI): **http://localhost:8000/docs**

#### Test with a Single Prediction (cURL)
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
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
```

**Expected JSON Response:**
```json
{
  "probability": 0.9757,
  "risk_tier": "High",
  "contact": true,
  "top_drivers": [
    "Age (48): increases churn risk (+0.95 log-odds)",
    "NumOfProducts (3): increases churn risk (+0.82 log-odds)",
    "Geography_Germany: increases churn risk (+0.54 log-odds)",
    "IsActiveMember (0): increases churn risk (+0.46 log-odds)",
    "Balance ($85,000): increases churn risk (+0.28 log-odds)"
  ]
}
```

#### Test in Python
```python
import requests

payload = {
    "CreditScore": 720,
    "Geography": "France",
    "Gender": "Male",
    "Age": 33,
    "Tenure": 6,
    "Balance": 35000.0,
    "NumOfProducts": 2,
    "HasCrCard": 1,
    "IsActiveMember": 1,
    "EstimatedSalary": 50000.0,
}

response = requests.post("http://localhost:8000/predict", json=payload)
print(response.json())
```

---

## 🔄 4. Reproduce Model Training from Scratch

To rebuild features, tune hyperparameters, calibrate probabilities, and re-export `models/churn_model.joblib`:

```bash
python -m churn.train --config default
```

For quick local testing without Optuna search:
```bash
python -m churn.train --config default --fast
```

---

## 🧪 5. Testing & Code Quality

Run the complete automated test suite and linter:

```bash
# Run Ruff lint check
ruff check .

# Run all 68 unit & integration tests
pytest -v
```

---

## 🐳 6. Running with Docker

Build and run the containerized FastAPI service:

```bash
# Build the Docker image
docker build -t bank-churn:latest .

# Run container on port 8000
docker run -p 8000:8000 bank-churn:latest
```

Test container health:
```bash
curl http://localhost:8000/health
```

---

## ❓ 7. Troubleshooting

| Issue | Solution |
| :--- | :--- |
| **PowerShell script execution disabled** | Run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned` before running `.venv\Scripts\Activate.ps1`. |
| **Port 8501 or 8000 already in use** | Specify a custom port: `streamlit run app/Home.py --server.port 8502` or `uvicorn churn.api:app --port 8001`. |
| **ImportError: No module named 'churn'** | Ensure the package is installed in development mode: `pip install -e .`. |
| **Missing test data in fresh clone** | The test suite and Streamlit app automatically use embedded mock customer archetypes if raw CSV files are not downloaded. Run `python -m churn.data` to fetch/generate data. |

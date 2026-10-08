# Bank Customer Churn Prediction & Decision Support System

[![CI - Lint & Tests](https://github.com/ronaldgosso/bank-churn/actions/workflows/ci.yml/badge.svg)](https://github.com/ronaldgosso/bank-churn/actions/workflows/ci.yml)
[![Deploy to Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/deploy?repository=ronaldgosso/bank-churn&branch=main&mainModule=app/Home.py)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Open in Colab (EDA)](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ronaldgosso/bank-churn/blob/main/notebooks/01_eda.ipynb)
[![Open in Colab (Modeling)](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ronaldgosso/bank-churn/blob/main/notebooks/02_modeling.ipynb)
[![Open in Colab (Evaluation)](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ronaldgosso/bank-churn/blob/main/notebooks/03_evaluation_explainability.ipynb)

> **An end-to-end machine learning system that predicts retail bank customer churn, optimizes intervention thresholds for maximum business profit, explains individual risk drivers via SHAP, and delivers decision support via an interactive Streamlit application and FastAPI service.**

---

## 🎯 Executive Summary & Problem Statement

**The Business Problem:**  
Retail banks face an average annual customer attrition rate of 20%. Acquiring a replacement customer costs 5x to 7x more than retaining an existing relationship. However, customer relationship teams operate under strict operational budgets and cannot offer expensive incentives ($50 retention packages) to every account. 

Default machine learning models typically use an arbitrary `0.5` classification threshold, which misses **over 45% of genuine churners**.

**The Solution:**  
This project implements a leak-free, cost-sensitive machine learning architecture:
1. **Calibrated Machine Learning:** Tuned XGBoost with domain feature engineering, wrapped in 5-fold Isotonic Probability Calibration (`CalibratedClassifierCV`) achieving a **0.8682 ROC-AUC** and **0.7174 PR-AUC** (3.5x over baseline).
2. **Cost-Sensitive Decision Optimization:** The intervention threshold is mathematically derived from unit retention economics ($\text{CLV} = \$500$, $\text{Offer Cost} = \$50$, $\text{Offer Success Rate} = 30\%$), yielding an optimal decision boundary of **$t = 0.3333$**.
3. **Actionable Decision Support:** A 5-page Streamlit web application and FastAPI REST microservice that translates raw churn probabilities into concrete financial actions, decile gain curves, and customer-level SHAP decision waterfalls.

---

## 📊 Key Results (Holdout Test Set)

*Evaluated strictly once on the 2,000-customer holdout test partition (`reports/metrics.json`):*

| Metric | Business Target | Verified Holdout Result | Performance Summary |
| :--- | :--- | :--- | :--- |
| **ROC-AUC** | $\ge 0.8500$ | **0.8682** | ✅ Exceeded target |
| **PR-AUC (Average Precision)** | > Baseline (0.2035) | **0.7174** | ✅ **3.5x over random baseline** |
| **Top 10% Decile Lift** | $\ge 3.00\times$ | **4.20x** | ✅ Captures **42.0%** of all churners in top decile |
| **Top 20% Quintile Lift** | — | **3.10x** | ✅ Captures **61.9%** of all churners in top 20% |
| **Probability Calibration** | Brier $< 0.1000$ | **0.0994** | ✅ Well calibrated (Isotonic regression) |
| **Recall (@ optimal $t=0.3333$)** | Maximize capture | **61.18%** | Captures 249 of 407 churners |
| **Precision (@ optimal $t=0.3333$)** | High intervention ROI | **63.85%** | Minimal wasted campaign spend |
| **F1 Score** | Balanced trade-off | **0.6248** | Optimal balance |
| **Holdout Net Profit** | Profitable campaign | **$17,850.00** | Net return on 2,000 test cohort |

> **Key Business Takeaway:** *Contacting the top 20% of customers captures 62.3% of churners (lift 3.11x) and yields an estimated campaign profit of $72,250.00 across the full portfolio.*

---

## 🏗️ System Architecture

![arch](./public/Customer%20Churn%20Prediction%20Pipeline.png)

---

## 📂 Repository Structure

```
bank-churn/
├── .github/workflows/ci.yml       # GitHub Actions CI (Ruff linting & Pytest suite)
├── .streamlit/config.toml         # Streamlit corporate theme & server settings
├── app/                           # Multi-page Streamlit application
│   ├── Home.py                    # Executive dashboard & headline KPIs
│   ├── utils.py                   # Caching (@st.cache_resource, @st.cache_data) & styles
│   └── pages/
│       ├── 1_Customer_Risk_Scorer.py   # Single account scoring, gauge, SHAP waterfall
│       ├── 2_Portfolio_Explorer.py     # Batch CSV upload, filters, campaign export
│       ├── 3_Business_Simulator.py     # ROI simulator, profit curve, decile gain chart
│       ├── 4_Model_Insights.py         # CV benchmarks, ROC/PR curves, reliability diagram
│       └── 5_About.py                  # Model card summary, tech stack, governance
├── data/
│   ├── raw/                       # Raw source data (gitignored)
│   └── processed/                 # train.csv (8,000 rows) & test.csv (2,000 rows)
├── models/                        # Serialized artifacts
│   ├── churn_model.joblib         # Calibrated XGBoost production pipeline
│   └── metadata.json              # Model architecture, hyperparameters, hash, timestamp
├── notebooks/                     # Colab-executable narrative notebooks
│   ├── 01_eda.ipynb               # 10 Exploratory figures & statistical tests
│   ├── 02_modeling.ipynb          # Benchmark, imbalance comparison, Optuna tuning
│   └── 03_evaluation_explainability.ipynb # Final holdout audit, SHAP, and fairness
├── reports/
│   ├── figures/                   # 25 exported high-resolution PNG charts
│   ├── business_summary.json      # Threshold and campaign optimization data
│   ├── data_summary.json          # Dataset partition properties
│   ├── fairness_summary.json      # Disparity and group metric audit
│   ├── metrics.json               # Final single-touch holdout metrics
│   └── model_comparison.csv       # 5-fold CV comparison across all models
├── src/churn/                     # Core business & machine learning library
│   ├── __init__.py
│   ├── api.py                     # FastAPI REST service (/health, /predict, /predict/batch)
│   ├── business.py                # Profit calculation, threshold optimization, gain tables
│   ├── config.py                  # Project paths, column definitions, unit economics
│   ├── data.py                    # Data loaders and stratified splitting
│   ├── evaluate.py                # Metrics, cross-validation, and diagnostic curves
│   ├── explain.py                 # SHAP TreeExplainer, waterfall, and driver strings
│   ├── fairness.py                # Demographic parity and group performance audit
│   ├── features.py                # FeatureEngineer transformer & ColumnTransformers
│   ├── predict.py                 # Pydantic CustomerInput, single & batch inference
│   ├── train.py                   # Training pipeline & CLI entrypoint
│   └── validation.py              # Pandera DataFrame schemas
├── tests/                         # Pytest test suite (68 unit & integration tests)
├── CHANGELOG.md                   # Chronological project version history & phase changes
├── DEPLOYMENT.md                  # 1-Click Streamlit Cloud deploy, CI/CD flow & keep-alive automation
├── Dockerfile                     # Multi-stage production container specification
├── MODEL_CARD.md                  # Comprehensive model card & regulatory documentation
├── PROJECT_SPEC.md                # Master engineering specification
├── QUICKSTART.md                  # 3-minute fast-start developer & user guide
├── pyproject.toml                 # Package configuration, Ruff linting, & Pytest settings
├── requirements.txt               # Pinned runtime dependencies
└── requirements-dev.txt           # Development and testing dependencies
```

---

## ☁️ Live Streamlit Deployment & Automation

[![Deploy to Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/deploy?repository=ronaldgosso/bank-churn&branch=main&mainModule=app/Home.py)

- **1-Click Cloud Deploy:** Click the badge above or use the [Direct Deployment Link](https://share.streamlit.io/deploy?repository=ronaldgosso/bank-churn&branch=main&mainModule=app/Home.py) to launch on Streamlit Community Cloud with pre-filled repository parameters.
- **Continuous Deployment:** Any commit merged into `main` automatically triggers a zero-downtime hot reload on Streamlit Cloud.
- **Automated Health Probing:** The [.github/workflows/streamlit-smoke-test.yml](.github/workflows/streamlit-smoke-test.yml) workflow pre-tests cold start and probes `/_stcore/health` on every pull request.
- **Scheduled Keep-Alive:** The [.github/workflows/streamlit-keepalive.yml](.github/workflows/streamlit-keepalive.yml) workflow pings the live URL to prevent free-tier inactive sleep timeouts.
- **Detailed Guide:** See [DEPLOYMENT.md](DEPLOYMENT.md) for full step-by-step instructions, secrets setup, and container options.

---

## 🚀 How to Run Locally

> 💡 **Looking for a 3-minute setup?** Check out the dedicated [Quickstart Guide](QUICKSTART.md) for quick-copy commands, cURL samples, and troubleshooting tips.

### 1. Environment Setup

```bash
# Clone the repository
git clone https://github.com/ronaldgosso/bank-churn.git
cd bank-churn

# Create and activate virtual environment
python -m venv .venv
# On Windows:
.venv\Scripts\Activate.ps1
# On Linux/macOS:
source .venv/bin/activate

# Install dependencies and local churn package
pip install -r requirements-dev.txt
pip install -e .
```

### 2. Launch the Streamlit Decision Support App

```bash
streamlit run app/Home.py
```
*Opens locally at `http://localhost:8501`.*

### 3. Launch the FastAPI Microservice

```bash
uvicorn churn.api:app --reload --port 8000
```
*Interactive Swagger API documentation available at `http://localhost:8000/docs`.*

**Example API Request:**
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

### 4. Reproduce Model Training from Scratch

```bash
# Full CLI pipeline (split -> feature engineering -> tuning -> calibration -> threshold)
python -m churn.train --config default
```

### 5. Run Quality Assurance Suite

```bash
# Code formatting and lint inspection
ruff check .

# Execute full unit & integration test suite (68 tests)
pytest -v
```

### 6. Run via Docker

```bash
# Build production Docker container
docker build -t bank-churn-api:latest .

# Run container on port 8000
docker run -p 8000:8000 bank-churn-api:latest
```

---

## 🛠️ Technology Stack

| Domain | Tools & Libraries |
| :--- | :--- |
| **Machine Learning** | XGBoost, LightGBM, Scikit-learn, Imbalanced-learn |
| **Explainability** | SHAP (`TreeExplainer`, Beeswarm, Waterfall plots) |
| **Optimization & Tracking** | Optuna (TPE Sampler), MLflow Tracking |
| **Data Quality & Validation** | Pandera (Schema enforcement), Pydantic v2 |
| **Web & Serving** | Streamlit, FastAPI, Uvicorn, Docker |
| **Visualization** | Plotly Graph Objects / Express, Matplotlib, Seaborn |
| **Engineering & CI** | Ruff, Pytest, GitHub Actions, Setuptools (`pyproject.toml`) |

---

## 🔍 Key Findings & SHAP Interpretability

1. **Age / Seniority Shock:** Churn propensity exhibits sharp non-linear acceleration between ages 45 and 65, peaking right before retirement.
2. **Product Saturation Trap:** Customers with 2 products have optimal retention (~7.6% churn rate); holding 3 or 4 products triggers severe attrition (**82% to 100%**), highlighting critical fee fatigue or misaligned cross-selling.
3. **Geographic Dynamics:** German accounts exhibit nearly 2x the baseline churn of French and Spanish accounts, driven by higher balances and competitive switching.
4. **Digital Engagement Defense:** Active membership (`IsActiveMember=1`) cuts churn risk in half across all customer demographics.

---

## ⚖️ Governance & Limitations

- **Single Snapshot:** The dataset captures a cross-sectional snapshot without time-series transaction velocity.
- **Parametric Economics:** Business simulator assumes uniform CLV ($500) and offer cost ($50). Live deployments should integrate customer-specific dynamic CLV.
- **Fairness Audit:** Measured selection rate ratios and recall gaps between protected demographic groups (documented in [MODEL_CARD.md](MODEL_CARD.md)).

---

## 💼 Portfolio Delivery Assets

### 📄 Resume Bullet Points
- *Architected an end-to-end retail bank customer churn prediction system using calibrated XGBoost, achieving **0.8682 ROC-AUC** and **0.7174 PR-AUC** (3.5x over baseline) on a strict holdout test partition.*
- *Implemented cost-sensitive decision optimization ($\text{CLV}=\$500$, intervention cost $=\$50$), mathematically determining an optimal decision boundary ($t=0.333$) that captures **62.3% of churners** within the top 20% of accounts (3.11x lift).*
- *Developed a 5-page Streamlit decision support platform and FastAPI microservice featuring real-time SHAP waterfall attributions, interactive ROI simulation, and portfolio batch scoring.*
- *Enforced software engineering best practices with clean architecture, Pandera schema validation, Pydantic type safety, 68 Pytest unit tests, and GitHub Actions CI.*

### 📱 LinkedIn Announcement Post
```text
I just launched an end-to-end Machine Learning & Decision Support platform for Bank Customer Churn! 🏦🚀

Most churn projects stop at predicting probabilities with default 0.5 thresholds—which often leaves significant revenue on the table. In retail banking, retention is fundamentally an economic trade-off: keeping an account is far cheaper than acquiring a new one, but retention budgets are finite.

Here is how I designed the system:
1️⃣ Leak-Free ML Engine: Benchmarked 5 model architectures, tuned XGBoost via Optuna, and wrapped the winner in 5-fold Isotonic Probability Calibration (0.8682 ROC-AUC, 0.7174 PR-AUC).
2️⃣ Cost-Sensitive Decision Theory: Derived an optimal threshold (t = 0.333) based on unit economics (CLV $500, offer cost $50, save rate 30%). Contacting the top 20% of at-risk customers captures 62.3% of churners (3.11x lift).
3️⃣ Explainability & Trust: Integrated SHAP TreeExplainer to produce plain-English customer drivers and decision waterfall charts for relationship managers.
4️⃣ Production Serving: Shipped a 5-page interactive Streamlit decision dashboard, FastAPI endpoints, Docker container, and 68 automated Pytest tests in CI.

Check out the full repository and model card:
🔗 GitHub: https://github.com/ronaldgosso/bank-churn

#MachineLearning #DataScience #XGBoost #Streamlit #FinTech #Python #MLOps #SHAP
```

---

## 👤 Author & Contact

- **Author:** Ronald Gosso
- **GitHub:** [@ronaldgosso](https://github.com/ronaldgosso)
- **Repository:** [bank-churn](https://github.com/ronaldgosso/bank-churn)
- **License:** MIT License

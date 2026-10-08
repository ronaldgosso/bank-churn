# Changelog

All notable changes to the **Bank Customer Churn Prediction & Decision Support System** will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.1.0] - 2026-10-08

### Added
- **Prescriptive Counterfactual Recourse Engine (`src/churn/predict.py`):**
  - Added `compute_actionable_recourse` evaluating realistic banking levers (Digital Activation, Product Consolidation/Cross-Sell, Card Incentive, Multi-Lever Retention Bundle).
  - Calculates probability delta ($\Delta p$), risk tier transitions, expected net campaign ROI, and generates bespoke Relationship Manager Playbooks with personalized talking points.
- **Interactive Recourse & What-If Sandbox (`app/pages/1_Customer_Risk_Scorer.py`):**
  - Scenario comparison table, Plotly probability delta comparison bar chart, optimal intervention card, and live interactive slider sandbox.
- **2x2 Strategic Value-Risk Matrix & Balance Runoff (`app/pages/2_Portfolio_Explorer.py`):**
  - Calculated Capital-at-Risk KPIs: Total High-Risk Balance ($) and Expected Annual Portfolio Balance Runoff ($\sum \text{Balance}_i \times p_i$).
  - Added 2x2 Value-vs-Risk quadrant scatter plot (VIP Save Priority, Automated Digital Save, Relationship Deepening, Standard Service).
  - Filter and export retention campaigns by strategic priority.
- **Developer Hub & REST API Console (`app/pages/6_Developer_API.py`):**
  - Dedicated interactive documentation and payload simulator for FastAPI endpoints (`/health`, `/meta`, `/predict`, `/predict/recourse`, `/predict/batch`).
  - Architecture explainer detailing Streamlit Community Cloud (Frontend UI) vs. Docker FastAPI container.
  - Multi-language ready-to-use code snippets for `cURL`, Python (`requests`), and TypeScript/JavaScript (`fetch`).
  - Container deployment guide for AWS ECS, Google Cloud Run, Render, and Railway.
- **FastAPI Microservice Upgrades (`src/churn/api.py`):**
  - Added `POST /predict/recourse` endpoint with `RecourseRequest` schema.
  - Added `GET /meta` endpoint returning feature specifications and economic assumptions.
- **Production Dependency & Keep-Alive Workflow Hardening:**
  - Added `pandera==0.34.1` to `requirements.txt` ensuring zero-dependency-drift deployment on Streamlit Cloud.
  - Hardened `.github/workflows/streamlit-keepalive.yml` with URL sanitization and dual root/health probes.
  - Expanded unit test suite to 73 tests (`tests/test_predict.py`, `tests/test_app.py`).

---

## [1.0.0] - 2026-10-08

### Added
- **Streamlit Deployment Automation Flow:**
  - Automated 1-Click deployment link to Streamlit Community Cloud (`https://share.streamlit.io/deploy`).
  - GitHub Actions deployment smoke test workflow (`.github/workflows/streamlit-smoke-test.yml`) verifying cold start and probing `/_stcore/health`.
  - GitHub Actions keep-alive scheduled ping workflow (`.github/workflows/streamlit-keepalive.yml`) preventing inactive sleep on free-tier instances.
  - Dedicated deployment architectural guide (`DEPLOYMENT.md`).
  - Added 1-Click deployment badges and direct links to `README.md` and `QUICKSTART.md`.

---

## [0.1.0] - 2026-10-08

### Initial Production Release

#### Added
- **Phase 0: Scaffold & Architecture**
  - Standard Python `src`-layout package structure (`src/churn/`).
  - Configured `pyproject.toml`, pinned `requirements.txt`, and development dependencies in `requirements-dev.txt`.
  - Configured `.gitignore`, `.pre-commit-config.yaml`, and initial GitHub Actions CI workflow.

- **Phase 1: Data Acquisition & Validation**
  - Download and ingestion pipeline for Kaggle Bank Customer Churn dataset (`data/raw/Churn_Modelling.csv`).
  - Pandera schema validation (`src/churn/validation.py`) with strict constraints and type checking.
  - Stratified 80/20 train/test split persisted to `data/processed/train.csv` (8,000 rows, 20.375% churn) and `data/processed/test.csv` (2,000 rows, 20.350% churn).
  - Data summary persisted in `reports/data_summary.json`.

- **Phase 2: Exploratory Data Analysis & Statistical Testing**
  - Top-to-bottom executable narrative notebook `notebooks/01_eda.ipynb` with Google Colab badge.
  - 10 publication-quality charts saved to `reports/figures/eda_01_*.png` through `eda_10_*.png`.
  - Hypothesis testing and verification using Chi-Square tests of independence and Mann-Whitney U rank tests.

- **Phase 3: Domain Feature Engineering & Preprocessing**
  - Scikit-learn custom `FeatureEngineer` transformer in `src/churn/features.py`.
  - Engineered 8 domain features: `BalanceSalaryRatio`, `ZeroBalance`, `ProductsPerTenure`, `TenureByAge`, `CreditScoreByAge`, `ActiveCardHolder`, `HighProducts`, `IsSenior`.
  - Leak-free `ColumnTransformer` pipelines (`build_linear_preprocessor`, `build_tree_preprocessor`, `build_pipeline`).

- **Phase 4: Model Benchmarking, Imbalance Handling & Optuna Tuning**
  - 5-fold Stratified Cross-Validation benchmark across 5 architectures: Dummy, Logistic Regression, Random Forest, LightGBM, and XGBoost.
  - Imbalance mitigation comparison (unweighted, class weighting, SMOTE inside `imblearn` pipeline).
  - Hyperparameter optimization with Optuna (50 trials with TPE Sampler).
  - Isotonic probability calibration (`CalibratedClassifierCV(method='isotonic', cv=5)`).
  - Benchmarks and winning parameters saved in `reports/model_comparison.csv` and `reports/best_params.json`.
  - SQLite-backed MLflow experiment tracking.

- **Phase 5: Cost-Sensitive Thresholding & Business Impact**
  - Decision theory engine implemented in `src/churn/business.py`.
  - Mathematical derivation of optimal decision boundary: $t = \frac{\text{Cost}}{\text{Success Rate} \times \text{CLV}} = 0.3333$.
  - Evaluation of expected and actual retention profit, cumulative gain curves, and decile lift tables.
  - Saved outputs: `reports/decile_gain_table.csv`, `reports/business_summary.json`, and business charts `reports/figures/business_*.png`.

- **Phase 6: Final Test Evaluation, SHAP Attribution & Fairness Audit**
  - Single untouched holdout evaluation on 2,000 test customers, persisting `reports/metrics.json`.
  - Verified holdout results: ROC-AUC = 0.8682, PR-AUC = 0.7174 (3.5x baseline), Top-decile lift = 4.20x, Brier score = 0.0994.
  - SHAP TreeExplainer implementation in `src/churn/explain.py` with beeswarm, feature importance, dependence, and customer waterfall charts.
  - Demographic fairness audit across Gender and Geography in `src/churn/fairness.py` and `reports/fairness_summary.json`.
  - Comprehensive model documentation in `MODEL_CARD.md`.

- **Phase 7: Production Inference Engine, CLI & API**
  - CLI training pipeline in `src/churn/train.py` with reproducible artifact generation (`models/churn_model.joblib` and `models/metadata.json`).
  - Production inference interfaces in `src/churn/predict.py`: `CustomerInput` (Pydantic schema), `predict_one`, and `predict_batch` with risk tier classification.
  - High-performance FastAPI REST API (`src/churn/api.py`) exposing `/health`, `/predict`, and `/predict/batch`.
  - Containerization specification in `Dockerfile`.

- **Phase 8: Multi-Page Streamlit Decision Support Application**
  - Streamlit dashboard in `app/Home.py` with headline KPIs loaded dynamically from `reports/metrics.json`.
  - **Page 1: Customer Risk Scorer:** Real-time scoring, gauge chart, risk tiers, plain-English drivers, and SHAP decision waterfall.
  - **Page 2: Portfolio Explorer:** Batch CSV upload, sample template download, probability distribution histogram, multi-attribute filtering, and campaign CSV export.
  - **Page 3: Business Simulator:** Interactive sliders for CLV, offer cost, and budget constraints; interactive profit curves and cumulative gain charts.
  - **Page 4: Model Insights:** CV benchmark tables, holdout ROC/PR curves, reliability diagrams, and fairness disparity audit tables.
  - **Page 5: About:** Model card governance, tech stack architecture, and reproduction commands.
  - Streamlit caching and theming utilities in `app/utils.py`.

- **Phase 9: Deployment Configuration & Theming**
  - Corporate branding and layout configuration in `.streamlit/config.toml`.
  - Pinned runtime dependencies in `requirements.txt`.

- **Phase 10: Portfolio Packaging & Documentation**
  - Professional `README.md` featuring architecture diagram, holdout metrics table, and local run instructions.
  - Deliverable portfolio assets: LinkedIn announcement post, resume bullet points, and interview talking points.
  - Quickstart guide in `QUICKSTART.md`.

#### Changed
- Updated CI workflow `.github/workflows/ci.yml` to Python 3.12 for modern NumPy and XGBoost compatibility.
- Added PEP 508 environment markers in `requirements.txt` (`numpy==2.4.6; python_version < "3.12"` and `numpy==2.5.3; python_version >= "3.12"`).
- Updated Dockerfile base image to `python:3.12-slim`.
- Updated pipeline architecture diagram reference in `README.md` to `public/Customer Churn Prediction Pipeline.png`.

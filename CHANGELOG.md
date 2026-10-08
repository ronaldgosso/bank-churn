# Changelog

All notable changes to the **Bank Customer Churn Prediction & Decision Support System** will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

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

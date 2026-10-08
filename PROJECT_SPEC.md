# Bank Customer Churn Prediction: Master Project Spec

*Agent-ready roadmap for Antigravity · Windows 11 · Google Colab · Streamlit*

## 1. How to Use This Spec

Save this file as `PROJECT_SPEC.md` in the repo root. It is the single source of truth. Every decision (features, models, metrics, thresholds, file names) is already made below, so the agent never has to guess or go looking for ideas.

Repeat this loop for every phase:

1. Paste: `Read PROJECT_SPEC.md. Implement Phase N exactly as written. Do not start Phase N+1.`
2. Let the agent finish, then check that phase's **Acceptance Criteria** yourself.
3. Run `ruff check .` then `pytest`.
4. Commit: `git commit -m 'phase N: short summary'`.

Why this works with a fast model: small scoped phases, explicit decisions and hard acceptance criteria. Open-ended goals make fast models drift or cut corners.

### Global Rules for the Agent

Paste these into the agent's rules/instructions file.

1. Follow the spec. Do not add libraries that are not in Section 5. If something is ambiguous, choose the option written in the spec; if the spec is silent, ask.
2. Random seed is always `42`.
3. Notebooks explore and narrate. All reusable logic lives in `src/churn/`. Notebooks import from `churn`; they never copy-paste logic.
4. The Streamlit app only displays. It imports from `churn` and contains no modeling logic.
5. Type hints and docstrings on every public function. Code must pass `ruff`.
6. No data leakage: split first, fit every transformer on training data only, inside an sklearn Pipeline.
7. The test set is touched exactly once, in Phase 6.
8. Never invent numbers. Every metric in a README, report or app comes from `reports/metrics.json`, written by code.
9. Windows-safe: use `pathlib`, no hard-coded backslashes, no bash-only commands in docs.
10. Every phase ends with: tests passing, a list of files changed, and a short summary of decisions made. Then stop.

## 2. Problem and Solution

**Business problem.** A retail bank loses customers every year. Winning a new customer usually costs more than keeping an existing one, but a retention team has a limited budget and cannot contact everyone.

**Objective.** Predict the probability that each customer will leave, rank customers by risk, and recommend who to contact so that expected retention profit is maximized.

**Solution.** An end-to-end system:

- A validated, leak-free ML pipeline that scores churn risk.
- A threshold chosen from business cost, not from the default 0.5.
- Explanations (SHAP) so a manager knows why a customer is flagged.
- A fairness and calibration review.
- A Streamlit decision-support app, tested, packaged and deployed.

**Success targets** (report honestly; never tune on the test set):

| Metric | Target |
| --- | --- |
| ROC-AUC on holdout | 0.85 or higher |
| PR-AUC | reported with the baseline (churn rate) for context |
| Top-decile lift | 3.0x or higher |
| Calibration | Brier score reported, calibration curve shown |

### Who Sees What

| Audience | Where they look | What it proves |
| --- | --- | --- |
| Recruiter / hiring manager | README, live app, demo GIF | Business thinking, finished product |
| Data scientist | `notebooks/`, `MODEL_CARD.md`, `reports/` | Rigorous EDA, leak-free validation, calibration, explainability, fairness |
| Software engineer | `src/churn/`, `tests/`, CI, `pyproject.toml` | Clean architecture, typed code, tests, reproducibility, deployment |
| Retention manager (end user) | Streamlit app | Decision support, not just a model |

## 3. Repository Structure

```
bank-churn/
├── data/
│   ├── raw/                 # Kaggle CSV (gitignored)
│   └── processed/           # train.csv, test.csv
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_modeling.ipynb
│   └── 03_evaluation_explainability.ipynb
├── src/churn/
│   ├── __init__.py
│   ├── config.py            # constants, column lists, cost assumptions
│   ├── data.py              # load, split
│   ├── validation.py        # pandera schema
│   ├── features.py          # FeatureEngineer transformer, preprocessors
│   ├── train.py             # CV, tuning, final fit, CLI
│   ├── evaluate.py          # metrics, plots
│   ├── business.py          # expected profit, thresholds, lift/gain
│   ├── explain.py           # SHAP helpers
│   ├── fairness.py          # group metrics
│   └── predict.py           # CustomerInput, predict_one, predict_batch
├── app/
│   ├── Home.py
│   └── pages/               # numbered Streamlit pages
├── tests/
├── models/                  # churn_model.joblib, metadata.json
├── reports/figures/         # saved charts
├── .github/workflows/ci.yml
├── pyproject.toml
├── requirements.txt         # runtime only (used by deployment)
├── requirements-dev.txt     # training, tuning, notebooks, tests
├── README.md
├── MODEL_CARD.md
└── PROJECT_SPEC.md
```

## 4. Environments and Data

| Task | Where | Why |
| --- | --- | --- |
| EDA, model comparison, tuning, SHAP | Google Colab | Zero setup, shareable via Open in Colab badge |
| `src/`, tests, app, CI, git | Local Windows 11 + VS Code / Antigravity | Real software engineering workflow |
| Dataset download, fallback runtime | Kaggle | Source of the data; Kaggle Notebooks work if Colab limits you |

The dataset is tiny (10k rows), so Colab's free CPU is enough. No GPU needed.

### Datasets

- **Primary (use for all phases):** the Kaggle dataset containing `Churn_Modelling.csv` (10,000 rows, 14 columns, target `Exited`). Search Kaggle for **Churn Modelling** or **Bank Customer Churn Prediction** and copy the `owner/slug` from the dataset URL.
- **Optional scale test:** Kaggle Playground Series S4E1, **Binary Classification with a Bank Churn Dataset** (about 165k training rows). Use it only after Phase 9 to show the pipeline scales.
- **Warning:** a Kaggle variant named *Bank Customer Churn* with `Customer-Churn-Records.csv` includes a `Complain` column that almost perfectly reveals the target. Do not use it, or drop `Complain` and say why. Spotting leakage is a strong interview story.

### Download on Windows (PowerShell)

```
pip install kaggle
# Kaggle > Settings > Create New Token, then save kaggle.json to %USERPROFILE%/.kaggle/
kaggle datasets download -d OWNER/SLUG -p data/raw --unzip
```

### Download in Colab

```python
from google.colab import files
files.upload()  # choose kaggle.json
!mkdir -p ~/.kaggle && cp kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
!kaggle datasets download -d OWNER/SLUG -p data/raw --unzip
```

Fallback: download the CSV manually from Kaggle, put it in Google Drive, and mount Drive in the notebook.

## 5. Locked Tech Stack

Python 3.11. No other libraries unless the spec says so.

| Purpose | Library |
| --- | --- |
| Data | pandas, numpy |
| Validation | pandera |
| Modeling | scikit-learn, xgboost, lightgbm, imbalanced-learn |
| Tuning and tracking | optuna, mlflow |
| Explainability | shap |
| Plots | matplotlib, seaborn (reports), plotly (app) |
| App | streamlit, pydantic |
| Optional API | fastapi, uvicorn |
| Artifacts | joblib |
| Quality | pytest, ruff, pre-commit |

## 6. Phases

### Phase 0: Scaffold (Day 1)

**Goal:** a clean, reproducible repo.

**Tasks**

- Create the structure in Section 3 with empty modules and docstrings.
- `pyproject.toml` with package name `churn`, `src` layout, installable via `pip install -e .`
- `requirements.txt` (runtime) and `requirements-dev.txt` with pinned versions.
- `.gitignore` (data/raw, .venv, mlruns, **pycache**), `ruff` config, `pre-commit` with ruff.
- One placeholder test so `pytest` runs.

**Windows setup**

```
python -m venv .venv
.venv/Scripts/Activate.ps1
pip install -r requirements-dev.txt
pip install -e .
```

**Acceptance:** fresh clone to green `pytest` in under 5 minutes; `import churn` works.

### Phase 1: Data Acquisition and Validation

**Goal:** trusted data and a locked test set.

**Tasks**

- `config.py`: `SEED=42`, `TARGET='Exited'`, `DROP_COLS=['RowNumber','CustomerId','Surname']`, `NUMERIC=['CreditScore','Age','Tenure','Balance','NumOfProducts','EstimatedSalary']`, `BINARY=['HasCrCard','IsActiveMember']`, `CATEGORICAL=['Geography','Gender']`.
- `data.py`: `load_raw()`, `make_split()` doing a stratified 80/20 split, saving `data/processed/train.csv` and `test.csv`.
- `validation.py`: pandera schema. CreditScore 300 to 900, Age 18 to 100, Tenure 0 to 10, NumOfProducts 1 to 4, Geography in France/Germany/Spain, Gender in Female/Male, target in 0/1, no nulls.
- Tests: schema accepts the real data and rejects a corrupted row; split is stratified; train and test do not overlap.

**Acceptance:** `reports/data_summary.json` lists row counts, churn rate per split and duplicate count. The test file is never loaded again until Phase 6.

### Phase 2: Exploratory Data Analysis (`01_eda.ipynb`, Colab)

**Goal:** insights, not just plots. Every chart ends with a one-line **So what?** in markdown.

Use the training split only.

**Required analysis**

| # | Question | Method |
| --- | --- | --- |
| 1 | What is the churn rate and how imbalanced is it? | Bar chart; note why accuracy is misleading |
| 2 | How do Age, Balance, CreditScore, Tenure and Salary distribute, split by churn? | Histograms / KDE, boxplots |
| 3 | Churn rate by Geography and Gender | Bar charts with counts |
| 4 | Churn rate by NumOfProducts | Bar chart (check 3 and 4 products carefully) |
| 5 | Churn rate by IsActiveMember and HasCrCard | Grouped bars |
| 6 | Churn by age band (18-29, 30-39, 40-49, 50-59, 60+) | Line / bar |
| 7 | Customers with zero balance: share and churn rate | Compare to non-zero balance |
| 8 | Geography x Gender interaction | Heatmap of churn rate |
| 9 | Correlations among numeric features | Heatmap |
| 10 | Are differences statistically significant? | Chi-square plus Cramér's V for categoricals; Mann-Whitney U for numerics |

**Patterns to verify, not assume:** older customers, Germany, inactive members, and customers with 3 or more products appear to churn more. State clearly whether the data confirms each one.

End the notebook with a **Key Findings** section: five bullets a manager can read in 30 seconds, and a list of modeling implications (class imbalance, nonlinear age effect, feature ideas).

**Acceptance:** notebook runs top to bottom after Restart and run all; figures saved to `reports/figures/eda_*.png`; Open in Colab badge works.

### Phase 3: Preprocessing and Feature Engineering

**Goal:** one reusable, leak-free feature pipeline.

**Tasks**

- In `features.py`, build `FeatureEngineer(BaseEstimator, TransformerMixin)` that adds:

| Feature | Formula |
| --- | --- |
| `BalanceSalaryRatio` | Balance / (EstimatedSalary + 1) |
| `ZeroBalance` | 1 if Balance == 0 else 0 |
| `ProductsPerTenure` | NumOfProducts / (Tenure + 1) |
| `TenureByAge` | Tenure / Age |
| `CreditScoreByAge` | CreditScore / Age |
| `ActiveCardHolder` | HasCrCard \* IsActiveMember |
| `HighProducts` | 1 if NumOfProducts >= 3 else 0 |
| `IsSenior` | 1 if Age >= 50 else 0 |

- Two preprocessors via `ColumnTransformer`: `build_linear_preprocessor()` (one-hot plus StandardScaler) and `build_tree_preprocessor()` (one-hot only, no scaling).
- Every transformer is fitted inside a Pipeline.

**Tests:** feature values correct on a hand-made 3-row DataFrame; no NaN or inf produced; fitting on train does not touch test; output column order is stable.

**Acceptance:** `build_pipeline(model_name)` returns a full Pipeline for any supported model.

### Phase 4: Modeling, Tuning and Selection (`02_modeling.ipynb`, Colab)

**Goal:** a defensible model choice.

**Models:** `DummyClassifier` (most frequent), Logistic Regression, Random Forest, XGBoost, LightGBM.

**Validation:** `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` on the training split only.

**Metrics per fold (mean and std):** ROC-AUC, PR-AUC (average precision), Recall, Precision, F1 at 0.5, Brier score.

**Experiments, in order**

1. Baseline comparison of all five models with default settings and class weighting (`class_weight='balanced'` or `scale_pos_weight`).
2. Imbalance strategy comparison for the best two models: none vs class weights vs SMOTE (SMOTE only inside an `imblearn` Pipeline so it is applied to training folds only).
3. Feature ablation: best model with and without engineered features. Report the delta.
4. Tuning: Optuna, TPE sampler with seed 42, 50 trials, objective = mean CV PR-AUC, for the top two models.
5. Calibration: wrap the winner with `CalibratedClassifierCV(method='isotonic', cv=5)` and compare Brier score before and after.

**Selection rule:** highest tuned CV PR-AUC; if within one standard deviation, prefer the simpler or better-calibrated model. Write the reasoning in one paragraph.

**Tracking:** log every run to local MLflow (params, metrics, artifact). Keep a screenshot of the MLflow table in `reports/figures/`.

**Outputs:** `reports/model_comparison.csv`, `reports/best_params.json`, and the implementation in `train.py` (CLI in Phase 7).

**Acceptance:** results table in the notebook; the notebook imports from `churn`, with no duplicated pipeline code.

### Phase 5: Threshold and Business Impact

**Goal:** turn probabilities into decisions and money.

**Assumptions** (in `config.py`, editable in the app, labeled clearly as assumptions):

```python
CLV_RETAINED = 500       # value of keeping a customer
OFFER_COST = 50          # cost of a retention offer
OFFER_SUCCESS_RATE = 0.30
```

**Logic**

- Expected profit of contacting customer i: `p_i * OFFER_SUCCESS_RATE * CLV_RETAINED - OFFER_COST`.
- Contact when positive, so the optimal threshold is `OFFER_COST / (OFFER_SUCCESS_RATE * CLV_RETAINED)` = 0.333 with the defaults. Show this derivation in the notebook.
- Compare against the default 0.5 and against the F1-optimal threshold.
- Choose and validate the threshold using out-of-fold predictions on the training split only.
- Budget mode: contact only the top k% (10, 20, 30) by risk.

**Functions in `business.py`:** `expected_profit()`, `optimal_threshold()`, `gain_table()`, `lift_at_k()`.

**Charts:** cumulative gain curve, lift chart, profit vs threshold curve, confusion matrix at the chosen threshold.

**Acceptance:** a sentence such as *Contacting the top 20% of customers captures X% of churners and yields an estimated profit of Y*, where X and Y are computed, not typed.

### Phase 6: Final Test, Explainability and Fairness (`03_evaluation_explainability.ipynb`)

**Goal:** the honest final exam.

**Tasks**

1. **Final test evaluation, once.** Load `test.csv`, score with the frozen model and threshold. Write `reports/metrics.json` with ROC-AUC, PR-AUC, recall, precision, F1, Brier, lift at 10/20%, estimated profit.
2. **Calibration curve** (reliability diagram) on test predictions.
3. **SHAP:** beeswarm summary, bar importance, dependence plots for Age, NumOfProducts and Balance, and waterfall plots for three customers (one high risk, one borderline, one low risk).
4. **Error analysis:** profile the false negatives (missed churners). What do they have in common?
5. **Fairness check:** by Gender and Geography report AUC, recall, precision and selection rate; compute the ratio of selection rates and recall gap between groups. Document findings and limits. Do not claim the model is fair, only what was measured.
6. **Limitations:** single snapshot, no time dimension, small dataset, assumed costs.

**Acceptance:** all figures saved in `reports/figures/`; `MODEL_CARD.md` drafted from `metrics.json` (intended use, data, metrics, fairness, limitations).

### Phase 7: Productionize the Code

**Goal:** software-engineering quality.

**Tasks**

- `train.py` CLI: `python -m churn.train --config default` runs split, CV, tuning flag, final fit, calibration, threshold selection, and saves `models/churn_model.joblib` and `models/metadata.json` (library versions, training date, feature list, threshold, metrics, data hash).
- `predict.py`: `CustomerInput` (Pydantic, with the same ranges as the schema), `predict_one(dict) -> dict` returning `probability`, `risk_tier`, `contact` (bool), `top_drivers` (top 5 SHAP features as plain-English strings); `predict_batch(df) -> df`.
- **Risk tiers**, relative to threshold `t`: High if `p >= t`, Medium if `p >= 0.5 * t`, otherwise Low.
- **Tests (pytest):** schema validation, feature engineer correctness, pipeline fit/predict round trip, `predict_one` output keys and probability in \[0, 1\], determinism with the seed, invalid input raises a clear error.
- **CI** `.github/workflows/ci.yml`: on push and PR, set up Python 3.11, install, run `ruff check .` and `pytest`.
- **Optional but recommended:** FastAPI app with `/health` and `/predict`, plus a Dockerfile.

**Acceptance:** `python -m churn.train` rebuilds the model from scratch; CI badge is green.

### Phase 8: Streamlit App

**Goal:** a decision-support tool a retention manager would actually use.

**Rules:** wide layout, sidebar navigation, `st.cache_resource` for the model, `st.cache_data` for data, plotly charts, one consistent color palette (green low, amber medium, red high). The app imports only from `churn`.

**Pages**

| Page | Contents |
| --- | --- |
| `Home.py` | Problem, headline KPIs from `metrics.json`, how to use the app |
| 1 Customer Risk Scorer | Input form (Pydantic-validated), risk gauge, tier, recommended action, top 5 drivers in plain English, SHAP waterfall |
| 2 Portfolio Explorer | Upload CSV for batch scoring, filters, top-N risk list, download scored CSV, button for a sample file; friendly errors on bad files |
| 3 Business Simulator | Sliders for CLV, offer cost, success rate and budget %; shows optimal threshold, expected profit, gain curve |
| 4 Model Insights | Model comparison table, ROC and PR curves, SHAP global importance, calibration curve, fairness table |
| 5 About | Model card summary, data source, limitations, links to GitHub and notebooks |

**Acceptance:** cold start under 5 seconds locally; no unhandled exception for empty, wrong or out-of-range input; every number on screen comes from code or `metrics.json`; run with `streamlit run app/Home.py`.

### Phase 9: Deployment

**Goal:** a live public URL.

**Streamlit Community Cloud (primary)**

1. Push the repo to GitHub (public). Commit `models/churn_model.joblib` if it is small (a few MB is fine).
2. Make `requirements.txt` runtime-only and pinned, and make the `churn` package importable (add `.` as a requirement, verified in a fresh venv). Fallback: insert the `src` path at the top of `Home.py`.
3. On share.streamlit.io, create the app: main file `app/Home.py`, Python 3.11.
4. Smoke test the live URL: every page loads, a sample customer scores, batch upload works.

**Fallback:** Hugging Face Spaces with the Streamlit SDK.

**Notes:** free apps sleep when idle, so mention this in the README. Do not commit secrets or raw data.

**Acceptance:** live URL in the README with a badge; a teammate can open it on a phone.

### Phase 10: Portfolio Packaging

**Goal:** make it land the job.

**README order:** title and one-line value statement, live demo and Colab badges, demo GIF (record with ScreenToGif on Windows), problem and approach, key results table (from `metrics.json`), architecture diagram, repo structure, how to run, tech stack, limitations, author links.

**Also deliver**

- `MODEL_CARD.md` finalized.
- GitHub repo description, topics (`machine-learning`, `churn-prediction`, `streamlit`, `shap`, `mlflow`), pinned on your profile.
- A LinkedIn post: problem, one surprising finding, the business result, live link.
- Resume bullets in this shape: *Built an end-to-end bank churn model (XGBoost, SHAP) with ROC-AUC X and profit-optimized thresholding; deployed a Streamlit decision tool with CI, tests and fairness analysis.*

**Interview talking points to prepare**

1. Why PR-AUC and recall over accuracy.
2. How you prevented leakage (Pipeline, single test touch, Complain column story).
3. Why the threshold is not 0.5 and how cost drives it.
4. What SHAP showed and one business recommendation.
5. What you would do next (time-based features, uplift modeling, monitoring for drift).

## 7. Suggested Timeline

| Days | Phases |
| --- | --- |
| 1 | 0, 1 |
| 2 to 3 | 2 |
| 4 | 3 |
| 5 to 7 | 4 |
| 8 to 9 | 5, 6 |
| 10 to 12 | 7 |
| 13 to 15 | 8 |
| 16 | 9 |
| 17 to 18 | 10 |

## 8. Definition of Done

- [ ] Live Streamlit URL works on desktop and phone.
- [ ] `pytest` and `ruff` pass locally and in CI.
- [ ] `python -m churn.train` reproduces the model from scratch.
- [ ] Test set used once; all README numbers come from `metrics.json`.
- [ ] EDA, modeling and evaluation notebooks run top to bottom with Open in Colab badges.
- [ ] SHAP, calibration, fairness and limitations are documented in `MODEL_CARD.md`.
- [ ] README has a demo GIF, results table and architecture diagram.
- [ ] LinkedIn post and resume bullets written.

# Model Card: Bank Customer Churn Prediction

## 1. Model Details
- **Model Name:** Calibrated XGBoost with Domain Feature Engineering (`xgboost`)
- **Developer:** Ronald Gosso
- **Release Date:** October 2026
- **Model Version:** 1.0.0
- **Model Architecture:** Gradient Boosted Decision Trees (`XGBClassifier`) wrapped in 5-fold Isotonic Probability Calibration (`CalibratedClassifierCV`)
- **Primary Objective:** Probability scoring for proactive customer retention decision support

## 2. Intended Use
- **Primary Use Case:** Prioritizing customer retention interventions and retention budget allocation for relationship managers.
- **Decision Rule:** Contact customer when predicted probability exceeds the business optimal threshold ($t = 0.3333$), derived from retention economics ($CLV = \$500$, Offer Cost = $\$50$, Success Rate = $30\%$).
- **Out-of-Scope Use Cases:** Automated credit denial, account closure, or adverse actions without human review.

## 3. Training & Evaluation Data
- **Source:** Kaggle Churn Modelling dataset (10,000 customers, 14 initial fields).
- **Split Protocol:** Stratified 80/20 train/test split with seed `42`.
  - **Training Split:** 8,000 rows (20.38% churn rate, 1,630 churners). Used exclusively for exploratory analysis, cross-validation, hyperparameter tuning, and threshold selection.
  - **Holdout Test Set:** 2,000 rows (20.35% churn rate, 407 churners). Evaluated strictly once during final audit.

## 4. Performance Metrics (Holdout Test Set)

All metrics below are computed directly from `reports/metrics.json` on the untouched holdout test partition:

| Metric | Target | Result | Status |
| --- | --- | --- | --- |
| **ROC-AUC** | $\ge 0.85$ | **0.8682** | Exceeded |
| **PR-AUC (Average Precision)** | Reported vs baseline (0.2035) | **0.7174** | 3.5x over baseline |
| **Recall (@ $t=0.3333$)** | — | **61.18%** | Captures 249/407 churners |
| **Precision (@ $t=0.3333$)** | — | **63.85%** | Highly actionable |
| **F1 Score** | — | **0.6248** | Balanced |
| **Brier Score Loss** | Reported with calibration | **0.0994** | Well calibrated (<0.10) |
| **Top 10% Decile Lift** | $\ge 3.0\times$ | **4.20x** | Exceeded |
| **Top 20% Quintile Lift** | — | **3.10x** | Captures 61.9% of churners |
| **Campaign Net Profit** | — | **$17,850.00** | Net gain on 2,000 test cohort |

## 5. Explainability (SHAP Insights)
Global feature attribution shows that churn risk is predominantly driven by:
1. **Age / Seniority:** Strong positive correlation with churn risk, accelerating sharply between ages 45–60.
2. **Number of Products:** Holding 2 products acts as a protective retention factor, whereas holding 3 or 4 products dramatically elevates attrition risk.
3. **Geography (Germany):** German accounts exhibit substantially higher baseline churn risk (+0.40 log-odds).
4. **Member Inactivity:** Inactive membership status (`IsActiveMember=0`) approximately doubles churn propensity.
5. **Balance / Salary Interaction:** High balance relative to estimated salary increases churn probability.

## 6. Fairness & Demographic Parity Audit
Audited across Gender and Geography on the holdout test set:

- **Gender Breakdown:**
  - Female Churn Rate: 24.3% | Selection Rate: 28.5% | Recall: 65.8% | Precision: 56.1% | ROC-AUC: 0.8624
  - Male Churn Rate: 17.1% | Selection Rate: 12.3% | Recall: 55.4% | Precision: 76.9% | ROC-AUC: 0.8688
  - *Selection Rate Ratio:* 0.432 (reflects higher empirical underlying churn among females).
  - *Recall Gap:* 10.4% between genders.
- **Geography Breakdown:**
  - Germany: 39.4% Churn | Selection Rate: 39.8% | Recall: 67.2% | Precision: 66.5% | ROC-AUC: 0.8407
  - France: 16.2% Churn | Selection Rate: 11.5% | Recall: 53.7% | Precision: 75.4% | ROC-AUC: 0.8753
  - Spain: 16.7% Churn | Selection Rate: 14.8% | Recall: 64.9% | Precision: 73.1% | ROC-AUC: 0.8732
  - *Selection Rate Ratio:* 0.289 (Germany has 2.4x higher baseline attrition).
  - *Recall Gap:* 13.5% across geographic regions.

*Limitation Note:* The model reflects historical demographic patterns present in the training cohort and should not be used as an autonomous gating mechanism without human retention specialist oversight.

## 7. System Limitations
1. **Cross-Sectional Data:** Lacks temporal balance drawdown velocity (e.g. 30/60/90-day transaction history).
2. **Economic Assumptions:** Fixed CLV (\$500) and incentive cost (\$50); dynamic LTV modeling would further refine profitability.
3. **Regional Sensitivity:** German portfolio requires localized customer relationship investigations.

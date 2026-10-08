# Bank Customer Churn Decision Support Platform
## Streamlit Web Application — Comprehensive Operational Guide & Metric Dictionary (`Streamlit_app.md`)

> **Document Version:** `1.1.0`  
> **Target Audience:** Branch Relationship Managers, Retention Campaign Planners, Retail Banking Risk Officers, Quantitative Analysts, and Software Engineers.  
> **Production Application URL:** [bankcustomerchurning.streamlit.app](https://bankcustomerchurning.streamlit.app/)  
> **Author & Project Lead:** [Ronald Gosso](https://github.com/ronaldgosso)

---

## Table of Contents

1. [Executive Summary & System Vision](#1-executive-summary--system-vision)
2. [Dual-Tier System Architecture (UI vs. REST Microservice)](#2-dual-tier-system-architecture-ui-vs-rest-microservice)
3. [Quick Start & Navigation](#3-quick-start--navigation)
4. [Page-by-Page Operational Guide ("How to Use")](#4-page-by-page-operational-guide-how-to-use)
   - [Home: Executive Overview & System Status](#home-executive-overview--system-status)
   - [Page 1: Customer Risk Scorer (Single Account Inference & Prescriptive Recourse)](#page-1-customer-risk-scorer)
   - [Page 2: Portfolio Explorer & Batch Scoring (Pandera Sanitization, 2x2 Matrix, Campaign Export)](#page-2-portfolio-explorer--batch-scoring)
   - [Page 3: Business Simulator (Unit Economics, Threshold Optimization, Decile Gains)](#page-3-business-simulator)
   - [Page 4: Model Insights & Technical Audit (CV Benchmarks, Calibration, Global SHAP, Fairness)](#page-4-model-insights--technical-audit)
   - [Page 5: About & System Governance (Model Card, Ethical Scope, Stack)](#page-5-about--system-governance)
   - [Page 6: Developer Hub & REST API Console (In-Browser Tester, Code Snippets, Docker)](#page-6-developer-hub--rest-api-console)
5. [The "What Means What" Deep Metric & Concept Dictionary](#5-the-what-means-what-deep-metric--concept-dictionary)
6. [Role-Based Operational Playbooks (User Personas)](#6-role-based-operational-playbooks-user-personas)
7. [Data Ingestion Dictionary & Pandera Sanitization Specifications](#7-data-ingestion-dictionary--pandera-sanitization-specifications)
8. [Troubleshooting, FAQs & Keepalive Architecture](#8-troubleshooting-faqs--keepalive-architecture)

---

## 1. Executive Summary & System Vision

### The Core Problem in Retail Banking
Customer acquisition in retail banking is between **5 to 7 times more expensive** than customer retention. When an account holder leaves, the bank does not merely lose a deposit line—it forfeits net interest margin (NIM), cross-selling capacity (mortgages, cards, wealth management), and transaction fee revenue. 

Historically, banks have approached churn with two suboptimal extremes:
1. **Blunt Mass Marketing:** Spraying expensive retention offers (e.g., fee waivers, cash bonuses) across all customers, squandering quarterly operational budgets on customers who were never at risk of leaving.
2. **Naive 0.50 Probability Cutoffs:** Using off-the-shelf machine learning classifiers with an uncalibrated default decision threshold of `0.50`. Because retail bank churn is naturally imbalanced (typically ~20%), a `0.50` cutoff causes the bank to **miss over 45% of genuine churners**, allowing millions of dollars in customer balances to silently runoff to competitors.

### The Solution: Prescriptive Decision Support
This Streamlit application transforms machine learning from a passive statistical prediction into an **active, cost-sensitive decision engine**:
- **Calibrated Probabilities:** An Optuna-tuned tree ensemble (XGBoost) wrapped in 5-fold Isotonic Regression ensures predicted probabilities reflect true real-world risk (Brier score = `0.0994`).
- **Mathematically Optimal Boundary ($t^* = 0.333$):** Threshold derived from Customer Lifetime Value (CLV = $500), offer intervention cost ($50), and intervention success rate (30%), maximizing net campaign dollars.
- **Explainable Diagnostics:** Real-time SHAP (SHapley Additive exPlanations) waterfalls break down the exact mathematical drivers behind every score.
- **Prescriptive Counterfactual Recourse:** Rather than just diagnosing *who* will leave, the engine simulates *what concrete actions* (digital activation, product consolidation, fee waivers) will lower departure risk and calculates expected net ROI.

---

## 2. Dual-Tier System Architecture (UI vs. REST Microservice)

To serve both non-technical business users and automated enterprise systems, the project maintains a **dual-layer architecture**:

```
                                  ┌──────────────────────────────────────────────┐
                                  │          Bank Customer Churn System          │
                                  └──────────────────────┬───────────────────────┘
                                                         │
                         ┌───────────────────────────────┴───────────────────────────────┐
                         ▼                                                               ▼
        ┌──────────────────────────────────┐                            ┌──────────────────────────────────┐
        │   Streamlit Cloud Frontend UI    │                            │     FastAPI REST Microservice    │
        │   bankcustomerchurning.          │                            │     Docker Container / Uvicorn   │
        │   streamlit.app                  │                            │     Port 8000                    │
        ├──────────────────────────────────┤                            ├──────────────────────────────────┤
        │ • Protocol: WebSockets / HTTP    │                            │ • Protocol: REST / JSON HTTP     │
        │ • Target: Bankers, Marketers,    │                            │ • Target: Core Banking Systems,  │
        │   Risk Officers, Executives      │                            │   Mobile Apps, CRM Webhooks      │
        │ • Modules: Interactive 7-page    │                            │ • Endpoints: /predict,           │
        │   Decision Support Suite         │                            │   /predict/recourse, /batch      │
        │ • State: In-Memory Session State │                            │ • Latency: Sub-10ms inference    │
        └──────────────────────────────────┘                            └──────────────────────────────────┘
```

> [!IMPORTANT]
> **Understanding Edge Routing & HTTP 404s:**  
> Navigating your web browser directly to a URL path like `https://bankcustomerchurning.streamlit.app/api/v2/user/details` will result in an **HTTP 404 Not Found**.  
> This occurs because Streamlit Community Cloud acts as a dedicated WebSocket frontend server, not a multi-tenant reverse proxy for arbitrary backend paths. Programmatic REST access is served via the FastAPI container (`churn.api:app`) documented on **Page 6: Developer Hub**.

---

## 3. Quick Start & Navigation

### Local Installation & Launch
To run the Streamlit application on your local workstation:

```bash
# 1. Clone repository and navigate to directory
git clone https://github.com/ronaldgosso/bank-churn.git
cd bank-churn

# 2. Activate virtual environment (Windows PowerShell)
.venv\Scripts\Activate.ps1

# 3. Launch Streamlit app
streamlit run app/Home.py
```
The browser will automatically open to `http://localhost:8501`.

### Application Navigation Tree
The application is organized into 7 distinct functional modules accessible via the left sidebar:

| Page | File | Primary Functional Purpose |
| :--- | :--- | :--- |
| **Home** | `app/Home.py` | Executive cockpit, headline KPI metrics, problem-solution architecture, model status badges. |
| **1. Customer Risk Scorer** | `app/pages/1_Customer_Risk_Scorer.py` | Single customer real-time inference, risk gauge, SHAP waterfall, and prescriptive recourse. |
| **2. Portfolio Explorer** | `app/pages/2_Portfolio_Explorer.py` | Batch CSV upload, Pandera data hygiene audit, 2x2 Strategic Action Matrix, campaign CSV export. |
| **3. Business Simulator** | `app/pages/3_Business_Simulator.py` | Interactive unit economics sliders, net profit vs. threshold curve, decile gains, and lift tables. |
| **4. Model Insights** | `app/pages/4_Model_Insights.py` | 5-fold CV benchmarks, holdout ROC/PR curves, probability calibration, global SHAP, fairness audit. |
| **5. About** | `app/pages/5_About.py` | Formal model card, training provenance, ethical governance boundaries, and reproducibility. |
| **6. Developer API** | `app/pages/6_Developer_API.py` | Interactive REST console, live request payload simulator, multi-language code snippets, Docker specs. |

---

## 4. Page-by-Page Operational Guide ("How to Use")

---

### Home: Executive Overview & System Status

The **Home** page serves as the executive summary and live system health monitor.

![Home Page Overview](file:///c:/Users/Neptune/Documents/Projects/bank-churn/public/fintech_decision_hero.png)

#### What You See on This Page:
1. **Headline Metric Cards (Top Banner):**
   - **Test ROC-AUC (`86.8%`):** Evaluated on the strict 2,000 holdout customer split. Exceeds the target requirement of $\ge 85.0\%$.
   - **Test PR-AUC (`71.7%`):** Precision-Recall Area Under Curve. Represents a **3.5x improvement** over the uninformative baseline (20.4%).
   - **Top 10% Decile Lift (`4.20x`):** Targeting the top 10% highest-risk customers captures **42.0% of all churners**.
   - **Optimal Decision Threshold (`0.333`):** The mathematically profit-optimal boundary derived from business economics.
   - **Holdout Net Profit (`$17,850`):** Total net dollars generated on the 2,000 test customers under standard intervention parameters.
2. **Executive Overview Narrative:** Details the business dilemma (acquisition vs. retention), the cost of uncalibrated models, and the prescriptive decision architecture.
3. **System & Model Status (Bottom Grid):**
   - **Production Model:** `CalibratedClassifierCV(XGBoost)`
   - **Training Dataset:** `8,000 customers (Stratified 80%)`
   - **Evaluation Holdout:** `2,000 customers (Strict Holdout)`
   - **Data Integrity SHA-256 Hash:** Cryptographic fingerprint ensuring zero data leakage between splits.

#### How to Use It:
- **For Senior Leadership:** Review headline KPIs and status badges to verify model health and operational readiness before committing marketing budget.
- **For Quick Orientation:** Use the sidebar navigation links to jump directly to specific operational modules.

---

### Page 1: Customer Risk Scorer

The **Customer Risk Scorer** provides on-demand, real-time risk assessment and prescriptive recourse for individual account holders.

```
┌────────────────────────────────────────────────────────────────────────────────────────────────┐
│ Page 1: Customer Risk Scorer                                                                   │
├────────────────────────────────────────────────────────────────────────────────────────────────┤
│ 1. Archetype Selection: [ Custom Input ] [ High Risk Customer ] [ Borderline ] [ Low Risk ]    │
│ 2. Feature Input Form:                                                                         │
│    - Credit Score (300-900)    - Tenure (0-10 yrs)             - Has Credit Card? (Yes/No)     │
│    - Geography (FR / DE / ES)  - Balance ($ USD)               - Is Active Member? (Yes/No)    │
│    - Gender (Female / Male)    - Number of Products (1 - 4)    - Estimated Salary ($ USD)      │
│    - Age (18 - 100)                                                                            │
├────────────────────────────────────────────────────────────────────────────────────────────────┤
│ 3. Decision Assessment:                                                                        │
│    [ 3-Zone Radial Risk Gauge ]       │ [ Operational Recommendation Banner ]                  │
│    - Green: Safe (<16.7%)             │ - High Risk: Prioritize Retention Outreach             │
│    - Amber: Watchlist (16.7% - 33.3%) │ - Expected profit positive under $50 offer             │
│    - Red: High Risk (>=33.3%)         │ - Actionable RM instructions                           │
├────────────────────────────────────────────────────────────────────────────────────────────────┤
│ 4. Explainability: Top 5 Drivers     │ SHAP Decision Waterfall Plot                            │
│    ▲ Age = 52 (+0.68 log-odds)        │ Visual step-by-step contribution of each feature       │
│    ▲ NumOfProducts = 3 (+0.54)        │ from baseline expected value E[f(x)] to f(x).          │
│    ▼ Balance = $115k (-0.12)          │                                                        │
├────────────────────────────────────────────────────────────────────────────────────────────────┤
│ 5. Prescriptive Recourse Table:                                                                │
│    Strategy            Lever         Cost   New Prob  Risk Drop  Expected Net ROI              │
│    Digital Activation  Active=1      $15    24.2%     -38.5%     +$32.50                       │
│    Product Consol.     Products=2    $30    18.1%     -54.0%     +$46.20                       │
│    Retention Bundle    Active+Prod   $50    12.4%     -68.5%     +$58.40 (★ OPTIMAL)           │
├────────────────────────────────────────────────────────────────────────────────────────────────┤
│ 6. Relationship Manager Playbook:                                                              │
│    - Personalized Greeting: "Thank you for your 2 years of partnership with our bank."        │
│    - Context Observation: "Noticed 3 products held with $115,000 balance."                    │
│    - Tailored Pitch: "Consolidate into fee-free premier checking + enable mobile alerts."      │
│ 7. Interactive What-If Sandbox (Test custom changes on the fly)                                │
└────────────────────────────────────────────────────────────────────────────────────────────────┘
```

#### Step-by-Step Instructions:
1. **Load a Representative Archetype (or Enter Custom Data):**
   - Click one of the quick radio presets:
     - **High Risk Customer:** 52-year-old German female, 3 products, inactive member, balance $115,000.
     - **Borderline Customer:** 42-year-old French female, 1 product, inactive, balance $92,000.
     - **Low Risk Customer:** 32-year-old French male, 2 products, active member, balance $45,000.
     - **Custom Input:** Freely adjust all 10 customer attributes.
2. **Click "Calculate Churn Risk":**
   - Triggers strict Pydantic schema validation.
   - Executes real-time inference via the calibrated XGBoost pipeline.
3. **Review the Risk Assessment Gauge & Operational Directive:**
   - **Radial Gauge:** Displays the exact calibrated probability ($0.0\%$ to $100.0\%$) with colored threshold zones:
     - **Safe (Green):** $p < 0.5 \times t^*$ ($< 16.7\%$). Normal account servicing.
     - **Watchlist (Amber):** $16.7\% \le p < 33.3\%$. Automated digital nudges.
     - **High Risk (Red):** $p \ge 33.3\%$. Direct RM retention outreach justified.
4. **Inspect Top Churn Drivers & SHAP Waterfall:**
   - The **Top 5 Factors** display clear red up-arrows (▲ increasing risk) and green down-arrows (▼ decreasing risk).
   - The **SHAP Waterfall Plot** illustrates how the customer's attributes shift the model output from the base value $E[f(X)]$ to the final prediction $f(x)$.
5. **Analyze Prescriptive Interventions (Counterfactual Recourse):**
   - View the **Evaluated Retention Interventions Table** comparing concrete actions:
     - *Digital Banking Activation:* Enrolling in mobile banking (`IsActiveMember = 1`).
     - *Product Optimization:* Shifting product count to the empirical sweet-spot (`NumOfProducts = 2`).
     - *Rewards Card Activation:* Adding a card with fee waiver (`HasCrCard = 1`).
     - *Comprehensive Retention Bundle:* Multi-lever intervention.
   - Compare new probabilities against the decision boundary on the **Probability Comparison Bar Chart**.
   - Note the **Optimal Strategy** marked with `★ OPTIMAL` based on maximum expected net ROI.
6. **Deploy the Relationship Manager Playbook:**
   - Copy the generated phone/meeting script containing the customer-specific greeting, balance observation, tailored pitch, and expected retention benefit.
7. **Simulate in the "What-If" Sandbox:**
   - Open the expandable sandbox at the bottom of the page to adjust `IsActiveMember`, `NumOfProducts`, or `HasCrCard` dynamically and observe immediate probability changes.

---

### Page 2: Portfolio Explorer & Batch Scoring

The **Portfolio Explorer** provides macro-level portfolio risk management, capital-at-risk analysis, automated data sanitization, and targeted campaign extraction.

![Sanitization Pipeline](file:///c:/Users/Neptune/Documents/Projects/bank-churn/public/sanitization_pipeline_flow.png)

#### Step-by-Step Instructions:
1. **Load Portfolio Data:**
   - **Upload Custom File:** Drag and drop any batch customer CSV into the uploader.
   - **Download Template CSV:** Click "Download Template CSV (50 rows)" for a pre-formatted example.
   - **Load Demo Records:** Click "Load 2,000 Holdout Test Records" to immediately explore the holdout dataset.
   - **Clear Portfolio:** Click "Clear Portfolio" at any time to unload active records and reset session state.
2. **Review the Pandera Sanitization & Hygiene Audit:**
   - The application automatically passes the uploaded file through `churn.validation.sanitize_portfolio_data`:
     - Resolves column aliases (e.g., `country` $\to$ `Geography`, `credit_score` $\to$ `CreditScore`).
     - Verifies presence of all 10 core modeling features.
     - Clamps out-of-range numeric anomalies (e.g., negative balances or ages outside 18–100).
     - Imputes missing values with medians and modes.
     - Preserves non-modeling identifiers (`CustomerId`, `Surname`, `RowNumber`) for customer outreach.
   - Inspect the **Data Hygiene Metric Cards** (Total Records, Core Features Verified, Identifiers Preserved, Data Hygiene Status).
3. **Assess Portfolio Risk & Balance Runoff Exposure:**
   - **Total Accounts & Assets:** Total customer headcount and total aggregate balance.
   - **High Risk Accounts:** Count and percentage of customers exceeding the decision boundary ($p \ge 0.333$).
   - **High Risk Balance:** Total dollar balance held by high-risk customers and share of total assets.
   - **Expected Balance Runoff:** Total probabilistic annual deposit loss:
     $$\text{Expected Runoff} = \sum_{i=1}^{N} \text{Balance}_i \times p_i$$
   - **Campaign Budget:** Required marketing budget under standard $50 offer cost:
     $$\text{Campaign Cost} = \text{Contacted Customers} \times \$50$$
4. **Analyze the 2x2 Strategic Value-vs-Risk Matrix:**
   - Customers are partitioned into four strategic action quadrants based on Churn Risk ($p \ge 0.333$) and Account Asset Median ($Balance \ge \$50,000$):
     - **VIP Save Priority (Red):** High Risk + High Asset. Top priority for personal relationship manager phone calls and bespoke concessions.
     - **Automated Digital Save (Amber):** High Risk + Modest Asset. Targeted for automated digital marketing, push notifications, and email incentives.
     - **Relationship Deepening (Blue):** Low Risk + High Asset. Prime candidates for cross-selling premium wealth products.
     - **Standard Service (Green):** Low Risk + Modest Asset. Standard banking service; zero retention spend required.
   - Switch to the **Probability Density Histogram** tab to visualize the overall distribution of portfolio risk against the threshold line.
5. **Filter and Extract Targeted Campaign Lists:**
   - Use the multi-select filter controls:
     - *Filter by Strategic Priority* (e.g., isolate only `VIP Save Priority`).
     - *Filter by Risk Tier* (`High`, `Medium`, `Low`).
     - *Filter by Geography* (`France`, `Germany`, `Spain`).
     - *Minimum Churn Probability Slider*.
   - The preview table automatically updates, displaying calibrated churn probabilities with interactive progress bars, formatted currency balances, and contact checkmarks.
   - *(Session State Persistence Note: Interacting with sliders or filters preserves the uploaded portfolio without reloading or resetting.)*
6. **Download Filtered Campaign CSV:**
   - Click the primary blue **Download Filtered Campaign CSV** button to export the tailored call-list directly to your machine.

---

### Page 3: Business Simulator

The **Business Simulator** allows financial executives and campaign planners to model the unit economics of retention campaigns, find profit-optimal decision boundaries, and evaluate decile capture curves.

![Retention Economics](file:///c:/Users/Neptune/Documents/Projects/bank-churn/public/retention_economics_concept.png)

#### Step-by-Step Instructions:
1. **Adjust Campaign Economics & Budget Parameters:**
   - **Customer Lifetime Value (CLV $):** Slider range `$100` to `$2,000` (default `$500`). The net future profit preserved when an at-risk customer is prevented from leaving.
   - **Retention Offer Cost ($):** Slider range `$10` to `$200` (default `$50`). Direct operational cost of reaching out (e.g., fee discount, statement credit, dedicated banker time).
   - **Offer Save Rate (%):** Slider range `5%` to `80%` (default `30%`). Empirical percentage of contacted churners who accept the intervention and remain.
   - **Budget Limit (Top K% Risk):** Slider range `5%` to `50%` (default `20%`). Operational headcount constraint restricting outreach to the highest-risk deciles.
2. **Review Financial Performance Comparisons:**
   - **Optimal Decision Threshold ($t^*$):** Calculated in real time via cost-sensitive decision theory:
     $$t^* = \frac{\text{Cost}}{\text{CLV} \times \text{Save Rate}} = \frac{50}{500 \times 0.30} = 0.333$$
   - **Optimal Threshold Net Profit:** Total net campaign profit generated on the holdout test set using threshold $t^*$.
   - **Default 0.50 Threshold Profit:** Profit generated using a naive uncalibrated 0.50 cutoff, highlighting the exact dollar loss and percentage of missed churners.
   - **Budget (Top K%) Profit:** Profit and decile lift achieved if strictly capping outreach to the specified top $K\%$ of accounts.
3. **Examine the Net Profit vs. Threshold Curve:**
   - Observe the inverted U-shaped curve showing net campaign profit across all potential thresholds from $0.05$ to $0.95$.
   - The peak of the curve aligns precisely with the theoretical optimal threshold ($t^* = 0.333$).
   - A dotted vertical line marks the default $0.50$ threshold, clearly demonstrating the financial shortfall of standard machine learning defaults.
4. **Inspect the Cumulative Gain & Lift Curves:**
   - **Cumulative Gain Chart:** Visualizes the percentage of total churners captured as the proportion of contacted customers increases.
   - Compares the model's convex capture curve against the 45-degree diagonal **Random Baseline**.
   - Identifies the user-selected budget cutoff point (e.g., contacting top 20% captures over 60% of churners).
5. **Audit the Decile Gain & Lift Breakdown Table:**
   - Inspect all 10 customer deciles sorted from highest risk to lowest risk:
     - **Decile Churn Rate:** Observed churn frequency within that specific decile.
     - **Cumulative Population %:** Total customer cohort reached ($10\%, 20\%, \dots, 100\%$).
     - **Cumulative Churn Captured %:** Total share of all portfolio churners captured.
     - **Decile Lift:** Multiplier over random chance within the specific decile.
     - **Cumulative Lift:** Overall lift across all contacted deciles up to that point.

---

### Page 4: Model Insights & Technical Audit

The **Model Insights** page provides full quantitative transparency into model training, evaluation benchmarks, diagnostic discrimination curves, probability calibration, SHAP feature rankings, and demographic fairness.

#### Step-by-Step Instructions:
1. **Cross-Validation Model Benchmark:**
   - Review the stratified 5-fold cross-validation table comparing 4 candidate architectures evaluated during Phase 3:
     - *Logistic Regression (L2 Baseline)*
     - *Random Forest Classifier*
     - *LightGBM Classifier*
     - *Optuna-Tuned XGBoost (Production Champion)*
   - Compare performance across 6 standardized metrics: ROC-AUC, PR-AUC, Recall, Precision, F1 Score, and Brier Score.
   - Best-in-class values are automatically highlighted in light green.
2. **Inspect Diagnostic Curves (Holdout Test Set):**
   - **ROC Curve (Left):** Area Under Curve = `0.8682` (exceeding the $\ge 0.8500$ production specification). Displays the False Positive Rate ($14.2\%$) at the operating threshold.
   - **Precision-Recall Curve (Right):** Area Under Curve = `0.7174` (3.5x higher than the horizontal 20.4% baseline churn line). Highlights the operating point with Precision = `63.8%` and Recall = `61.2%`.
3. **Evaluate Probability Calibration (Reliability Diagram):**
   - **Reliability Plot (Top Subplot):** Plots predicted probability bins against actual empirical churn frequency. The calibrated XGBoost curve closely tracks the diagonal 45-degree dotted line ("Perfect Calibration").
   - **Histogram (Bottom Subplot):** Displays the customer density distribution across probability buckets.
   - **Brier Score:** Achieves `0.0994` (beating the $< 0.1000$ target).
   - **Decile Reliability Table:** Confirms minimal drift ($< 2.5\%$) across all probability deciles.
4. **Review Global SHAP Feature Importance:**
   - Horizontal bar chart of mean absolute SHAP values ($|\log\text{-odds}|$) across all 10 features and engineered ratios.
   - **Primary Takeaways:**
     - *Age:* Dominant non-linear driver; churn spikes sharply between ages 45 and 65.
     - *Number of Products:* Customers with 1 product churn at ~27%; 2 products is optimal (~7.6%); but 3 or 4 products triggers a catastrophic **82% to 100% churn rate**.
     - *Geography (Germany):* German accounts experience ~2x the churn rate of France or Spain.
     - *Active Membership:* Consistent protective factor across all demographic slices.
5. **Audit Demographic Fairness & Disparity Metrics:**
   - **Gender Audit Table:** Compares Female vs. Male cohorts across sample size, churn base rate, ROC-AUC, selection rate, recall, and precision.
   - **Geography Audit Table:** Compares France, Germany, and Spain.
   - **Disparity Metrics:** Selection Rate Ratio, Recall Gap, Precision Gap.
   - **Governance Disclosure:** Explains that selection rate differences stem from underlying historical base-rate variations (e.g., German accounts naturally churn at 30% vs. 17% in France), while within-group ROC-AUC discrimination remains consistently strong ($> 0.85$) across all sub-populations.

---

### Page 5: About & System Governance

The **About** page contains the formal system documentation, model card, technical architecture diagram, known limitations, and author credentials.

#### Step-by-Step Instructions:
1. **Model Card Tab:**
   - Inspect formal model specifications: Model architecture, hyperparameter optimization method (Optuna TPE), calibration algorithm (Isotonic Regression), sample sizes, and official target vs. achieved score cards.
   - Review intended operational purpose and explicitly prohibited out-of-scope uses (e.g., credit underwriting, loan denials, account terminations).
2. **Limitations & Governance Tab:**
   - Review known technical boundaries:
     - *Cross-Sectional Snapshot:* Absence of longitudinal time-series data (e.g., 90-day balance velocity, transaction recency).
     - *Heuristic Financial Parameters:* Unit economics assume flat $500 CLV and $50 cost; recommends dynamic per-customer estimation in future releases.
     - *Geographic Distribution:* Trained on European retail banking records; recalibration required before deploying in Latin American or North American jurisdictions.
3. **Software Architecture Tab:**
   - Inspect the end-to-end system architecture schematic (`Bank Churn System Architecture.png`).
   - Review the complete technology stack: Scikit-learn, XGBoost, LightGBM, Imbalanced-learn, Optuna, MLflow, Pandera, Streamlit, FastAPI, Docker, and Pytest.
4. **Reproducibility & Author Tab:**
   - Review one-line bash commands to reproduce the entire training, testing, linting, and serving workflow.
   - Access author contact details and GitHub repository links.

---

### Page 6: Developer Hub & REST API Console

The **Developer Hub** provides interactive testing tools, client code generators, and deployment specifications for the production FastAPI microservice.

#### Step-by-Step Instructions:
1. **Test Endpoints In-Browser:**
   - **POST `/predict` (Single Customer):**
     - Select an archetype preset (`High Risk`, `Borderline`, `Low Risk`).
     - Inspect or edit the JSON payload.
     - Click "Send Request to /predict" to execute in-memory inference and view the live response JSON, HTTP status (`200 OK`), and sub-millisecond execution latency.
   - **POST `/predict/recourse` (Prescriptive Analytics):**
     - Select a customer payload and input custom CLV and save-rate values.
     - Click "Send Request to /predict/recourse" to test counterfactual scenario generation and RM talking points.
   - **POST `/predict/batch`:**
     - Click "Execute Batch Prediction" to simulate sending an array of customer records in a single payload and inspect the formatted result table.
   - **GET `/health` & `/meta`:**
     - Review microservice health status, active model version, and Pandera column schemas.
2. **Copy Multi-Language Client Code:**
   - Navigate between tabs to copy production-ready integration snippets:
     - **cURL (Bash):** For terminal testing and command-line scripts.
     - **Python (Requests):** For Python microservices, Jupyter notebooks, or Airflow ETL pipelines.
     - **TypeScript / JavaScript (Fetch):** For web portals, React/Vue dashboards, or CRM integrations.
3. **Container Deployment & Docker Instructions:**
   - Copy exact Docker commands to build and run the production image:
     ```bash
     docker build -t bank-churn-api:latest .
     docker run -d -p 8000:8000 --name bank-churn-service bank-churn-api:latest
     ```
   - Review cloud hosting deployment guides for Render, Railway, Fly.io, AWS ECS, Google Cloud Run, and Hugging Face Spaces.

---

## 5. The "What Means What" Deep Metric & Concept Dictionary

This section provides an exhaustive, unambiguous explanation of every term, metric, and mathematical formula used throughout the application.

---

### 1. Predicted Churn Probability ($p$)
- **What It Means:** The calibrated statistical likelihood (between $0.000$ and $1.000$, or $0.0\%$ to $100.0\%$) that an individual account holder will terminate their banking relationship within the observation window.
- **Why It Matters:** Unlike uncalibrated classification scores that only offer relative rankings, this number represents an **empirical frequency**. A predicted probability of $0.35$ means that out of 100 customers with this exact risk profile, approximately 35 will actually close their accounts.

---

### 2. Cost-Sensitive Optimal Decision Threshold ($t^*$)
- **What It Means:** The exact probability cutoff above which it becomes financially rational to make a retention intervention offer.
- **Mathematical Derivation:**
  Let:
  - $V = \text{Customer Lifetime Value (CLV preserved if saved)} = \$500$
  - $C = \text{Direct Cost of the retention intervention} = \$50$
  - $r = \text{Offer success / save rate} = 30\% = 0.30$

  The expected profit of intervening on a customer with predicted churn probability $p$ is:
  $$\mathbb{E}[\text{Profit}] = p \times r \times V - C$$

  Intervention is profitable when $\mathbb{E}[\text{Profit}] \ge 0$:
  $$p \times r \times V \ge C \implies p \ge \frac{C}{r \times V}$$

  Substituting the project's standard retail banking assumptions:
  $$t^* = \frac{50}{0.30 \times 500} = \frac{50}{150} = 0.3333 \quad (33.3\%)$$
- **Why It Matters:** Any customer with $p \ge 0.333$ yields a positive expected return on intervention spend. Setting the cutoff at $0.50$ arbitrarily throws away profitable retention opportunities.

---

### 3. The 0.50 Default Threshold Flaw
- **What It Means:** The standard, uncalibrated threshold used by default in standard classification libraries (`model.predict()`).
- **Why It Fails:** Default $0.50$ cutoffs assume that False Positives (spending $50 on a customer who would not have churned) and False Negatives (losing a $500 customer because you did not intervene) carry equal business costs. In retail banking, losing an account costs **10 times more** than an offer. Using $0.50$ misses $45.2\%$ of all churners in the portfolio, leaving an estimated **$8,500+ in net profit on the table** for every 2,000 customers.

---

### 4. Expected Balance Runoff
- **What It Means:** The total probabilistic dollar amount of customer deposits expected to leave the bank over the coming annual cycle.
- **Mathematical Formula:**
  $$\text{Expected Runoff} = \sum_{i=1}^{N} \text{Balance}_i \times p_i$$
- **Why It Matters:** In treasury and asset-liability management (ALM), banks must maintain adequate reserve capital against anticipated deposit flight. Expected Balance Runoff provides liquidity planners with a probabilistic dollar forecast rather than a vague churn count.

---

### 5. Risk Tiers (High, Medium, Low)
The application partitions customers into three operational bands relative to the optimal threshold $t^* = 0.333$:

| Risk Tier | Probability Range | Color Code | Operational Meaning & Required Action |
| :--- | :--- | :--- | :--- |
| **High Risk** | $p \ge t^*$ ($\ge 33.3\%$) | Coral Red (`#ef4444`) | **Active Retention Target:** Expected profit is positive under direct outreach. Assign to relationship manager or trigger direct retention package. |
| **Medium Risk** | $0.5 \times t^* \le p < t^*$ ($16.7\% \le p < 33.3\%$) | Amber (`#f59e0b`) | **Watchlist:** Direct $50 offer is not economically justified, but departure risk is elevated. Trigger low-cost automated digital engagement (mobile push, email check-in). |
| **Low Risk** | $p < 0.5 \times t^*$ ($< 16.7\%$) | Emerald Green (`#10b981`) | **Standard Account:** Customer exhibits healthy engagement patterns. Zero intervention spend; preserve operational budget. |

---

### 6. The 2x2 Strategic Action Matrix
Cross-references Churn Risk ($p \ge t^*$) with Asset Value ($\text{Balance} \ge \text{Median Non-Zero Balance}$ or $\$50,000$):

```
                       High Asset (Balance >= $50k)
                                     ▲
                                     │
           RELATIONSHIP DEEPENING    │    VIP SAVE PRIORITY
          (Low Risk / High Asset)    │  (High Risk / High Asset)
          • Cross-sell wealth mgmt   │  • Dedicated RM outreach
          • Premium card upgrades    │  • Fee waivers & rate bonus
                                     │
       ◄─────────────────────────────┼─────────────────────────────►
        Low Risk (p < 0.333)         │         High Risk (p >= 0.333)
                                     │
             STANDARD SERVICE        │   AUTOMATED DIGITAL SAVE
         (Low Risk / Modest Asset)   │ (High Risk / Modest Asset)
          • Standard branch service  │  • Mobile push notifications
          • Self-service digital     │  • Email campaign / survey
                                     │
                                     ▼
                      Modest Asset (Balance < $50k)
```

- **VIP Save Priority:** Accounts with large deposits that are at severe risk of leaving. Highest priority for senior branch personnel.
- **Automated Digital Save:** Accounts with modest balances where an expensive RM call is not cost-effective, but automated digital incentives can cost-effectively preserve the relationship.
- **Relationship Deepening:** High-deposit customers who are highly loyal. Ideal for upselling private banking or wealth advisory services.
- **Standard Service:** Stable, standard retail accounts requiring no special intervention.

---

### 7. ROC-AUC (Receiver Operating Characteristic — Area Under Curve)
- **What It Means:** A measure of the model's ability to rank customers correctly across all possible decision thresholds, regardless of class balance.
- **Value Achieved:** `0.8682` (86.8%).
- **Interpretation:** If you randomly choose one customer who churned and one customer who stayed, there is an **86.8% probability** that the model assigned a higher churn risk to the genuine churner. Scores above 0.85 are considered excellent in retail financial modeling.

---

### 8. PR-AUC (Precision-Recall Area Under Curve)
- **What It Means:** Evaluates the trade-off between Precision and Recall specifically for the minority positive class (churners).
- **Value Achieved:** `0.7174` (71.7%).
- **Why It Matters:** In imbalanced datasets (~20% churn), ROC-AUC can be overly optimistic due to a large number of True Negatives. PR-AUC focuses exclusively on the churn class. An uninformative model achieves a PR-AUC equal to the baseline churn rate (`20.35%`). The model's score of `71.74%` represents a **3.5x improvement** over random chance.

---

### 9. Decile Lift and Cumulative Lift
- **Decile Lift:** How many times better the model is at finding churners in a specific decile compared to random selection.
  $$\text{Decile Lift} = \frac{\text{Churn Rate in Decile}}{\text{Overall Portfolio Churn Rate}}$$
- **Top 10% Decile Lift (`4.20x`):** The top 10% highest-risk customers have a churn rate of **85.5%**, compared to the portfolio average of **20.4%** ($85.5 / 20.4 \approx 4.20\times$).
- **Cumulative Lift:** The overall lift achieved when contacting all customers down to decile $k$. Contacting the top 20% achieves a cumulative lift of **3.1x**, capturing over 62% of all churners.

---

### 10. Cumulative Gain (Capture Rate)
- **What It Means:** The cumulative percentage of total churners captured when contacting a given percentage of the customer base.
- **Example:** In a test portfolio of 2,000 customers containing 407 churners, contacting the top 20% (400 customers) captures **252 churners** (a **61.9% capture rate**).

---

### 11. Brier Score & Probability Calibration
- **Brier Score:** The mean squared difference between predicted probabilities and actual binary outcomes ($0$ or $1$):
  $$\text{BS} = \frac{1}{N} \sum_{i=1}^{N} (p_i - y_i)^2$$
- **Value Achieved:** `0.0994` (beating the $< 0.1000$ target).
- **Why It Matters:** Tree ensemble models often produce uncalibrated probabilities with sigmoidal distortion. By applying **Isotonic Regression** via 5-fold cross-validation, the predicted probabilities are recalibrated to match real-world empirical churn rates.

---

### 12. SHAP Values ($\phi_i$) & Waterfall Plots
- **SHAP Value:** The exact marginal contribution of a specific feature to the prediction's deviation from the baseline expected value, measured in log-odds:
  $$\ln\left(\frac{p}{1-p}\right) = E[f(X)] + \sum_{j=1}^{M} \phi_j$$
- **Positive SHAP (+):** Pushes the prediction toward departure (increases churn risk). Displayed as red bars.
- **Negative SHAP (-):** Pushes the prediction toward retention (decreases churn risk). Displayed as green bars.
- **Waterfall Plot:** Graphically demonstrates how the baseline log-odds moves step-by-step through each customer attribute to produce the final calibrated score.

---

### 13. Counterfactual Recourse & Expected Net ROI
- **Counterfactual Recourse:** Identifying the minimal, realistic changes to actionable customer features that will shift an account from High Risk to Safe.
- **Actionable vs. Non-Actionable Levers:**
  - *Non-Actionable (Protected/Fixed):* Age, Gender, Geography, Tenure cannot be altered by a banker.
  - *Actionable Levers:* `IsActiveMember` (mobile app enrollment), `NumOfProducts` (account consolidation), `HasCrCard` (rewards card issuance).
- **Expected Net ROI Calculation:**
  $$\text{Expected Net ROI} = (\Delta p) \times \text{CLV} \times \text{Save Rate} - \text{Intervention Cost}$$
  Where $\Delta p = p_{\text{original}} - p_{\text{new}}$.

---

### 14. Demographic Fairness & Disparity Metrics
- **Demographic Parity / Selection Rate Ratio:** The ratio of the selection rate (percentage flagged as high risk) in an unprivileged group to that of a privileged group:
  $$\text{SRR} = \frac{\text{Selection Rate}_{\text{Unprivileged}}}{\text{Selection Rate}_{\text{Privileged}}}$$
  Under the US EEOC "Four-Fifths (80%) Rule", an SRR between 0.80 and 1.25 indicates parity.
- **Equal Opportunity / Recall Gap:** The difference in True Positive Rates (Recall) across protected groups:
  $$\text{Recall Gap} = |\text{Recall}_{\text{Group A}} - \text{Recall}_{\text{Group B}}|$$
  A small recall gap ensures that genuine churners are detected and offered retention support at equal rates regardless of demographic background.

---

## 6. Role-Based Operational Playbooks (User Personas)

---

### Playbook A: Branch Relationship Manager (RM)
*Goal: Retain high-value retail clients during routine reviews or inbound service inquiries.*

1. **Morning Risk Review:**
   - Open **Page 1: Customer Risk Scorer**.
   - Input the customer's 10 profile attributes before an upcoming client meeting or outbound phone call.
2. **Review Risk Tier & Waterfall:**
   - If the gauge is **Red (High Risk)**, note the top 3 drivers on the SHAP waterfall.
   - For example, if `NumOfProducts = 3` and `IsActiveMember = 0`, the client is suffering from account fragmentation and digital disengagement.
3. **Execute Prescriptive Recourse:**
   - Review the **Prescriptive Interventions Table**. Identify the strategy with the highest Expected Net ROI.
   - Read the generated **Relationship Manager Playbook** script directly from the screen:
     - Greet using their tenure.
     - Cite their balance and current products.
     - Pitch the recommended action (e.g., "We can consolidate your secondary checking accounts into our premier fee-free tier and activate mobile instant alerts today").
4. **Log Outcome:** Record intervention in the bank CRM and monitor account status.

---

### Playbook B: Retention Marketing Campaign Lead
*Goal: Maximize net campaign return across a quarterly budget of $50,000.*

1. **Portfolio Ingestion:**
   - Open **Page 2: Portfolio Explorer**.
   - Upload the quarterly branch portfolio CSV (e.g., 10,000 customers).
   - Review the **Pandera Sanitization Audit** to verify 10/10 features are valid.
2. **Budget Optimization:**
   - Open **Page 3: Business Simulator**.
   - Set Offer Cost to `$50` and expected Save Rate to `30%`.
   - Observe that the optimal threshold is `0.333`.
   - With a $50,000 budget, the team can contact at most $1,000$ customers ($50,000 / 50$).
   - Adjust the **Budget Limit (Top K% Risk)** slider until the contact count matches $1,000$ customers (e.g., Top 10%).
   - Confirm expected decile lift ($4.2x$) and net campaign profit.
3. **Campaign List Extraction:**
   - Return to **Page 2: Portfolio Explorer**.
   - Filter by `VIP Save Priority` and `Automated Digital Save`.
   - Set Minimum Churn Probability to the budget quantile.
   - Click **Download Filtered Campaign CSV**.
   - Distribute the VIP list to branch RMs and inject the digital list into the automated marketing platform.

---

### Playbook C: Retail Banking Risk & Credit Officer
*Goal: Monitor deposit runoff risk, evaluate capital exposure, and prevent discriminatory lending practices.*

1. **Portfolio Runoff Audit:**
   - Open **Page 2: Portfolio Explorer**.
   - Examine **High Risk Balance** and **Expected Balance Runoff**.
   - Compare Expected Runoff against quarterly liquidity contingency reserves.
2. **Fairness & Governance Review:**
   - Open **Page 4: Model Insights & Audit**.
   - Inspect Section 5 (**Demographic Fairness & Disparity Audit**).
   - Verify that model discrimination (ROC-AUC) remains balanced ($> 0.85$) across Gender (Female vs. Male) and Geography (France, Germany, Spain).
   - Confirm that selection rate disparities align with empirical base-rate differences and document compliance for regulatory oversight.

---

### Playbook D: Lead Machine Learning Engineer / Platform Auditor
*Goal: Monitor inference microservice health, test schema validation, and integrate REST endpoints.*

1. **Endpoint Health & Diagnostics:**
   - Open **Page 6: Developer Hub & REST API Console**.
   - Inspect `GET /health` and `GET /meta` responses.
   - Verify training timestamp and model hash.
2. **Interactive Payload Validation:**
   - Test single customer payloads against `/predict` and observe inference latency ($< 15\text{ms}$).
   - Test batch inference against `/predict/batch`.
3. **Container Deployment:**
   - Build and run the official Docker image using the provided container specifications.
   - Connect downstream microservices using the generated Python or TypeScript code snippets.

---

## 7. Data Ingestion Dictionary & Pandera Sanitization Specifications

### Complete Feature Dictionary

| Column Name | Pandera Type | Acceptable Range | Required? | Default Imputation | Description |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `CreditScore` | `int` | `300` to `900` | **Yes** | Median (`650`) | Customer credit bureau score. |
| `Geography` | `str` | `France`, `Germany`, `Spain` | **Yes** | Mode (`France`) | Primary country of account residency. |
| `Gender` | `str` | `Female`, `Male` | **Yes** | Mode (`Female`) | Customer gender. |
| `Age` | `int` | `18` to `100` | **Yes** | Median (`37`) | Customer age in years. |
| `Tenure` | `int` | `0` to `10` | **Yes** | Median (`5`) | Number of years client has held an account. |
| `Balance` | `float` | $\ge 0.0$ | **Yes** | Median (`0.0` or median) | Current deposit balance in USD. |
| `NumOfProducts` | `int` | `1` to `4` | **Yes** | Mode (`1`) | Total number of bank products held. |
| `HasCrCard` | `int` | `0` or `1` | **Yes** | Mode (`1`) | Binary flag: 1 = holds credit card, 0 = none. |
| `IsActiveMember` | `int` | `0` or `1` | **Yes** | Mode (`1`) | Binary flag: 1 = active digital/branch user, 0 = inactive. |
| `EstimatedSalary`| `float` | $\ge 0.0$ | **Yes** | Median (`$100,000`) | Estimated annual gross salary in USD. |
| `CustomerId` | `int` | $> 0$ | *Optional* | Preserved | Non-modeling customer identifier. |
| `Surname` | `str` | Text | *Optional* | Preserved | Customer last name (preserved for RM outreach). |
| `RowNumber` | `int` | $> 0$ | *Optional* | Preserved | Row index (preserved if present). |
| `Exited` | `int` | `0` or `1` | *Optional* | Ground truth | Target label: 1 = churned, 0 = retained. |

### Automatic Column Alias Resolution
The pipeline automatically detects and resolves common column synonyms during portfolio upload:

```
credit_score, creditscore, score   ──►  CreditScore
country, nation                    ──►  Geography
sex                                ──►  Gender
num_of_products, products, numprod ──►  NumOfProducts
has_cr_card, hascreditcard, card   ──►  HasCrCard
is_active_member, active_member    ──►  IsActiveMember
estimated_salary, salary           ──►  EstimatedSalary
customer_id, client_id             ──►  CustomerId
last_name, surname, name           ──►  Surname
churn, churned, status             ──►  Exited
```

### Outlier Clamping & Data Coercion Policies
1. **Case Normalization:** String columns (`Geography`, `Gender`) are automatically stripped of whitespace and title-cased. Single-letter gender codes (`'M'`, `'F'`) are expanded to `'Male'` and `'Female'`.
2. **Numeric Clamping:** Numeric values outside plausible banking bounds are clamped to schema extremes (e.g., Credit Score $> 900$ is clamped to $900$; Age $< 18$ is coerced to $18$).
3. **Identifier Preservation:** Non-modeling identifiers (`CustomerId`, `Surname`) are isolated before inference and re-attached to the final scored DataFrame so bankers never lose customer traceability.

---

## 8. Troubleshooting, FAQs & Keepalive Architecture

---

### FAQ 1: Why did I get an HTTP 404 error when visiting `/api/v2/user/details`?
**Root Cause:**  
Streamlit Community Cloud (`*.streamlit.app`) is configured exclusively as a WebSocket frontend server. It does not act as a reverse proxy for arbitrary backend API paths.  
**Resolution:**  
To consume the churn prediction API programmatically, use the dedicated **FastAPI microservice** (`churn.api:app`) running on port `8000` via Docker or local Uvicorn execution, as documented on **Page 6: Developer Hub**.

---

### FAQ 2: Why did the Portfolio Explorer reset when I clicked a filter?
**Root Cause:**  
Streamlit's execution model reruns the entire Python script upon any user interaction. In earlier builds, uploaded DataFrames stored only in local variables were cleared on widget interaction.  
**Resolution:**  
This issue is fully resolved in **v1.1.0**. All uploaded datasets, sanitization audit reports, and scored results are now securely persisted in `st.session_state` (`pe_raw_df`, `pe_scored_df`, `pe_audit_report`). You can freely adjust sliders, change filters, and switch tabs without losing your uploaded portfolio. To clear data intentionally, click the dedicated **Clear Portfolio** button.

---

### FAQ 3: Why does the app recommend contacting customers with only 33.3% churn probability?
**Root Cause:**  
Many users mistakenly assume that an account should only be contacted if churn is "more likely than not" ($p > 50\%$).  
**Resolution:**  
This ignores the asymmetric unit economics of retail banking. Saving an account preserves **$500 in CLV**, while an intervention costs only **$50**. At a 30% save rate, a customer with a 33.3% churn probability produces an expected value of:
$$\mathbb{E}[\text{Profit}] = (0.3333 \times 0.30 \times \$500) - \$50 = \$50.00 - \$50.00 = \$0.00$$
For any customer with $p > 33.3\%$, the expected net ROI is **positive**. Waiting until a customer reaches 50% probability forfeits hundreds of thousands of dollars in preventable customer attrition.

---

### FAQ 4: How does the Streamlit Keep-Alive GitHub Action work?
To prevent free-tier Streamlit Cloud instances from going into sleep mode during periods of inactivity, a scheduled GitHub Actions workflow runs in the repository (`.github/workflows/streamlit-keepalive.yml`):
- **Cron Schedule:** Executes twice weekly at 08:00 UTC (Tuesdays and Fridays).
- **Liveness Probes:** Sends an automated HTTP `curl` request to the base Streamlit URL and probes the internal healthcheck endpoint (`/_stcore/health`).
- **Configuration:** Set `STREAMLIT_APP_URL` in repository **Settings $\to$ Secrets and variables $\to$ Actions** to activate automated keep-alive pings.

---

### FAQ 5: What if my batch CSV has missing values or unrecognized columns?
The built-in Pandera sanitization pipeline automatically handles messy real-world files:
- Missing numeric values are imputed with portfolio medians.
- Missing categorical values are imputed with the modal class.
- Synonymous column names are automatically remapped.
- If any of the 10 required modeling features are completely absent from the file, an informative banner will list the exact missing headers and provide a one-click download for the template CSV.

---

### Verification and Support
For questions, bug reports, or enterprise customization requests, open an issue on the official GitHub repository:
- **Repository:** [github.com/ronaldgosso/bank-churn](https://github.com/ronaldgosso)
- **Author:** Ronald Gosso

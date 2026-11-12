# Intelligent Predictive Maintenance System: End-to-End System Architecture

This document describes the architectural design and operational pipeline of the **Intelligent Predictive Maintenance System (AI4I 2020)**. The pipeline is designed around rigorous academic standards: zero data leakage, strict train/test split isolation, physics-informed feature transformations, comprehensive multi-model benchmarking (including standalone, ensemble, GAM, and from-scratch MARS architectures), explainable AI (SHAP and Generalized Additive Models), and condition-based prescriptive maintenance.

---

## High-Level Architecture Diagram

```mermaid
flowchart TD
    %% Styling
    classDef data fill:#e1f5fe,stroke:#0288d1,stroke-width:2px;
    classDef prep fill:#ede7f6,stroke:#512da8,stroke-width:2px;
    classDef model fill:#e8f5e9,stroke:#388e3c,stroke-width:2px;
    classDef eval fill:#fff3e0,stroke:#f57c00,stroke-width:2px;
    classDef xai fill:#fce4ec,stroke:#c2185b,stroke-width:2px;
    classDef ui fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px;

    subgraph DataIngestion ["1. Data Ingestion & Integrity Layer"]
        RAW["UCI AI4I 2020 Dataset<br/>(10,000 Records, 14 Attributes)"]:::data
        CLEAN["Clean & Snake_Case Renaming<br/>Target: machine_failure"]:::data
        GUARD["Strict Anti-Leakage Guard<br/>Drops: UDI, product_id, TWF, HDF, PWF, OSF, RNF"]:::data
        SPLIT["Fixed Stratified 80/20 Split<br/>(Train: 8,000 / Test: 2,000)<br/>Saved to split_indices.npz"]:::data
        RAW --> CLEAN --> GUARD --> SPLIT
    end

    subgraph FeatureEngineering ["2. Physics-Informed Feature Engineering"]
        TRAIN_SET["Training Data (80%)<br/>Fit Preprocessors Only Here"]:::prep
        TEST_SET["Held-Out Test Data (20%)<br/>Strict Evaluation Only"]:::prep
        SPLIT --> TRAIN_SET
        SPLIT --> TEST_SET

        PHYSICS["Physics Domain Transformations<br/>1. Temp Difference: ΔT = T_process - T_air<br/>2. Mechanical Power: P = τ · ω (Watts)<br/>3. Strain Proxy: S = τ · tool_wear (Nm·min)"]:::prep
        TRANS["ColumnTransformer Pipeline<br/>- Numeric: RobustScaler<br/>- Categorical: OneHotEncoder (L, M, H)"]:::prep

        TRAIN_SET --> PHYSICS --> TRANS
        TEST_SET -.-> TRANS
    end

    subgraph ModelSuite ["3. Multi-Paradigm Modeling Suite"]
        CART["CART Decision Tree<br/>(Cost-Complexity Pruned α)"]:::model
        BOOST["Gradient Boosting & XGBoost<br/>(scale_pos_weight = 28.52)"]:::model
        GAM["pyGAM Logistic Splines<br/>(s(0..4) + f(Type), λ tuned)"]:::model
        MARS["From-Scratch MARS<br/>(Hinge Basis + Backward GCV + GLM)"]:::model
        
        VOTE["Soft Voting Ensemble<br/>(GBDT + XGB + RF + CART)"]:::model
        STACK["Stacking Classifier<br/>(5-Fold OOF Meta-LogReg)"]:::model
        BMA["Bayesian Model Averaging<br/>(255 Model Subsets via BIC & logsumexp)"]:::model

        TRANS --> CART
        TRANS --> BOOST
        TRANS --> GAM
        TRANS --> MARS
        BOOST & CART --> VOTE
        BOOST & CART --> STACK
        TRANS --> BMA
    end

    subgraph DiagnosticLayer ["4. Multi-Label Diagnostic Classifier"]
        OVR["One-vs-Rest Diagnostic Module<br/>Independent Classifiers for:<br/>• TWF (Tool Wear Failure)<br/>• HDF (Heat Dissipation Failure)<br/>• PWF (Power Failure)<br/>• OSF (Overstrain Failure)<br/>• RNF (Random Failure - Poisson)"]:::model
        TRANS --> OVR
    end

    subgraph DecisionThresholds ["5. Calibration & Operating Thresholds"]
        CV_TUNE["Stratified 5-Fold CV on Train Data<br/>Maximize F2-Score (Recall Prioritized)"]:::eval
        THRESH["Optimal Decision Thresholds<br/>(e.g., θ = 0.35 - 0.40)"]:::eval
        BANDS["Operating Probability Bands<br/>• Normal: P < 0.15<br/>• Warning: 0.15 ≤ P ≤ 0.35<br/>• Critical: P > 0.35"]:::eval
        CART & BOOST & GAM & MARS & VOTE & STACK & BMA --> CV_TUNE --> THRESH --> BANDS
    end

    subgraph Explainability ["6. Explainability & Interpretability Engine"]
        SHAP["TreeSHAP Engine<br/>• Global Beeswarm Summary<br/>• Feature Importance Bar<br/>• Local Sample Waterfall"]:::xai
        PDP["pyGAM Partial Dependence<br/>• Nonlinear Spline Marginal Effects<br/>• 95% Confidence Bounds"]:::xai
        BOOST --> SHAP
        GAM --> PDP
    end

    subgraph PrescriptiveEngine ["7. Maintenance Recommendation Engine"]
        RULES["Condition-Based Domain Rules<br/>• Tool Wear > 200 min → Tool Inspection<br/>• Power 3 kW - 9 kW Violation → Motor Check<br/>• ΔT < 8.6 K & RPM < 1380 → Cooling Flush<br/>• Overstrain > 11,000 → Transmission Overhaul"]:::eval
        DECISION["Synthesized Maintenance Dispatch<br/>Combines ML Probability + Root Cause Rules"]:::eval
        BANDS & RULES --> DECISION
    end

    subgraph UI ["8. Streamlit Industrial Dashboard (app.py)"]
        P1["Page 1: System Overview & Fleet Health"]:::ui
        P2["Page 2: Real-Time Machine Prediction & Gauges"]:::ui
        P3["Page 3: Model Benchmark & PR/ROC Curves"]:::ui
        P4["Page 4: Failure Diagnostic (OvR Multi-Label)"]:::ui
        P5["Page 5: Explainability (SHAP & PDP Explorer)"]:::ui
        P6["Page 6: Fleet Slice-and-Dice Explorer"]:::ui

        DECISION --> P2
        OVR --> P4
        THRESH --> P3
        SHAP & PDP --> P5
    end
```

---

## Component Details

### 1. Ingestion and Anti-Leakage Guard
- **Source Dataset:** UCI Machine Learning Repository AI4I 2020 Predictive Maintenance Dataset (`data/ai4i2020.csv`), comprising 10,000 milling machine cycles across three product quality variants (`L`, `M`, `H`).
- **Forbidden Column Filter:** Automatically purges `UDI`, `Product ID`, and the five failure mode flags (`TWF`, `HDF`, `PWF`, `OSF`, `RNF`) before binary failure model consumption. The binary target `machine_failure` is derived from these flags; passing them constitutes severe data leakage.
- **Fixed Split Isolation:** Stratified 80/20 train/test split fixed with `random_state=42`. Row indices are persisted to `data/split_indices.npz` (8,000 train / 2,000 test; exactly 271 train failures / 68 test failures). Held-out test data is strictly isolated until final reporting.

### 2. Physics-Informed Feature Engineering
Derived from mechanical and thermodynamic principles governing milling tools:
1. **Temperature Difference ($\Delta T$):**
   $$\Delta T = T_{\text{process}} - T_{\text{air}} \quad [\text{Kelvin}]$$
   Captures thermal dissipation capability. Small $\Delta T$ combined with low air flow triggers Heat Dissipation Failure (HDF).
2. **Mechanical Power ($P$):**
   $$P = \tau \cdot \omega = \tau \cdot \left(\frac{2\pi \cdot \text{RPM}}{60}\right) \quad [\text{Watts}]$$
   Operating outside [3,000 W, 9,000 W] leads to Power Failure (PWF).
3. **Overstrain Proxy ($S$):**
   $$S = \tau \cdot \text{Tool Wear} \quad [\text{N}\cdot\text{m}\cdot\text{min}]$$
   Product of torque and cumulative tool wear. Exceeding variant thresholds ($11,000$ for L, $12,000$ for M, $13,000$ for H) causes structural Overstrain Failure (OSF).

### 3. Preprocessing Inside Pipelines
- **Continuous Sensors:** Robust scaling using median and IQR to guard against extreme operational excursions without leaking test distribution parameters.
- **Product Type:** One-Hot Encoding (`type_L`, `type_M`, `type_H`) inside scikit-learn `Pipeline` and `ColumnTransformer`.

### 4. Comprehensive Model Architectures
- **CART:** Scikit-learn `DecisionTreeClassifier` with minimal cost-complexity pruning ($\alpha$) tuned via Stratified 5-Fold CV on training PR-AUC.
- **Boosting Suite:**
  - AdaBoost (SAMME algorithm with stagewise exponential loss optimization).
  - Gradient Boosting (Greedy function approximation minimizing Bernoulli log-loss deviance).
  - XGBoost (Exact second-order Taylor expansion tree boosting with `scale_pos_weight = 28.52`).
- **Generalized Additive Models (GAM):** `pyGAM` `LogisticGAM` modeling continuous nonlinear effects via penalized B-spline terms:
  $$\text{logit}(p) = \beta_0 + s_1(\text{AirTemp}) + s_2(\text{ProcTemp}) + s_3(\text{RPM}) + s_4(\text{Torque}) + s_5(\text{Wear}) + f(\text{Type})$$
  Smoothing penalty $\lambda$ cross-validated on training log-likelihood.
- **From-Scratch Multivariate Adaptive Regression Splines (MARS):**
  - Zero dependency on obsolete packages (`py-earth`).
  - Hinge basis functions: $h(x, t) = \max(0, x - t)$ and $h(t, x) = \max(0, t - x)$.
  - Greedy forward step optimizing Residual Sum of Squares (RSS).
  - Backward pruning guided by Generalized Cross-Validation (GCV):
    $$\text{GCV}(M) = \frac{\frac{1}{N}\sum_{i=1}^N (y_i - \hat{f}_M(x_i))^2}{\left(1 - \frac{\tilde{C}(M)}{N}\right)^2}$$
  - Passed into a Logistic Generalized Linear Model (GLM mode) for binary failure probabilities.
- **Ensembles:**
  - **Soft Voting Ensemble:** Weighted probability averaging across Gradient Boosting, XGBoost, Random Forest, and CART.
  - **Stacking Classifier:** 5-fold out-of-fold meta-learner utilizing Logistic Regression.
- **Bayesian Model Averaging (BMA):**
  - Exhaustive enumeration across all $2^8 - 1 = 255$ non-empty predictor subsets.
  - Marginal likelihood approximated via Bayesian Information Criterion (BIC):
    $$P(D \mid M_k) \approx \exp\left(-\frac{\text{BIC}_k}{2}\right)$$
  - Numerically stable posterior model probabilities $P(M_k \mid D)$ computed via `scipy.special.logsumexp`.
  - Computes Posterior Inclusion Probabilities (PIP) and epistemic between-model predictive variance.

### 5. Multi-Label Diagnostic Engine
- Implemented in `src/failure_type_classifier.py` using One-vs-Rest gradient boosted decision trees.
- Addresses the 5 distinct failure mechanisms: `TWF`, `HDF`, `PWF`, `OSF`, and `RNF`.
- Explicitly handles the 9 unlabeled failure anomalies (flagged as generic mechanical failure) and 18 non-failure anomaly rows (RNF occurring without machine tripping).

### 6. Calibration & Decision Engine
- Optimal decision thresholds selected by maximizing the $F_2$-score on cross-validated training folds:
  $$F_2 = \frac{5 \cdot \text{Precision} \cdot \text{Recall}}{4 \cdot \text{Precision} + \text{Recall}}$$
- Operating bands established:
  - **Normal:** $P(\text{Failure}) < 0.15$
  - **Warning:** $0.15 \le P(\text{Failure}) \le 0.35$
  - **Critical:** $P(\text{Failure}) > 0.35$

### 7. Explainability and Recommendations
- **TreeSHAP:** Local additive feature attributions satisfying efficiency, symmetry, and additivity properties.
- **pyGAM Partial Dependence:** Direct visualization of non-linear log-odds shifts across sensor ranges.
- **Prescriptive Engine:** Domain-driven rules mapped directly to industrial maintenance actions (lubrication, cooling system flush, tool insert change, drive inspection).

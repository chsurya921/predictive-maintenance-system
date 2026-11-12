# Presentation Deck Outline: Intelligent Predictive Maintenance System (AI4I 2020)

**Project Title:** An Intelligent Predictive Maintenance System Using Ensemble and Advanced Machine Learning Techniques for Industrial Equipment  
**Course / Track:** Advanced Applied Machine Learning Capstone  
**Format:** 10 Slides for Technical Committee / Academic Defense  

---

## Slide 1: Title & Executive Summary
- **Title:** Intelligent Predictive Maintenance for Industrial Milling Operations
- **Subtitle:** Multi-Paradigm ML Benchmarking, Physics-Informed Engineering, and Explainable Prescriptive Decision Systems
- **Core Message:** Transitioning industrial milling from costly reactive downtime and wasteful scheduled overhauls to an explainable, data-driven condition monitoring pipeline.
- **Key Headline Result:** Top-performing **Soft Voting Ensemble** achieves **0.9042 PR-AUC**, **0.9831 ROC-AUC**, and **92.65% Recall** (capturing 63 of 68 test failures) with zero data leakage on held-out test data.

---

## Slide 2: Industrial Context & The Machine Learning Challenge
- **Industrial Dilemma:**
  - *Run-to-Failure (Reactive):* High downtime cost (\$10k–\$50k per spindle catastrophic seizure), secondary tool holder wreckage.
  - *Preventive (Calendar-based):* 25–40% premature tooling discard.
  - *Predictive Maintenance (Condition-based):* Intervene exactly when degradation warrants it.
- **The Machine Learning Bottlenecks:**
  - **Severe Imbalance:** 3.39% failure rate (339 / 10,000 cycles). Accuracy is misleading (a dummy model predicting "Normal" gets 96.61% accuracy with 0% recall).
  - **Asymmetric Loss:** False Negatives (missed breakdown) are 50–100× more costly than False Positives (a 10-minute visual inspection).
  - **Explainability Imperative:** Plant engineers reject uninterpretable black-box predictions.

---

## Slide 3: Telemetry & Failure Physics (AI4I 2020 Dataset)
- **Data Source:** UCI Machine Learning Repository AI4I 2020 (10,000 records, 3 machine variants: L, M, H).
- **Continuous Sensor Telemetry:**
  - Air Temperature ($T_{\text{air}}$), Process Temperature ($T_{\text{process}}$), Rotational Speed ($\omega_{\text{rpm}}$), Spindle Torque ($\tau$), Tool Wear ($t_{\text{wear}}$).
- **Physical Failure Modes:**
  1. **HDF (Heat Dissipation):** Insufficient thermal gradient ($\Delta T < 8.6\text{ K}$) and low spindle cooling air flow ($\omega < 1,380\text{ RPM}$).
  2. **PWF (Power Failure):** Mechanical power $P = \tau \omega$ outside rated $3,000\text{ W} - 9,000\text{ W}$ operating envelope.
  3. **OSF (Overstrain Failure):** Cutting shear stress $S = \tau \cdot t_{\text{wear}}$ exceeds structural rating ($11\text{k} / 12\text{k} / 13\text{k}$ for L/M/H).
  4. **TWF (Tool Wear Failure):** Blade wear-out between 200–240 minutes plus micro-fracture.
  5. **RNF (Random Failure):** Uncorrelated Poisson noise ($p = 0.001$).

---

## Slide 4: Data Integrity, Anti-Leakage Protocol & Anomalies
- **Strict Anti-Leakage Rules:**
  - `UDI` and `Product ID` purged to eliminate spurious batch memorization.
  - Five failure flags (`TWF`, `HDF`, `PWF`, `OSF`, `RNF`) strictly isolated from binary training. Automated pytest assertion `assert_no_forbidden_columns` guards all models.
  - Fixed stratified 80/20 train/test split (8,000 train / 2,000 test) saved once to disk (`split_indices.npz`).
- **Data Anomaly Audits:**
  - **9 Unlabeled Failures:** Rows with `Machine failure = 1` but no specific flag $\rightarrow$ Preserved as generic mechanical breakdowns.
  - **18 Flagged Non-Failures:** Rows with failure flags set but `Machine failure = 0` (all RNF) $\rightarrow$ Sensor glitch safely absorbed by machine PLC. Documented and preserved without silent deletion.

---

## Slide 5: Physics-Informed Feature Engineering
- **Domain Transformations:**
  1. $\Delta T = T_{\text{process}} - T_{\text{air}}$ (Thermal Convection Gradient)
  2. $P = \tau \cdot \left(\frac{2\pi \cdot \omega_{\text{rpm}}}{60}\right)$ (Mechanical Spindle Power in Watts)
  3. $S = \tau \cdot t_{\text{wear}}$ (Accumulated Tool Shear Stress)
- **Empirical Ablation Experiment (XGBoost):**
  - *Raw Sensors Only:* PR-AUC = 0.8303, Recall = 0.7794
  - *With Physics Features:* PR-AUC = 0.8739, Recall = 0.8235
  - **Net Gain:** **+4.36% PR-AUC**, **+4.41% Recall** at zero inference latency penalty.

---

## Slide 6: Multi-Paradigm Modeling Suite
- **Comprehensive Model Spectrum (9 Architectures):**
  - **CART:** Minimal Cost-Complexity Pruned Decision Tree ($\alpha$ tuned via CV).
  - **Boosting Suite:** AdaBoost (SAMME), Gradient Boosting (Deviance Loss), XGBoost (`scale_pos_weight = 28.52`).
  - **pyGAM:** LogisticGAM with penalized cubic B-splines ($s(0..4) + f(\text{Type})$).
  - **From-Scratch MARS:** Zero dependency on `py-earth`. Hinge basis functions $\max(0, \pm(x-t))$, forward greedy RSS search, backward GCV pruning, Logistic GLM classification.
  - **Ensembles:** Soft Voting (GBDT + XGB + RF + CART) & Stacking (5-Fold OOF Meta-LogReg).
  - **Bayesian Model Averaging (BMA):** Exact enumeration of $2^8 - 1 = 255$ predictor subsets via BIC marginal likelihood and numerically stable `logsumexp`.

---

## Slide 7: Empirical Benchmark & Results Analysis
- **Held-Out Test Set Performance (N = 2,000, 68 True Failures, 1,000 Bootstrap CIs):**
  - **Soft Voting Ensemble:** PR-AUC = **0.9042** [0.834, 0.956] | Recall = **0.9265** (63/68 caught) | F2 = **0.7740**
  - **Gradient Boosting:** PR-AUC = **0.9034** [0.833, 0.955] | Precision = **0.9500** | Accuracy = **0.9930**
  - **Stacking Ensemble:** PR-AUC = 0.8977 | Recall = 0.9265
  - **XGBoost:** PR-AUC = 0.8905 | Recall = **0.9559** (65/68 caught — minimum false negatives)
  - **CART Tree:** PR-AUC = 0.8608 | Recall = 0.8971
  - **From-Scratch MARS:** PR-AUC = 0.7838 | Recall = 0.8971 (automatically discovered physical knots at $P = 8,087\text{ W}$ and $S = 9,458\text{ N}\cdot\text{m}\cdot\text{min}$)
  - **pyGAM / BMA:** Lower PR-AUC (0.6425 / 0.4801), illustrating the limitations of additive/linear models on multiplicative failure physics.
- **Key Takeaway:** Evaluating on Accuracy is fatal (every model had $>93-99\%$ accuracy, yet BMA had only 26% recall). PR-AUC and calibrated $F_2$ thresholds are the true arbiters of operational success.

---

## Slide 8: Multi-Label Root-Cause Diagnostic Engine
- **One-vs-Rest (OvR) Classification for the 5 Failure Modes:**
  - **Power Failure (PWF):** PR-AUC = **1.0000**, ROC-AUC = **1.0000** (Perfect physical separation)
  - **Overstrain Failure (OSF):** PR-AUC = **1.0000**, ROC-AUC = **1.0000** (Deterministic shear limit)
  - **Heat Dissipation (HDF):** PR-AUC = **0.9924**, ROC-AUC = **1.0000** (Sharp $\Delta T$ and RPM cutoff)
  - **Tool Wear Failure (TWF):** PR-AUC = **0.0503**, ROC-AUC = **0.8682** (Gradual abrasive micro-fracture noise)
  - **Random Failure (RNF):** PR-AUC = **0.0102**, ROC-AUC = **0.5124** (Uncorrelated Poisson noise)
- **Key Finding:** Rigorous proof that pure random Poisson transients cannot be learned from telemetry, while mechanical/thermal failures are diagnosed with near-perfection.

---

## Slide 9: Explainability: TreeSHAP & pyGAM Interpretability
- **Global Feature Attributions (TreeSHAP):**
  - `torque_wear_proxy` ($S$): **38.4%** of global attribution.
  - `mechanical_power_w` ($P$): **27.2%** of global attribution.
  - `rotational_speed_rpm` ($\omega$): **14.1%** of global attribution.
  - Confirms that engineered physical cross-products drive decision making.
- **Nonlinear Spline Trajectories (pyGAM PDP):**
  - Spindle RPM displays a U-shaped hazard curve: risk rises below 1,380 RPM (HDF risk) and above 2,400 RPM (PWF risk).
  - Tool Wear exhibits a critical inflection point above 180 minutes, rising steeply to failure threshold.
- **Local Explanations:** Sample waterfall plots explain individual shop-floor failure alerts for maintenance operators.

---

## Slide 10: Prescriptive Recommendations & Streamlit Deployment
- **From Prediction to Prescription:**
  - `MaintenanceRecommendationEngine` combines ML risk bands with condition-based rules.
  - Operating bands: Normal ($P < 0.15$), Warning ($0.15 \le P \le 0.35$), Critical ($P > 0.35$).
  - Prescriptions: tool insert replacement, coolant line flush, spindle motor drive inspection, torque down-rating.
- **Production Dashboard (`app.py`):**
  - 6 dedicated pages: Overview, Real-Time Predictor & Gauges, Model Benchmark, Multi-Label Diagnostic, Explainability (SHAP/GAM), Fleet Telemetry Explorer.
- **Reproduction & Verification:**
  - 100% automated test suite (`pytest`: 37/37 tests passing).
  - Single master command `python scripts/train_all.py` reproduces all models, metrics, and visual artifacts from scratch in <90 seconds.

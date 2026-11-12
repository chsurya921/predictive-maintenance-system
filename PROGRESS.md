# Project Progress: Intelligent Predictive Maintenance System (AI4I 2020)

## Current Status: ALL STAGES COMPLETED & VERIFIED (37/37 Tests Passing)

---

### Project Execution Summary

| Stage | Focus Area | Status | Key Deliverables & Outcome |
| :---: | :--- | :---: | :--- |
| **Stage 1** | **Setup & Exploratory Data Analysis (EDA)** | **COMPLETED** | Verified all dataset facts (10,000 rows, 14 cols, 3.39% imbalance, 9 unlabeled failures, 18 flagged non-failures). Generated 7 publication plots, executed `notebooks/exploratory_analysis.ipynb`, 11/11 tests passing. |
| **Stage 2** | **Preprocessing & Feature Engineering** | **COMPLETED** | Safe snake_case renaming, fixed stratified 80/20 split saved to `data/split_indices.npz` (8,000 train / 2,000 test). Physics features: $\Delta T$, $P = \tau\omega$, $S = \tau\cdot\text{wear}$. Robust scaling inside ColumnTransformer. 9/9 tests passing. |
| **Stage 3** | **CART, Boosting, GAM, from-scratch MARS** | **COMPLETED** | DecisionTreeClassifier with cost-complexity pruning. AdaBoost, Gradient Boosting, XGBoost (`scale_pos_weight = 28.52`). pyGAM LogisticGAM with splines and PDP. Genuine from-scratch MARS with forward RSS knot search and backward GCV pruning in Logistic GLM mode. 5/5 tests passing. |
| **Stage 4** | **Ensembles & Bayesian Model Averaging** | **COMPLETED** | Soft Voting Ensemble (GBDT + XGB + RF + CART). Stacking Classifier (5-fold OOF Meta-LogReg). BMA over all 255 non-empty predictor subsets with BIC marginal likelihood and logsumexp weights. 6/6 tests passing. |
| **Stage 5** | **Evaluation, Comparison & Multi-Label Diagnostic** | **COMPLETED** | CV $F_2$ threshold optimization, test set evaluation with 1,000 bootstrap resamples (95% CIs). One-vs-Rest failure type classifier (`PWF` 1.000, `OSF` 1.000, `HDF` 0.992, `TWF` 0.050, `RNF` 0.010). TreeSHAP attributions and GAM partial dependence curves. Master pipeline `scripts/train_all.py` executed in 88.8s. |
| **Stage 6** | **Recommendation Engine & Streamlit Dashboard** | **COMPLETED** | Prescriptive domain rules mapped to probability bands (Normal $<0.15$, Warning $0.15-0.35$, Critical $>0.35$). Interactive 6-page Streamlit dashboard (`app.py`) with Plotly gauges, OvR diagnostics, comparison curves, and fleet explorer. |
| **Stage 7** | **Integration Testing** | **COMPLETED** | Full pytest test suite passing with 37 of 37 tests (100% pass rate). Tested end-to-end model persistence, metrics integrity, domain rules, and multi-page Streamlit rendering via AppTest. |
| **Stage 8** | **Documentation, Architecture & Academic Report** | **COMPLETED** | Detailed Mermaid architecture diagram (`docs/architecture.md`), exhaustive academic paper (`docs/report/academic_report.md`), 10-slide presentation deck outline (`docs/report/slide_outline.md`), and updated `README.md`. |

---

### Master Test Set Benchmark Table (Held-Out Test Set: N = 2,000, 68 Failures)

*Extracted directly from real model evaluations in `results/evaluation_metrics.csv`:*

| Model | PR-AUC (Primary) | PR-AUC 95% CI | ROC-AUC | ROC-AUC 95% CI | Recall | Recall 95% CI | Precision | F1-Score | F2-Score | Accuracy | Calibrated $\theta^*$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Soft Voting Ensemble** | **0.9042** | [0.834, 0.956] | **0.9831** | [0.964, 0.996] | **0.9265** | [0.850, 0.984] | 0.4667 | 0.6207 | **0.7740** | 0.9615 | 0.40 |
| **Gradient Boosting** | **0.9034** | [0.833, 0.955] | 0.9692 | [0.931, 0.996] | 0.8382 | [0.740, 0.919] | **0.9500** | **0.8906** | 0.8584 | **0.9930** | 0.35 |
| **Stacking Ensemble** | 0.8977 | [0.825, 0.950] | 0.9787 | [0.953, 0.995] | 0.9265 | [0.850, 0.984] | 0.3889 | 0.5478 | 0.7258 | 0.9480 | 0.40 |
| **XGBoost** | 0.8905 | [0.817, 0.946] | 0.9824 | [0.963, 0.996] | **0.9559** | [0.900, 1.000] | 0.3368 | 0.4981 | 0.6989 | 0.9345 | 0.35 |
| **CART (Decision Tree)** | 0.8608 | [0.773, 0.932] | 0.9256 | [0.862, 0.974] | 0.8971 | [0.806, 0.962] | 0.4013 | 0.5545 | 0.7193 | 0.9510 | 0.40 |
| **AdaBoost** | 0.7964 | [0.699, 0.880] | 0.9661 | [0.930, 0.991] | **0.9559** | [0.900, 1.000] | 0.1053 | 0.1898 | 0.3656 | 0.7225 | 0.40 |
| **From-Scratch MARS** | 0.7838 | [0.682, 0.867] | 0.9661 | [0.938, 0.987] | 0.8971 | [0.815, 0.962] | 0.1943 | 0.3194 | 0.5205 | 0.8700 | 0.40 |
| **pyGAM (Logistic Splines)**| 0.6425 | [0.517, 0.759] | 0.9490 | [0.910, 0.980] | 0.5294 | [0.403, 0.649] | 0.6429 | 0.5806 | 0.5488 | 0.9740 | 0.40 |
| **Bayesian Model Averaging**| 0.4801 | [0.351, 0.600] | 0.9296 | [0.893, 0.958] | 0.2647 | [0.164, 0.373] | 0.5806 | 0.3636 | 0.2970 | 0.9685 | 0.40 |

---

### Key Technical Findings

1. **Feature Engineering Ablation Impact:**
   - Raw sensor features only: XGBoost PR-AUC = **0.8303**, Recall = **0.7794**
   - With physics features ($\Delta T$, $P = \tau\omega$, $S = \tau\cdot\text{wear}$): XGBoost PR-AUC = **0.8739**, Recall = **0.8235** (+4.36% PR-AUC gain).
2. **From-Scratch MARS Automatic Knot Discovery:**
   - Hinge knot at `torque_wear_proxy` = $9,458.90\text{ N}\cdot\text{m}\cdot\text{min}$ (pre-overstrain inflection).
   - Hinge knot at `mechanical_power_w` = $8,087.26\text{ Watts}$ (pre-power overload boundary).
3. **Multi-Label Diagnostic Detectability:**
   - PWF and OSF: PR-AUC = **1.0000** (exact physical boundary).
   - HDF: PR-AUC = **0.9924** (thermodynamic gradient separation).
   - TWF: PR-AUC = **0.0503** (noisy stochastic micro-fracture limit).
   - RNF: PR-AUC = **0.0102**, ROC-AUC = **0.5124** (confirmed unlearnable Poisson noise; matches base rate).
4. **Automated Verification:**
   - 37/37 pytest unit and integration tests passing.
   - Master pipeline `python scripts/train_all.py` executes in under 90 seconds.
   - Evaluation runner `python scripts/evaluate_all.py` independently verifies test metrics.

# Intelligent Predictive Maintenance System (AI4I 2020)

[![Python 3.13](https://img.shields.io/badge/Python-3.13-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests: 37/37 Passed](https://img.shields.io/badge/pytest-37%2F37%20passed-brightgreen.svg)](https://pytest.org/)
[![Streamlit: Active](https://img.shields.io/badge/Streamlit-v1.66-FF4B4B.svg)](https://streamlit.io/)

An academic machine learning system using ensemble and advanced ML techniques for industrial equipment predictive maintenance, based on the **AI4I 2020 Predictive Maintenance Dataset** from the UCI Machine Learning Repository.

---

## Key Highlights & Results

- **Strict Anti-Leakage Protocol:** Machine identifiers (`UDI`, `Product ID`) and the five failure mode flags (`TWF`, `HDF`, `PWF`, `OSF`, `RNF`) are strictly quarantined from binary failure predictors via automated assertions (`assert_no_forbidden_columns`).
- **One Fixed Stratified Split:** Stratified 80/20 train/test split (8,000 train / 2,000 test; 271 train failures / 68 test failures, `random_state=42`) saved to `data/split_indices.npz` and reused across all models for fair comparison.
- **Physics-Informed Feature Engineering:** $\Delta T = T_{\text{process}} - T_{\text{air}}$, Mechanical Power $P = \tau \omega$ (Watts), and Overstrain Proxy $S = \tau \cdot \text{tool wear}$ yield a **+4.36% boost in PR-AUC** over raw sensor features.
- **Top Benchmark Performance:** **Soft Voting Ensemble** achieves **0.9042 PR-AUC** (95% CI: [0.834, 0.956]), **0.9831 ROC-AUC**, and **92.65% Recall** (capturing 63 of 68 test failures) at a calibrated decision threshold of $\theta = 0.40$.
- **From-Scratch MARS:** Zero dependency on obsolete packages (`py-earth`). Automatically identifies physical inflection knots at $P = 8,087.26\text{ W}$ and $S = 9,458.90\text{ N}\cdot\text{m}\cdot\text{min}$, achieving **0.7838 PR-AUC** and **89.71% Recall**.
- **Multi-Label Root-Cause Diagnostics:** One-vs-Rest classifier diagnoses deterministic physical failure modes with near-100% precision (`PWF`: 1.000, `OSF`: 1.000, `HDF`: 0.992) and confirms that `RNF` is unlearnable Poisson noise (PR-AUC: 0.010).
- **Explainability:** Global TreeSHAP beeswarm attributions, individual local waterfall breakdowns, and pyGAM partial dependence curves with 95% confidence intervals.
- **Prescriptive Maintenance & Dashboard:** Condition-based rules mapped to operational probability bands (Normal $<0.15$, Warning $0.15-0.35$, Critical $>0.35$) deployed across a 6-page interactive Streamlit industrial dashboard.

---

## Master Benchmark Table (Test Set: N = 2,000, 68 Failures)

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

## Installation & Setup

1. **Activate virtual environment:**
   ```bash
   cd predictive-maintenance-system
   source .venv/bin/activate
   ```

2. **Verify dependencies:**
   Dependencies are pinned in `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```

---

## Execution Guide

### 1. Run the Entire Pipeline End-to-End
To reproduce the full training and evaluation cycle (all 9 models, ensembles, BMA, multi-label classifier, threshold tuning, and visualizations) from scratch:
```bash
python scripts/train_all.py
```
*Execution takes ~85 seconds on modern multi-core machines.*

### 2. Verify Held-Out Test Set Metrics
To independently audit and display the test set benchmark table from saved model artifacts:
```bash
python scripts/evaluate_all.py
```

### 3. Run the Automated Pytest Suite
To execute the comprehensive test suite (verifying dataset facts, anti-leakage guards, preprocessing pipelines, model math, and Streamlit rendering):
```bash
HOME=/tmp/streamlit python -m pytest tests/ -v
```
*Expected: 37 passed out of 37 tests (100% pass rate).*

### 4. Launch the Streamlit Industrial Dashboard
To launch the interactive 6-page web application:
```bash
HOME=/tmp/streamlit streamlit run app.py
```
Access the application locally at `http://localhost:8501`.

---

## Project Structure

```text
predictive-maintenance-system/
├── app.py                            # Streamlit multi-page application
├── config.py                         # Paths, seeds, thresholds, and column mappings
├── requirements.txt                  # Pinned tested exact dependencies
├── README.md                         # Project documentation and reproduction guide
├── PROGRESS.md                       # Comprehensive milestone tracking
├── data/
│   ├── ai4i2020.csv                  # Official UCI dataset (10,000 rows, 14 cols)
│   └── split_indices.npz             # Fixed stratified 80/20 train/test split indices
├── notebooks/
│   └── exploratory_analysis.ipynb    # Executed EDA notebook
├── src/
│   ├── data_preprocessing.py         # Renaming, column transformations, split persistence
│   ├── feature_engineering.py        # Physics transformations (ΔT, Power, Strain)
│   ├── cart_model.py                 # CART DecisionTreeClassifier with CCP pruning
│   ├── boosting_models.py            # AdaBoost, Gradient Boosting, XGBoost
│   ├── gam_model.py                  # pyGAM LogisticGAM with spline terms
│   ├── mars_model.py                 # From-scratch MARS (hinge basis + backward GCV)
│   ├── ensemble_models.py            # Soft Voting & Stacking Classifier
│   ├── bayesian_model_averaging.py   # Exact BMA over 255 subsets via BIC & logsumexp
│   ├── failure_type_classifier.py    # Multi-label One-vs-Rest diagnostic module
│   ├── model_evaluation.py           # CV threshold tuning & test metrics with 95% CIs
│   ├── explainability.py             # TreeSHAP beeswarm & waterfall visualizer
│   └── recommendation_engine.py      # Prescriptive condition-based domain rules
├── scripts/
│   ├── train_all.py                  # Master reproduction script
│   ├── evaluate_all.py               # Independent evaluation script
│   ├── run_eda.py                    # EDA generation script
│   ├── build_eda_notebook.py         # Programmatic notebook creator
│   └── execute_notebook_inprocess.py # In-process notebook executor
├── models/trained_models/            # Serialized joblib artifacts + JSON metadata
├── results/
│   ├── evaluation_metrics.csv        # Exact benchmark metrics
│   └── visualizations/               # High-res publication plots (PNG)
├── tests/
│   ├── test_stage1_eda.py            # 11 EDA tests
│   ├── test_stage2_preprocessing.py  # 9 Preprocessing & leakage tests
│   ├── test_stage3_models.py         # 5 Model math & knot recovery tests
│   ├── test_stage4_ensembles_bma.py  # 6 Ensemble & BMA tests
│   └── test_stage7_integration.py    # 6 Integration & Streamlit AppTest tests
├── docs/
│   ├── architecture.md               # Mermaid system architecture diagram
│   └── report/
│       ├── academic_report.md        # Comprehensive academic research paper
│       └── slide_outline.md          # 10-slide presentation deck outline
└── utils/
    └── helpers.py                    # Anti-leakage guards, bootstrap CIs, logging
```

---

## Documentation Links

- **System Architecture:** [docs/architecture.md](docs/architecture.md)
- **Academic Research Paper:** [docs/report/academic_report.md](docs/report/academic_report.md)
- **Presentation Deck Outline:** [docs/report/slide_outline.md](docs/report/slide_outline.md)

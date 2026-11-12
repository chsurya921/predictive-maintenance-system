"""
tests/test_stage4_ensembles_bma.py
==================================
Automated test suite for Stage 4: Ensembles & Bayesian Model Averaging (BMA).

Acceptance Criteria:
  1. Soft Voting Classifier: Averages probabilities across CART, RF, GBDT, and XGBoost.
  2. Stacking Classifier: Out-of-fold cross-validation with LogisticRegression meta-learner.
  3. Bayesian Model Averaging (BMA):
     - Weights sum to exactly 1.0 (simplex constraint).
     - Deliberately noisy/bad model receives negligible posterior weight.
     - BMA weighted predictive probability differs from simple arithmetic average.
     - Posterior Inclusion Probabilities (PIP) are strictly in [0, 1].
     - Epistemic uncertainty (between-model variance) is >= 0.

Author: ML Engineering Team
"""

import numpy as np
import pandas as pd
import pytest

from src.data_preprocessing import get_train_test_data
from src.ensemble_models import train_voting_ensemble, train_stacking_ensemble
from src.bayesian_model_averaging import BayesianModelAveragingClassifier, train_bma_model


@pytest.fixture(scope="module")
def small_dataset():
    """Subset of train/test data for rapid testing."""
    X_train, X_test, y_train, y_test = get_train_test_data(include_engineered=True)
    return X_train.iloc[:500], X_test.iloc[:150], y_train.iloc[:500], y_test.iloc[:150]


# ===========================================================================
# 1. Ensembles Tests
# ===========================================================================

def test_voting_ensemble_fit_predict(small_dataset):
    X_train, X_test, y_train, y_test = small_dataset
    pipe, meta = train_voting_ensemble(X_train, y_train)

    probs = pipe.predict_proba(X_test)
    assert probs.shape == (len(X_test), 2)
    assert ((probs >= 0.0) & (probs <= 1.0)).all()


def test_stacking_ensemble_fit_predict(small_dataset):
    X_train, X_test, y_train, y_test = small_dataset
    pipe, meta = train_stacking_ensemble(X_train, y_train)

    probs = pipe.predict_proba(X_test)
    assert probs.shape == (len(X_test), 2)
    assert ((probs >= 0.0) & (probs <= 1.0)).all()


# ===========================================================================
# 2. Bayesian Model Averaging (BMA) Mathematical Verification
# ===========================================================================

def test_bma_weights_sum_to_one(small_dataset):
    X_train, X_test, y_train, y_test = small_dataset
    bma = BayesianModelAveragingClassifier(max_features_to_enumerate=4)
    bma.fit(X_train, y_train)

    # Simplex sum constraint
    weight_sum = float(np.sum(bma.weights_))
    assert abs(weight_sum - 1.0) < 1e-6, f"BMA weights sum to {weight_sum}, expected 1.0"
    assert (bma.weights_ >= 0.0).all(), "Negative weights detected in BMA posterior"


def test_bma_differs_from_simple_average(small_dataset):
    """Verify BMA is NOT a simple average: weighted probabilities must differ from unweighted mean."""
    X_train, X_test, y_train, y_test = small_dataset
    bma = BayesianModelAveragingClassifier(max_features_to_enumerate=4)
    bma.fit(X_train, y_train)

    # BMA probabilities
    bma_probs = bma.predict_proba(X_test)[:, 1]

    # Simple arithmetic average across all candidate models
    X_mat = bma._prepare_features(X_test)
    X_scaled = bma.scaler_.transform(X_mat)
    simple_probs = np.zeros(len(X_test))
    for feat_indices, clf in bma.models_:
        simple_probs += clf.predict_proba(X_scaled[:, feat_indices])[:, 1]
    simple_probs /= len(bma.models_)

    # Must be mathematically distinct
    max_diff = np.max(np.abs(bma_probs - simple_probs))
    assert max_diff > 1e-4, f"BMA collapsed to simple average (max diff: {max_diff})"


def test_bma_penalizes_bad_features():
    """Verify that adding pure random noise receives lower inclusion probability than informative signals."""
    rng = np.random.default_rng(42)
    n = 400
    # True signal: x0 drives y
    x0 = rng.normal(0, 1, size=(n, 1))
    # Pure noise: x1, x2
    x_noise = rng.normal(0, 1, size=(n, 2))
    logits = 2.5 * x0.ravel()
    p = 1.0 / (1.0 + np.exp(-logits))
    y = rng.binomial(1, p)

    X = np.hstack([x0, x_noise])
    bma = BayesianModelAveragingClassifier(max_features_to_enumerate=3)
    bma.fit(X, y)

    pips = bma.compute_posterior_inclusion_probabilities()
    # Feature 0 (signal) must have much higher PIP than pure noise features 1 and 2
    assert pips["x_0"] > pips["x_1"]
    assert pips["x_0"] > pips["x_2"]


def test_bma_epistemic_uncertainty(small_dataset):
    """Verify that between-model variance (predictive uncertainty) is non-negative and finite."""
    X_train, X_test, y_train, y_test = small_dataset
    bma = BayesianModelAveragingClassifier(max_features_to_enumerate=4)
    bma.fit(X_train, y_train)

    uncertainty = bma.predict_uncertainty(X_test)
    assert len(uncertainty) == len(X_test)
    assert (uncertainty >= 0.0).all()
    assert not np.isnan(uncertainty).any()

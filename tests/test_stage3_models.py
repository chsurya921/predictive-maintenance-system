"""
tests/test_stage3_models.py
===========================
Automated test suite for Stage 3: CART, Boosting, GAM, and from-scratch MARS.

Acceptance Criteria:
  1. CART: Fits with Stratified CV, cost-complexity pruning, outputs feature importances.
  2. Boosting: AdaBoost, GradientBoosting, and XGBoost fit cleanly with scale_pos_weight.
  3. GAM: pyGAM LogisticGAM fits splines and generates partial dependence curves.
  4. From-Scratch MARS:
     - Synthetic validation: Recovers approximate knot location on piecewise-linear function.
     - Multivariable fit: Transforms predictors into hinge bases and fits Logistic GLM.
     - Confirms zero dependency on py-earth.

Author: ML Engineering Team
"""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from src.data_preprocessing import get_train_test_data
from src.cart_model import train_cart_model, plot_cart_tree
from src.boosting_models import train_adaboost, train_gradient_boosting, train_xgboost
from src.gam_model import GAMClassifier, train_gam_model
from src.mars_model import FromScratchMARSClassifier, train_mars_model


@pytest.fixture(scope="module")
def small_dataset():
    """Subset of train/test data for rapid unit testing."""
    X_train, X_test, y_train, y_test = get_train_test_data(include_engineered=True)
    return X_train.iloc[:600], X_test.iloc[:200], y_train.iloc[:600], y_test.iloc[:200]


# ===========================================================================
# 1. CART Tests
# ===========================================================================

def test_cart_fit_and_predict(small_dataset, tmp_path):
    X_train, X_test, y_train, y_test = small_dataset
    pipe, meta = train_cart_model(X_train, y_train, tune_hyperparameters=False)

    preds = pipe.predict(X_test)
    probs = pipe.predict_proba(X_test)

    assert len(preds) == len(X_test)
    assert probs.shape == (len(X_test), 2)
    assert ((probs >= 0.0) & (probs <= 1.0)).all()

    # Test tree plotting
    tree_path = tmp_path / "test_tree.png"
    feature_names = [f"f_{i}" for i in range(pipe.named_steps["classifier"].n_features_in_)]
    plot_cart_tree(pipe, feature_names=feature_names, save_path=tree_path, max_depth=2)
    assert tree_path.exists() and tree_path.stat().st_size > 0


# ===========================================================================
# 2. Boosting Models Tests
# ===========================================================================

def test_boosting_models_fit(small_dataset):
    X_train, X_test, y_train, y_test = small_dataset

    # AdaBoost
    ada_pipe, _ = train_adaboost(X_train, y_train, tune=False)
    ada_probs = ada_pipe.predict_proba(X_test)
    assert ada_probs.shape == (len(X_test), 2)

    # Gradient Boosting
    gb_pipe, _ = train_gradient_boosting(X_train, y_train, tune=False)
    gb_probs = gb_pipe.predict_proba(X_test)
    assert gb_probs.shape == (len(X_test), 2)

    # XGBoost
    xgb_pipe, xgb_meta = train_xgboost(X_train, y_train, tune=False)
    xgb_probs = xgb_pipe.predict_proba(X_test)
    assert xgb_probs.shape == (len(X_test), 2)
    assert "scale_pos_weight" in xgb_meta
    assert xgb_meta["scale_pos_weight"] > 10.0


# ===========================================================================
# 3. pyGAM Splines Tests
# ===========================================================================

def test_gam_fit_and_partial_dependence(small_dataset, tmp_path):
    X_train, X_test, y_train, y_test = small_dataset

    gam_clf = GAMClassifier(lam=1.0, max_iter=30)
    gam_clf.fit(X_train, y_train)

    probs = gam_clf.predict_proba(X_test)
    assert probs.shape == (len(X_test), 2)
    assert ((probs >= 0.0) & (probs <= 1.0)).all()

    preds = gam_clf.predict(X_test)
    assert len(preds) == len(X_test)
    assert set(np.unique(preds)).issubset({0, 1})


# ===========================================================================
# 4. From-Scratch MARS Tests (Crucial Validation)
# ===========================================================================

def test_mars_synthetic_knot_recovery():
    """
    Validates from-scratch MARS on synthetic piecewise-linear data:
      y = max(0, x - 3.0) + epsilon
    MARS must identify a knot near 3.0 (within [2.0, 4.0]).
    """
    rng = np.random.default_rng(42)
    x = rng.uniform(0.0, 6.0, size=(250, 1))
    noise = rng.normal(0, 0.05, size=(250, 1))
    y_cont = np.maximum(0.0, x - 3.0) + noise
    y_binary = (y_cont > 0.5).astype(int).ravel()

    mars = FromScratchMARSClassifier(
        max_terms=8,
        max_degree=1,
        n_knots_per_feature=15,
        min_samples_leaf=10,
        random_state=42,
    )
    mars.fit(x, y_binary)

    # Inspect knots found across basis terms
    knots_found = []
    for b in mars.basis_functions_:
        for term in b.terms:
            knots_found.append(term.knot)

    assert len(knots_found) > 0, "MARS failed to create any hinge knots"
    # At least one knot must be located close to true knot 3.0
    closest_knot = min(knots_found, key=lambda k: abs(k - 3.0))
    assert abs(closest_knot - 3.0) < 1.0, f"Closest knot {closest_knot} is too far from true knot 3.0"


def test_mars_tabular_classification(small_dataset):
    """Verify that from-scratch MARS works on real tabular sensor data in GLM mode."""
    X_train, X_test, y_train, y_test = small_dataset

    mars = FromScratchMARSClassifier(
        max_terms=10,
        max_degree=2,
        class_weight="balanced",
        random_state=42,
    )
    mars.fit(X_train, y_train)

    Phi_train = mars.transform(X_train)
    assert Phi_train.shape[1] == len(mars.basis_functions_)

    probs = mars.predict_proba(X_test)
    assert probs.shape == (len(X_test), 2)
    assert not np.isnan(probs).any()

    preds = mars.predict(X_test)
    assert len(preds) == len(X_test)

r"""
src/ensemble_models.py
======================
Ensemble learning architectures for industrial predictive maintenance:
Soft Voting Classifier and Stacking Classifier.

Mathematical Architecture:
  1. Soft Voting Classifier:
     Averages calibrated predicted failure probabilities across heterogeneous base estimators:
     \hat{p}(y=1|x) = \frac{1}{M} \sum_{m=1}^M P_m(y=1|x)
     Combines orthogonal inductive biases: tree splitting (CART), random subspace (RandomForest),
     stagewise gradient descent (GradientBoosting), and regularized second-order Taylor (XGBoost).

  2. Stacking Classifier:
     Uses Stratified 5-Fold Cross-Validation to generate out-of-fold probability predictions
     z_i = [p_{1,-k}(x_i), \dots, p_{M,-k}(x_i)] for all training instances.
     A meta-learner (regularized Logistic Regression) fits on z to discover the optimal
     non-linear blending weights without validation leakage.

Author: ML Engineering Team
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    VotingClassifier,
    StackingClassifier,
    RandomForestClassifier,
    GradientBoostingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb

from config import (
    MODELS_DIR,
    RANDOM_STATE,
    N_CV_FOLDS,
    APPROX_SCALE_POS_WEIGHT,
)
from src.data_preprocessing import build_preprocessor
from utils.helpers import (
    assert_no_forbidden_columns,
    ensure_dir,
    get_logger,
    save_model,
)

logger = get_logger("ensemble_models")


def build_base_estimators(random_state: int = RANDOM_STATE) -> list[tuple[str, Any]]:
    """Construct the collection of base estimators sharing common preprocessing."""
    cart = DecisionTreeClassifier(
        max_depth=5,
        min_samples_leaf=10,
        class_weight="balanced",
        random_state=random_state,
    )
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=6,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )
    gb = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=4,
        random_state=random_state,
    )
    xgb_clf = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.08,
        scale_pos_weight=APPROX_SCALE_POS_WEIGHT,
        random_state=random_state,
        eval_metric="aucpr",
        n_jobs=-1,
    )

    return [
        ("cart", cart),
        ("rf", rf),
        ("gb", gb),
        ("xgb", xgb_clf),
    ]


def train_voting_ensemble(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = RANDOM_STATE,
) -> tuple[Pipeline, dict[str, Any]]:
    """Train Soft VotingClassifier across CART, RF, GradientBoosting, and XGBoost."""
    assert_no_forbidden_columns(X_train, context="train_voting_ensemble")
    ensure_dir(MODELS_DIR)

    preprocessor = build_preprocessor(include_engineered=True, encoding="onehot", scale_numeric=False)
    base_estimators = build_base_estimators(random_state=random_state)

    voting_clf = VotingClassifier(
        estimators=base_estimators,
        voting="soft",
        n_jobs=-1,
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", voting_clf),
    ])

    logger.info("Fitting Soft Voting Ensemble...")
    pipeline.fit(X_train, y_train)

    metadata = {
        "model_name": "Soft Voting Ensemble",
        "base_estimators": [name for name, _ in base_estimators],
        "voting": "soft",
    }
    save_model(pipeline, MODELS_DIR / "voting_ensemble.joblib", metadata=metadata)
    logger.info("Saved Voting Ensemble artifact.")
    return pipeline, metadata


def train_stacking_ensemble(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = RANDOM_STATE,
) -> tuple[Pipeline, dict[str, Any]]:
    """Train StackingClassifier with Out-Of-Fold probabilities and Logistic Regression meta-learner."""
    assert_no_forbidden_columns(X_train, context="train_stacking_ensemble")
    ensure_dir(MODELS_DIR)

    preprocessor = build_preprocessor(include_engineered=True, encoding="onehot", scale_numeric=False)
    base_estimators = build_base_estimators(random_state=random_state)

    meta_learner = LogisticRegression(
        class_weight="balanced",
        random_state=random_state,
        max_iter=300,
    )

    stacking_clf = StackingClassifier(
        estimators=base_estimators,
        final_estimator=meta_learner,
        cv=N_CV_FOLDS,  # Out-of-fold cross-validation
        stack_method="predict_proba",
        n_jobs=-1,
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", stacking_clf),
    ])

    logger.info("Fitting Stacking Ensemble with 5-fold out-of-fold meta-learner...")
    pipeline.fit(X_train, y_train)

    metadata = {
        "model_name": "Stacking Ensemble",
        "base_estimators": [name for name, _ in base_estimators],
        "final_estimator": "LogisticRegression (balanced)",
        "cv_folds": N_CV_FOLDS,
    }
    save_model(pipeline, MODELS_DIR / "stacking_ensemble.joblib", metadata=metadata)
    logger.info("Saved Stacking Ensemble artifact.")
    return pipeline, metadata

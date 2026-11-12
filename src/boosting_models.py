r"""
src/boosting_models.py
======================
Ensemble Boosting implementations for predictive maintenance:
AdaBoost, Gradient Boosting, and XGBoost.

Mathematical Foundations & Loss Functions:
  1. AdaBoost (Adaptive Boosting):
     - Loss Function: Exponential loss L(y, f(x)) = \exp(-y f(x)), y \in \{-1, +1\}.
     - Additive Model: F_M(x) = \sum_{m=1}^M \alpha_m h_m(x).
     - Stagewise update: Reweights training instances at each iteration m:
       w_i^{(m+1)} = w_i^{(m)} \exp(-\alpha_m y_i h_m(x_i))
       increasing focus on previously misclassified industrial anomalies.

  2. Gradient Boosting (GBDT):
     - Loss Function: Binomial deviance (negative Bernoulli log-likelihood):
       L(y, p) = - [y \log(p) + (1 - y) \log(1 - p)]
     - Optimization: Gradient descent in function space. At stage m, fits a regression
       tree h_m(x) to pseudo-residuals:
       r_{im} = -\left[\frac{\partial L(y_i, F(x_i))}{\partial F(x_i)}\right]_{F=F_{m-1}} = y_i - p_{m-1}(x_i)

  3. XGBoost (Extreme Gradient Boosting):
     - Objective: Second-order Taylor series expansion of arbitrary convex loss:
       \mathcal{L}^{(t)} \approx \sum_{i=1}^N \left[ g_i f_t(x_i) + \frac{1}{2} h_i f_t^2(x_i) \right] + \Omega(f_t)
       where g_i = \partial_{\hat{y}^{(t-1)}} l(y_i, \hat{y}^{(t-1)}) and h_i = \partial^2_{\hat{y}^{(t-1)}} l(y_i, \hat{y}^{(t-1)}).
     - Regularization: \Omega(f_t) = \gamma T + \frac{1}{2} \lambda \sum_{j=1}^T w_j^2 prevents overfitting.
     - Imbalance Handling: scale_pos_weight = \frac{N_{\text{neg}}}{N_{\text{pos}}} \approx 28.5 balances
       gradient updates for rare failure events.

Author: ML Engineering Team
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
import xgboost as xgb

from config import (
    MODELS_DIR,
    RANDOM_STATE,
    N_CV_FOLDS,
    APPROX_SCALE_POS_WEIGHT,
    COL_TEMP_DIFF,
)
from src.data_preprocessing import build_preprocessor
from utils.helpers import (
    assert_no_forbidden_columns,
    ensure_dir,
    get_logger,
    save_model,
)

logger = get_logger("boosting_models")


def train_adaboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    tune: bool = True,
    random_state: int = RANDOM_STATE,
) -> tuple[Pipeline, dict[str, Any]]:
    """Train and tune an AdaBoostClassifier (SAMME, exponential loss)."""
    assert_no_forbidden_columns(X_train, context="train_adaboost")
    ensure_dir(MODELS_DIR)

    include_engineered = COL_TEMP_DIFF in X_train.columns
    preprocessor = build_preprocessor(include_engineered=include_engineered, encoding="onehot", scale_numeric=False)
    base_model = AdaBoostClassifier(random_state=random_state)

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", base_model),
    ])

    if tune:
        param_grid = {
            "classifier__n_estimators": [50, 100, 150],
            "classifier__learning_rate": [0.1, 0.5, 1.0],
        }
        cv = StratifiedKFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=random_state)
        grid = GridSearchCV(pipeline, param_grid=param_grid, scoring="average_precision", cv=cv, n_jobs=-1)
        grid.fit(X_train, y_train)
        best_pipeline = grid.best_estimator_
        best_params = grid.best_params_
        best_cv = float(grid.best_score_)
    else:
        pipeline.fit(X_train, y_train)
        best_pipeline = pipeline
        best_params = base_model.get_params()
        best_cv = 0.0

    metadata = {
        "model_name": "AdaBoost",
        "loss_function": "Exponential Loss (SAMME)",
        "best_params": best_params,
        "pr_auc_cv_mean": best_cv,
    }
    save_model(best_pipeline, MODELS_DIR / "adaboost_model.joblib", metadata=metadata)
    logger.info(f"AdaBoost trained. Best CV PR-AUC: {best_cv:.4f}")
    return best_pipeline, metadata


def train_gradient_boosting(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    tune: bool = True,
    random_state: int = RANDOM_STATE,
) -> tuple[Pipeline, dict[str, Any]]:
    """Train and tune a GradientBoostingClassifier (binomial deviance / log loss)."""
    assert_no_forbidden_columns(X_train, context="train_gradient_boosting")
    ensure_dir(MODELS_DIR)

    include_engineered = COL_TEMP_DIFF in X_train.columns
    preprocessor = build_preprocessor(include_engineered=include_engineered, encoding="onehot", scale_numeric=False)
    base_model = GradientBoostingClassifier(random_state=random_state, loss="log_loss")

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", base_model),
    ])

    if tune:
        param_grid = {
            "classifier__n_estimators": [50, 100, 150],
            "classifier__learning_rate": [0.05, 0.1, 0.2],
            "classifier__max_depth": [3, 4, 5],
            "classifier__subsample": [0.8, 1.0],
        }
        cv = StratifiedKFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=random_state)
        grid = GridSearchCV(pipeline, param_grid=param_grid, scoring="average_precision", cv=cv, n_jobs=-1)
        grid.fit(X_train, y_train)
        best_pipeline = grid.best_estimator_
        best_params = grid.best_params_
        best_cv = float(grid.best_score_)
    else:
        pipeline.fit(X_train, y_train)
        best_pipeline = pipeline
        best_params = base_model.get_params()
        best_cv = 0.0

    metadata = {
        "model_name": "GradientBoosting",
        "loss_function": "Binomial Deviance (Log Loss)",
        "best_params": best_params,
        "pr_auc_cv_mean": best_cv,
    }
    save_model(best_pipeline, MODELS_DIR / "gradient_boosting_model.joblib", metadata=metadata)
    logger.info(f"GradientBoosting trained. Best CV PR-AUC: {best_cv:.4f}")
    return best_pipeline, metadata


def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    tune: bool = True,
    random_state: int = RANDOM_STATE,
) -> tuple[Pipeline, dict[str, Any]]:
    """
    Train and tune XGBoost with exact scale_pos_weight calculated from training folds.
    """
    assert_no_forbidden_columns(X_train, context="train_xgboost")
    ensure_dir(MODELS_DIR)

    # Compute exact scale_pos_weight from training set
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    scale_pos_weight = float(n_neg / n_pos)
    logger.info(f"XGBoost scale_pos_weight: {scale_pos_weight:.2f} ({n_neg} neg / {n_pos} pos)")

    include_engineered = COL_TEMP_DIFF in X_train.columns
    preprocessor = build_preprocessor(include_engineered=include_engineered, encoding="onehot", scale_numeric=False)

    base_model = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="aucpr",
        scale_pos_weight=scale_pos_weight,
        random_state=random_state,
        n_jobs=-1,
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", base_model),
    ])

    if tune:
        param_grid = {
            "classifier__n_estimators": [60, 100, 150],
            "classifier__max_depth": [3, 4, 6],
            "classifier__learning_rate": [0.03, 0.08, 0.15],
            "classifier__subsample": [0.8, 1.0],
            "classifier__colsample_bytree": [0.8, 1.0],
        }
        cv = StratifiedKFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=random_state)
        grid = GridSearchCV(pipeline, param_grid=param_grid, scoring="average_precision", cv=cv, n_jobs=-1)
        grid.fit(X_train, y_train)
        best_pipeline = grid.best_estimator_
        best_params = grid.best_params_
        best_cv = float(grid.best_score_)
    else:
        pipeline.fit(X_train, y_train)
        best_pipeline = pipeline
        best_params = base_model.get_params()
        best_cv = 0.0

    metadata = {
        "model_name": "XGBoost",
        "loss_function": "Regularized Log-Loss (Second-Order Taylor Expansion)",
        "scale_pos_weight": scale_pos_weight,
        "best_params": best_params,
        "pr_auc_cv_mean": best_cv,
    }
    save_model(best_pipeline, MODELS_DIR / "xgboost_model.joblib", metadata=metadata)
    logger.info(f"XGBoost trained. Best CV PR-AUC: {best_cv:.4f}")
    return best_pipeline, metadata


def compare_boosting_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> dict[str, tuple[Pipeline, dict[str, Any]]]:
    """Train all 3 boosting algorithms and name the strongest based on PR-AUC."""
    logger.info("Comparing Boosting Models: AdaBoost vs GradientBoosting vs XGBoost...")
    ada_pipe, ada_meta = train_adaboost(X_train, y_train)
    gb_pipe, gb_meta   = train_gradient_boosting(X_train, y_train)
    xgb_pipe, xgb_meta = train_xgboost(X_train, y_train)

    scores = {
        "AdaBoost": ada_meta["pr_auc_cv_mean"],
        "GradientBoosting": gb_meta["pr_auc_cv_mean"],
        "XGBoost": xgb_meta["pr_auc_cv_mean"],
    }
    strongest = max(scores, key=scores.get)
    logger.info(f"Strongest Boosting Model: {strongest} (PR-AUC: {scores[strongest]:.4f})")
    print(f"\nBoosting Comparison (CV PR-AUC): {scores} -> Strongest: {strongest}")

    return {
        "AdaBoost": (ada_pipe, ada_meta),
        "GradientBoosting": (gb_pipe, gb_meta),
        "XGBoost": (xgb_pipe, xgb_meta),
    }

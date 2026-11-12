r"""
src/cart_model.py
=================
Classification and Regression Trees (CART) implementation for predictive maintenance.

Mathematical Foundations:
  1. Gini Impurity:
     I_G(t) = 1 - \sum_{k=1}^K p(k|t)^2
     Measures node class heterogeneity. CART searches at each step for the feature
     and split threshold s that maximizes impurity reduction:
     \Delta I_G(s, t) = I_G(t) - \frac{N_L}{N} I_G(t_L) - \frac{N_R}{N} I_G(t_R)

  2. Cost-Complexity Pruning (Minimal Cost-Complexity):
     R_\alpha(T) = R(T) + \alpha |T|
     where R(T) is the misclassification cost (or deviance), |T| is tree terminal node
     complexity, and \alpha \ge 0 is the complexity parameter governing the bias-variance
     trade-off. Pruning prevents overfitting to sensor noise in rare-failure regimes.

Author: ML Engineering Team
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, plot_tree

from config import (
    MODELS_DIR,
    VIZ_DIR,
    RANDOM_STATE,
    N_CV_FOLDS,
)
from src.data_preprocessing import build_preprocessor, get_train_test_data
from utils.helpers import (
    assert_no_forbidden_columns,
    ensure_dir,
    get_logger,
    save_model,
)

logger = get_logger("cart_model")


def train_cart_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    encoding: str = "ordinal",
    tune_hyperparameters: bool = True,
    random_state: int = RANDOM_STATE,
) -> tuple[Pipeline, dict[str, Any]]:
    """
    Train and tune a cost-complexity regularized DecisionTreeClassifier within an sklearn Pipeline.

    Parameters
    ----------
    X_train              : Training features DataFrame.
    y_train              : Binary target Series.
    encoding             : 'ordinal' (monotonic L<M<H grade) or 'onehot'.
    tune_hyperparameters : If True, runs 5-fold Stratified CV optimizing PR-AUC.
    random_state         : Seed for reproducibility.

    Returns
    -------
    (fitted_pipeline, tuning_metadata)
    """
    assert_no_forbidden_columns(X_train, context="train_cart_model")
    ensure_dir(MODELS_DIR)

    preprocessor = build_preprocessor(include_engineered=True, encoding=encoding, scale_numeric=False)

    base_dt = DecisionTreeClassifier(
        criterion="gini",
        class_weight="balanced",
        random_state=random_state,
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", base_dt),
    ])

    if tune_hyperparameters:
        logger.info("Tuning CART hyperparameters via Stratified 5-Fold CV (PR-AUC)...")
        param_grid = {
            "classifier__max_depth": [3, 4, 5, 6, 8],
            "classifier__min_samples_split": [10, 20, 40],
            "classifier__min_samples_leaf": [5, 10, 20],
            "classifier__ccp_alpha": [0.0, 0.001, 0.005],
        }

        cv = StratifiedKFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=random_state)
        grid = GridSearchCV(
            pipeline,
            param_grid=param_grid,
            scoring="average_precision",  # PR-AUC for severe imbalance
            cv=cv,
            n_jobs=-1,
            refit=True,
        )
        grid.fit(X_train, y_train)

        best_pipeline = grid.best_estimator_
        best_params = grid.best_params_
        best_cv_score = float(grid.best_score_)
        logger.info(f"CART Best CV PR-AUC: {best_cv_score:.4f} with params: {best_params}")
    else:
        pipeline.fit(X_train, y_train)
        best_pipeline = pipeline
        best_params = base_dt.get_params()
        best_cv_score = 0.0

    metadata = {
        "model_name": "CART (DecisionTreeClassifier)",
        "best_params": best_params,
        "pr_auc_cv_mean": best_cv_score,
        "features_used": list(X_train.columns),
        "encoding": encoding,
    }

    # Save model artifact
    artifact_path = MODELS_DIR / "cart_model.joblib"
    save_model(best_pipeline, artifact_path, metadata=metadata)
    logger.info(f"Saved CART model artifact to {artifact_path}")

    return best_pipeline, metadata


def plot_cart_tree(
    pipeline: Pipeline,
    feature_names: list[str],
    save_path: Path | None = None,
    max_depth: int = 3,
) -> None:
    """Export visualization of the pruned decision tree architecture."""
    if save_path is None:
        save_path = VIZ_DIR / "cart_decision_tree.png"
    ensure_dir(save_path.parent)

    dt_model: DecisionTreeClassifier = pipeline.named_steps["classifier"]

    plt.figure(figsize=(20, 10), dpi=300)
    plot_tree(
        dt_model,
        max_depth=max_depth,
        feature_names=feature_names,
        class_names=["Normal", "Failure"],
        filled=True,
        rounded=True,
        fontsize=9,
        precision=2,
    )
    plt.title(f"CART Decision Tree Architecture (Top {max_depth} Levels, Pruned)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    logger.info(f"Saved CART tree diagram to {save_path.name}")

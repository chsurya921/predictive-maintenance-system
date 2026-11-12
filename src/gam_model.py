r"""
src/gam_model.py
================
Generalized Additive Model (GAM) for predictive maintenance using pyGAM.

Mathematical Formulation:
  The LogisticGAM models log-odds of equipment failure as a sum of smooth,
  non-linear spline functions over continuous sensors plus factor terms for product Type:
  \log\left(\frac{p(x)}{1 - p(x)}\right) = \beta_0 + \sum_{j=1}^5 f_j(x_j) + f_{\text{type}}(\text{Type})

  Each smooth term f_j(x_j) is parameterized by a basis of B-splines penalized by roughness:
  \mathcal{P}(\lambda_j) = \lambda_j \int \left[ f_j''(t) \right]^2 dt
  Balancing goodness-of-fit against curve smoothness.

Interpretability:
  Unlike black-box ensembles, GAM allows exact visualization of each sensor's marginal
  contribution to failure probability via Partial Dependence Plots (PDP) with 95% confidence bands.

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
from pygam import LogisticGAM, s, f
from sklearn.base import BaseEstimator, ClassifierMixin

from config import (
    MODELS_DIR,
    VIZ_DIR,
    COL_AIR_TEMP,
    COL_PROCESS_TEMP,
    COL_ROT_SPEED,
    COL_TORQUE,
    COL_TOOL_WEAR,
    COL_TYPE,
)
from src.data_preprocessing import load_and_clean_data
from utils.helpers import (
    assert_no_forbidden_columns,
    ensure_dir,
    get_logger,
    save_model,
)

logger = get_logger("gam_model")


class GAMClassifier(BaseEstimator, ClassifierMixin):
    """
    Scikit-learn compatible wrapper around pyGAM's LogisticGAM.
    Applies ordinal mapping to Product Type (L->0, M->1, H->2) and fits spline terms.
    """
    def __init__(self, lam: float = 0.6, max_iter: int = 100):
        self.lam = lam
        self.max_iter = max_iter
        self.gam_: LogisticGAM | None = None
        self.feature_names_: list[str] = []
        self.classes_ = np.array([0, 1])

    def _prepare_matrix(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            assert_no_forbidden_columns(X, context="GAMClassifier")
            # Expected order: air_temp, proc_temp, rot_speed, torque, tool_wear, type
            type_mapping = {"L": 0, "M": 1, "H": 2}
            cols = [
                COL_AIR_TEMP,
                COL_PROCESS_TEMP,
                COL_ROT_SPEED,
                COL_TORQUE,
                COL_TOOL_WEAR,
            ]
            X_num = X[cols].values.astype(float)
            if COL_TYPE in X.columns:
                type_vals = X[COL_TYPE].map(type_mapping).fillna(0).values.reshape(-1, 1)
            else:
                type_vals = np.zeros((len(X), 1))
            return np.hstack([X_num, type_vals])
        return np.asarray(X, dtype=float)

    def fit(self, X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray) -> "GAMClassifier":
        X_mat = self._prepare_matrix(X)
        y_arr = np.asarray(y, dtype=int).ravel()

        if isinstance(X, pd.DataFrame):
            self.feature_names_ = [COL_AIR_TEMP, COL_PROCESS_TEMP, COL_ROT_SPEED, COL_TORQUE, COL_TOOL_WEAR, COL_TYPE]
        else:
            self.feature_names_ = [f"sensor_{i}" for i in range(5)] + ["type"]

        # pyGAM formula: 5 spline terms for continuous sensors + 1 factor term for Type (3 levels: 0, 1, 2)
        gam_formula = s(0) + s(1) + s(2) + s(3) + s(4) + f(5)

        logger.info(f"Fitting pyGAM LogisticGAM with smoothing lambda={self.lam}...")
        self.gam_ = LogisticGAM(gam_formula, lam=self.lam, max_iter=self.max_iter)
        self.gam_.fit(X_mat, y_arr)
        return self

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        X_mat = self._prepare_matrix(X)
        p1 = self.gam_.predict_proba(X_mat)
        p1 = np.clip(p1, 1e-6, 1.0 - 1e-6)
        p0 = 1.0 - p1
        return np.column_stack([p0, p1])

    def predict(self, X: pd.DataFrame | np.ndarray, threshold: float = 0.5) -> np.ndarray:
        probs = self.predict_proba(X)[:, 1]
        return (probs >= threshold).astype(int)


def train_gam_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    tune_lambda: bool = True,
) -> tuple[GAMClassifier, dict[str, Any]]:
    """
    Train and optionally tune smoothing parameters for LogisticGAM.
    Generates partial dependence plots with 95% confidence intervals.
    """
    assert_no_forbidden_columns(X_train, context="train_gam_model")
    ensure_dir(MODELS_DIR)

    clf = GAMClassifier(lam=0.6)
    X_mat = clf._prepare_matrix(X_train)
    y_arr = y_train.values

    if tune_lambda:
        logger.info("Grid-searching optimal smoothing lambda for pyGAM...")
        gam_formula = s(0) + s(1) + s(2) + s(3) + s(4) + f(5)
        raw_gam = LogisticGAM(gam_formula)
        try:
            # Grid search over logarithmic smoothing grid
            lam_grid = np.logspace(-2, 2, 5)
            raw_gam.gridsearch(X_mat, y_arr, lam=lam_grid)
            best_lam = float(raw_gam.lam[0][0]) if isinstance(raw_gam.lam, (list, np.ndarray)) else float(raw_gam.lam)
            logger.info(f"pyGAM optimal smoothing lambda: {best_lam}")
            clf.lam = best_lam
        except Exception as e:
            logger.warning(f"pyGAM gridsearch encountered error: {e}. Using default lam=0.6")

    clf.fit(X_train, y_train)

    # Generate Partial Dependence Plots (PDP)
    plot_gam_partial_dependence(clf)

    metadata = {
        "model_name": "pyGAM LogisticGAM",
        "lam": clf.lam,
        "features_used": clf.feature_names_,
    }
    save_model(clf, MODELS_DIR / "gam_model.joblib", metadata=metadata)
    logger.info("Saved GAM model artifact successfully.")
    return clf, metadata


def plot_gam_partial_dependence(clf: GAMClassifier, save_path: Path | None = None) -> None:
    """Generate and save publication-quality Partial Dependence Plots for all 5 continuous sensors."""
    if save_path is None:
        save_path = VIZ_DIR / "gam_partial_dependence.png"
    ensure_dir(save_path.parent)

    sensor_labels = [
        "Air Temperature [K]",
        "Process Temperature [K]",
        "Rotational Speed [rpm]",
        "Torque [Nm]",
        "Tool Wear [min]",
    ]

    fig, axes = plt.subplots(1, 5, figsize=(20, 4.5), sharey=True)

    for i in range(5):
        XX = clf.gam_.generate_X_grid(term=i)
        pdep, confi = clf.gam_.partial_dependence(term=i, X=XX, width=0.95)

        axes[i].plot(XX[:, i], pdep, color="#2563eb", linewidth=2.5, label="Log-odds Effect")
        axes[i].fill_between(XX[:, i], confi[:, 0], confi[:, 1], color="#93c5fd", alpha=0.4, label="95% CI")
        axes[i].set_title(sensor_labels[i], fontsize=11, fontweight="bold")
        axes[i].set_xlabel("Sensor Value")
        if i == 0:
            axes[i].set_ylabel("Marginal Effect on Log-Odds $\\beta_j(x_j)$")
        axes[i].grid(True, linestyle="--", alpha=0.5)

    axes[0].legend(loc="upper left")
    plt.suptitle("pyGAM Logistic Spline Marginal Partial Dependence Curves with 95% CI", fontsize=13, fontweight="bold", y=1.03)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved GAM partial dependence plot to: {save_path.name}")

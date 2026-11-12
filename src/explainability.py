r"""
src/explainability.py
=====================
Model Explainability and Feature Attribution using SHAP (SHapley Additive exPlanations).

Mathematical Principles:
  Based on cooperative game theory, Shapley values uniquely satisfy four desirable axioms:
  Efficiency, Symmetry, Dummy (Null player), and Additivity:
  \phi_i(x) = \sum_{S \subseteq F \setminus \{i\}} \frac{|S|! (|F| - |S| - 1)!}{|F|!} \left[ f(S \cup \{i\}) - f(S) \right]

  For tree ensembles, TreeSHAP (TreeExplainer) evaluates exact polynomial-time
  feature attributions:
  f(x) = \phi_0 + \sum_{j=1}^M \phi_j(x)
  decomposing machine failure risk into explicit physical sensor contributions.

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
import shap

from config import VIZ_DIR
from src.data_preprocessing import get_train_test_data
from utils.helpers import (
    assert_no_forbidden_columns,
    ensure_dir,
    get_logger,
)

logger = get_logger("explainability")


def generate_shap_explanations(
    fitted_pipeline: Any,
    X_test: pd.DataFrame,
    max_samples: int = 300,
    save_dir: Path | None = None,
) -> shap.Explanation:
    """
    Compute TreeSHAP values for the gradient boosted model and generate publication figures.
    """
    if save_dir is None:
        save_dir = VIZ_DIR
    ensure_dir(save_dir)

    assert_no_forbidden_columns(X_test, context="SHAP explanation")
    logger.info(f"Generating SHAP explanations using {max_samples} test samples...")

    # Extract classifier and preprocessor
    preprocessor = fitted_pipeline.named_steps["preprocessor"]
    classifier = fitted_pipeline.named_steps["classifier"]

    # Transform test features
    X_test_sub = X_test.iloc[:max_samples].copy()
    X_trans = preprocessor.transform(X_test_sub)

    # Get feature names after preprocessing
    feature_names = preprocessor.get_feature_names_out()

    # Build TreeExplainer
    explainer = shap.TreeExplainer(classifier)
    shap_values = explainer(X_trans)

    # 1. SHAP Beeswarm Summary Plot
    plt.figure(figsize=(10, 6), dpi=300)
    shap.summary_plot(shap_values.values, X_trans, feature_names=feature_names, show=False)
    plt.title("TreeSHAP Global Feature Attribution (Beeswarm Summary)", fontsize=13, fontweight="bold", pad=15)
    plt.tight_layout()
    plt.savefig(save_dir / "shap_summary_beeswarm.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 2. SHAP Mean Absolute Feature Importance Bar Chart
    plt.figure(figsize=(9, 5), dpi=300)
    shap.summary_plot(shap_values.values, X_trans, feature_names=feature_names, plot_type="bar", show=False)
    plt.title("Mean Absolute SHAP Feature Importance (Global Impact)", fontsize=13, fontweight="bold", pad=15)
    plt.tight_layout()
    plt.savefig(save_dir / "shap_feature_importance_bar.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 3. Individual Instance Waterfall Plot (First Failure Case)
    y_test_sub = get_train_test_data()[3].iloc[:max_samples].values
    failure_indices = np.where(y_test_sub == 1)[0]
    sample_idx = int(failure_indices[0]) if len(failure_indices) > 0 else 0

    plt.figure(figsize=(9, 5), dpi=300)
    explanation_single = shap.Explanation(
        values=shap_values.values[sample_idx],
        base_values=explainer.expected_value if np.isscalar(explainer.expected_value) else explainer.expected_value[0],
        data=X_trans[sample_idx],
        feature_names=feature_names,
    )
    shap.plots.waterfall(explanation_single, show=False)
    plt.title(f"Local Instance SHAP Decomposition (Test Sample #{sample_idx}, Failure)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_dir / "shap_waterfall_individual.png", dpi=300, bbox_inches="tight")
    plt.close()

    logger.info("Saved SHAP summary beeswarm, importance bar, and individual waterfall plots.")
    return shap_values

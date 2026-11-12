r"""
src/model_evaluation.py
=======================
Evaluation metrics, threshold calibration, and bootstrap confidence intervals.

Methodological Guidelines Enforced:
  1. Threshold Selection on Training CV:
     Optimal operating threshold t* is picked strictly via 5-fold cross-validation
     on training data by maximizing the F_2 score:
     F_2 = \frac{5 \cdot \text{Precision} \cdot \text{Recall}}{4 \cdot \text{Precision} + \text{Recall}}
     Penalizes undetected failures (False Negatives) 2x more heavily than false alarms.
     The held-out test set is NEVER used for threshold tuning.

  2. Metrics Reported:
     PR-AUC (Average Precision, primary metric for severe imbalance),
     ROC-AUC, Recall@t*, Precision@t*, F1@t*, Accuracy, and Confusion Matrix.

  3. Bootstrap Confidence Intervals:
     Because the test set contains only 68 positive failure events, point estimates
     carry inherent uncertainty. We report 95% non-parametric bootstrap confidence intervals
     (1,000 resamples) for all key metrics.

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
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    fbeta_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
)
from sklearn.model_selection import StratifiedKFold

from config import (
    METRICS_CSV,
    VIZ_DIR,
    RANDOM_STATE,
    N_CV_FOLDS,
)
from utils.helpers import (
    bootstrap_ci,
    ensure_dir,
    get_logger,
)

logger = get_logger("model_evaluation")


def find_optimal_threshold_cv(
    model: Any,
    X_train: pd.DataFrame | np.ndarray,
    y_train: pd.Series | np.ndarray,
    beta: float = 2.0,
    random_state: int = RANDOM_STATE,
) -> float:
    """
    Find optimal classification threshold t* on TRAINING data using Stratified 5-Fold CV.
    Maximizes F_beta (default beta=2.0 for recall-heavy industrial monitoring).
    """
    y_arr = np.asarray(y_train).ravel()
    cv = StratifiedKFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=random_state)

    oof_probs = np.zeros(len(y_arr))
    for train_idx, val_idx in cv.split(X_train, y_arr):
        if isinstance(X_train, pd.DataFrame):
            X_tr, y_tr = X_train.iloc[train_idx], y_arr[train_idx]
            X_va = X_train.iloc[val_idx]
        else:
            X_tr, y_tr = X_train[train_idx], y_arr[train_idx]
            X_va = X_train[val_idx]

        # Clone or fit model on train fold
        model.fit(X_tr, y_tr)
        oof_probs[val_idx] = model.predict_proba(X_va)[:, 1]

    # Grid search threshold t in [0.05, 0.90] to maximize F_beta
    thresholds = np.linspace(0.05, 0.90, 86)
    best_thresh = 0.5
    best_fbeta = -1.0

    for t in thresholds:
        preds = (oof_probs >= t).astype(int)
        score = fbeta_score(y_arr, preds, beta=beta, zero_division=0)
        if score > best_fbeta:
            best_fbeta = score
            best_thresh = float(t)

    logger.info(f"CV Optimal Threshold: {best_thresh:.3f} (Max CV F_{beta}: {best_fbeta:.4f})")
    return best_thresh


def evaluate_model_on_test(
    model: Any,
    X_test: pd.DataFrame | np.ndarray,
    y_test: pd.Series | np.ndarray,
    threshold: float = 0.5,
    n_bootstrap: int = 1000,
    random_state: int = RANDOM_STATE,
) -> dict[str, Any]:
    """
    Evaluate fitted estimator on shared held-out test set with 95% bootstrap intervals.
    """
    y_arr = np.asarray(y_test).ravel()
    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= threshold).astype(int)

    # Point estimates
    acc = float(accuracy_score(y_arr, preds))
    prec = float(precision_score(y_arr, preds, zero_division=0))
    rec = float(recall_score(y_arr, preds, zero_division=0))
    f1 = float(f1_score(y_arr, preds, zero_division=0))
    f2 = float(fbeta_score(y_arr, preds, beta=2.0, zero_division=0))
    roc_auc = float(roc_auc_score(y_arr, probs))
    pr_auc = float(average_precision_score(y_arr, probs))
    cm = confusion_matrix(y_arr, preds).tolist()

    # Bootstrap CIs for key metrics
    _, pr_lower, pr_upper = bootstrap_ci(y_arr, probs, average_precision_score, n_bootstrap, random_state=random_state)
    _, roc_lower, roc_upper = bootstrap_ci(y_arr, probs, roc_auc_score, n_bootstrap, random_state=random_state)
    _, rec_lower, rec_upper = bootstrap_ci(
        y_arr, probs, lambda yt, ys: recall_score(yt, (ys >= threshold).astype(int), zero_division=0),
        n_bootstrap, random_state=random_state
    )

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "recall_ci_95": f"[{rec_lower:.3f}, {rec_upper:.3f}]",
        "f1": f1,
        "f2": f2,
        "roc_auc": roc_auc,
        "roc_auc_ci_95": f"[{roc_lower:.3f}, {roc_upper:.3f}]",
        "pr_auc": pr_auc,
        "pr_auc_ci_95": f"[{pr_lower:.3f}, {pr_upper:.3f}]",
        "pr_auc_lower": pr_lower,
        "pr_auc_upper": pr_upper,
        "threshold": threshold,
        "confusion_matrix": cm,
        "y_true": y_arr,
        "y_prob": probs,
    }


def plot_model_comparison_curves(
    results_dict: dict[str, dict[str, Any]],
    save_dir: Path | None = None,
) -> None:
    """Generate publication-ready ROC curves, PR curves, and Confusion Matrices."""
    if save_dir is None:
        save_dir = VIZ_DIR
    ensure_dir(save_dir)

    colors = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
        "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22"
    ]

    # 1. ROC Curves
    plt.figure(figsize=(9, 7), dpi=300)
    for idx, (m_name, res) in enumerate(results_dict.items()):
        fpr, tpr, _ = roc_curve(res["y_true"], res["y_prob"])
        auc_val = res["roc_auc"]
        c = colors[idx % len(colors)]
        plt.plot(fpr, tpr, color=c, lw=2, label=f"{m_name} (ROC-AUC = {auc_val:.3f})")

    plt.plot([0, 1], [0, 1], "k--", lw=1.5, alpha=0.7, label="Chance Level (0.50)")
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=11, fontweight="bold")
    plt.ylabel("True Positive Rate (Recall)", fontsize=11, fontweight="bold")
    plt.title("Receiver Operating Characteristic (ROC) Comparison", fontsize=13, fontweight="bold")
    plt.legend(loc="lower right", fontsize=9)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_dir / "roc_curves_comparison.png", dpi=300)
    plt.close()

    # 2. Precision-Recall Curves (Primary for Imbalanced Maintenance)
    plt.figure(figsize=(9, 7), dpi=300)
    baseline_prev = np.mean(list(results_dict.values())[0]["y_true"])

    for idx, (m_name, res) in enumerate(results_dict.items()):
        prec, rec, _ = precision_recall_curve(res["y_true"], res["y_prob"])
        pr_val = res["pr_auc"]
        c = colors[idx % len(colors)]
        plt.plot(rec, prec, color=c, lw=2, label=f"{m_name} (PR-AUC = {pr_val:.3f})")

    plt.axhline(baseline_prev, color="k", linestyle="--", lw=1.5, alpha=0.7, label=f"Random Baseline ({baseline_prev*100:.1f}%)")
    plt.xlabel("Recall (True Failure Catch Rate)", fontsize=11, fontweight="bold")
    plt.ylabel("Precision (Positive Predictive Value)", fontsize=11, fontweight="bold")
    plt.title("Precision-Recall (PR) Curves: Industrial Failure Detection", fontsize=13, fontweight="bold")
    plt.legend(loc="lower left", fontsize=9)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_dir / "pr_curves_comparison.png", dpi=300)
    plt.close()

    logger.info("Saved ROC and PR curves comparison figures.")

r"""
src/bayesian_model_averaging.py
===============================
Mathematically Justified Bayesian Model Averaging (BMA) for Predictive Maintenance.

Mathematical Foundations:
  1. Candidate Model Space \mathcal{M} = \{ M_1, \dots, M_K \}:
     Full power set of non-empty predictor subsets: 2^P - 1 models (e.g. 255 models for P=8).

  2. Prior Distribution over Models p(M_k):
     Uniform prior over the model space: p(M_k) = \frac{1}{K} (non-informative).

  3. Marginal Likelihood Approximation via BIC:
     By Laplace approximation around the MLE \hat{\theta}_k:
     \log p(D|M_k) \approx \log L(D|\hat{\theta}_k) - \frac{d_k}{2} \log N = -\frac{1}{2} \text{BIC}_k
     where \text{BIC}_k = -2 \log L_k + d_k \log N, with d_k = |M_k| + 1.

  4. Posterior Model Probability p(M_k|D):
     p(M_k|D) = \frac{p(D|M_k) p(M_k)}{\sum_j p(D|M_j) p(M_j)}
     Computed in log space with logsumexp for strict numerical stability:
     \log p(M_k|D) = -\frac{1}{2}\text{BIC}_k - \text{logsumexp}_j\left(-\frac{1}{2}\text{BIC}_j\right)
     w_k = \exp(\log p(M_k|D)), \quad \sum_{k=1}^K w_k = 1.0

  5. BMA Predictive Probability:
     P(y=1|x, D) = \sum_{k=1}^K p(M_k|D) \cdot P(y=1|x, M_k)

  6. Posterior Inclusion Probability (PIP) per Feature j:
     \text{PIP}(X_j) = \sum_{k: X_j \in M_k} p(M_k|D)

  7. Posterior Entropy & Effective Number of Models:
     H = -\sum_k w_k \log w_k, \quad N_{\text{eff}} = \exp(H)

  8. Predictive Epistemic Uncertainty (Between-Model Variance):
     \sigma^2_{\text{BMA}}(x) = \sum_{k=1}^K w_k \left[ P(y=1|x, M_k) - \bar{P}(y=1|x) \right]^2

All weights w_k are derived STRICTLY on TRAINING data.

Author: ML Engineering Team
"""

from __future__ import annotations

import itertools
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler

from config import (
    MODELS_DIR,
    RANDOM_STATE,
    COL_TYPE,
)
from utils.helpers import (
    assert_no_forbidden_columns,
    ensure_dir,
    get_logger,
    save_model,
)

logger = get_logger("bma_model")


class BayesianModelAveragingClassifier(BaseEstimator, ClassifierMixin):
    """
    Bayesian Model Averaging (BMA) Classifier over feature subset logistic models.
    """
    def __init__(
        self,
        max_features_to_enumerate: int = 8,
        regularization_C: float = 1e4,  # unpenalized approximation
        random_state: int = RANDOM_STATE,
    ):
        self.max_features_to_enumerate = max_features_to_enumerate
        self.regularization_C = regularization_C
        self.random_state = random_state

        self.models_: list[tuple[list[int], LogisticRegression]] = []
        self.weights_: np.ndarray = np.array([])
        self.bics_: np.ndarray = np.array([])
        self.feature_names_: list[str] = []
        self.scaler_: StandardScaler | None = None
        self.classes_ = np.array([0, 1])

    def _prepare_features(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            assert_no_forbidden_columns(X, context="BMA")
            X_df = X.copy()
            if COL_TYPE in X_df.columns:
                X_df[COL_TYPE] = X_df[COL_TYPE].map({"L": 0.0, "M": 1.0, "H": 2.0}).fillna(0.0)
            return X_df.values.astype(float)
        return np.asarray(X, dtype=float)

    def fit(self, X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray) -> "BayesianModelAveragingClassifier":
        if isinstance(X, pd.DataFrame):
            assert_no_forbidden_columns(X, context="BMA fit")
            self.feature_names_ = list(X.columns)
        else:
            self.feature_names_ = [f"x_{i}" for i in range(X.shape[1])]

        X_mat = self._prepare_features(X)
        y_vec = np.asarray(y, dtype=int).ravel()
        n_samples, n_features = X_mat.shape

        # Scale features strictly on training data
        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X_mat)

        # Enumerate candidate predictor subsets
        p_subset = min(n_features, self.max_features_to_enumerate)
        subsets: list[list[int]] = []
        for r in range(1, p_subset + 1):
            for combo in itertools.combinations(range(p_subset), r):
                subsets.append(list(combo))

        n_models = len(subsets)
        logger.info(f"Enumerating {n_models} candidate logistic models for BMA...")

        models_list = []
        bics = []

        for idx, feat_indices in enumerate(subsets):
            X_sub = X_scaled[:, feat_indices]
            # Fit MLE Logistic Regression
            clf = LogisticRegression(
                C=self.regularization_C,
                solver="lbfgs",
                max_iter=300,
                random_state=self.random_state,
            )
            clf.fit(X_sub, y_vec)

            # Evaluate log-likelihood on training data
            probs = clf.predict_proba(X_sub)[:, 1]
            probs = np.clip(probs, 1e-12, 1.0 - 1e-12)
            loss_val = log_loss(y_vec, probs, normalize=False)
            log_lik = -loss_val

            # Number of estimated parameters: coefficients + intercept
            d_k = len(feat_indices) + 1
            bic = -2.0 * log_lik + d_k * np.log(n_samples)

            models_list.append((feat_indices, clf))
            bics.append(bic)

        self.models_ = models_list
        self.bics_ = np.array(bics)

        # Compute posterior model probabilities in log-space via logsumexp
        # log p(D|M_k) \approx -0.5 * BIC_k
        log_unnorm_posterior = -0.5 * self.bics_
        log_norm_const = logsumexp(log_unnorm_posterior)
        log_weights = log_unnorm_posterior - log_norm_const

        self.weights_ = np.exp(log_weights)
        # Ensure exact simplex normalization
        self.weights_ /= np.sum(self.weights_)

        logger.info(
            f"BMA training completed. Models: {len(self.models_)}, "
            f"Weights sum: {self.weights_.sum():.6f}, "
            f"Top model weight: {self.weights_.max():.4f}"
        )
        return self

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Compute BMA posterior expected failure probability."""
        X_mat = self._prepare_features(X)
        X_scaled = self.scaler_.transform(X_mat)

        weighted_probs = np.zeros(len(X_mat), dtype=float)
        for weight, (feat_indices, clf) in zip(self.weights_, self.models_):
            if weight < 1e-9:
                continue  # skip negligible models for performance
            X_sub = X_scaled[:, feat_indices]
            p_k = clf.predict_proba(X_sub)[:, 1]
            weighted_probs += weight * p_k

        p1 = np.clip(weighted_probs, 1e-6, 1.0 - 1e-6)
        p0 = 1.0 - p1
        return np.column_stack([p0, p1])

    def predict(self, X: pd.DataFrame | np.ndarray, threshold: float = 0.5) -> np.ndarray:
        probs = self.predict_proba(X)[:, 1]
        return (probs >= threshold).astype(int)

    def predict_uncertainty(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        r"""
        Compute predictive epistemic uncertainty (between-model variance):
        \sigma^2(x) = \sum_k w_k (P_k(x) - \bar{P}(x))^2
        """
        X_mat = self._prepare_features(X)
        X_scaled = self.scaler_.transform(X_mat)
        p_bma = self.predict_proba(X)[:, 1]

        var_between = np.zeros(len(X_mat), dtype=float)
        for weight, (feat_indices, clf) in zip(self.weights_, self.models_):
            if weight < 1e-9:
                continue
            X_sub = X_scaled[:, feat_indices]
            p_k = clf.predict_proba(X_sub)[:, 1]
            var_between += weight * ((p_k - p_bma) ** 2)

        return var_between

    def compute_posterior_inclusion_probabilities(self) -> dict[str, float]:
        """Compute PIP for each predictor."""
        pips: dict[str, float] = {name: 0.0 for name in self.feature_names_[:self.max_features_to_enumerate]}
        for weight, (feat_indices, _) in zip(self.weights_, self.models_):
            for idx in feat_indices:
                fname = self.feature_names_[idx]
                pips[fname] += float(weight)
        return pips

    def compute_posterior_entropy(self) -> tuple[float, float]:
        """Compute posterior entropy H and effective number of models N_eff."""
        nonzero_w = self.weights_[self.weights_ > 1e-12]
        H = -float(np.sum(nonzero_w * np.log(nonzero_w)))
        n_eff = float(np.exp(H))
        return H, n_eff


def train_bma_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> tuple[BayesianModelAveragingClassifier, dict[str, Any]]:
    """Train Bayesian Model Averaging on training features only."""
    assert_no_forbidden_columns(X_train, context="train_bma_model")
    ensure_dir(MODELS_DIR)

    bma = BayesianModelAveragingClassifier(max_features_to_enumerate=8)
    bma.fit(X_train, y_train)

    pips = bma.compute_posterior_inclusion_probabilities()
    entropy, n_eff = bma.compute_posterior_entropy()

    metadata = {
        "model_name": "Bayesian Model Averaging (BMA)",
        "num_models_evaluated": len(bma.models_),
        "effective_num_models": n_eff,
        "posterior_entropy": entropy,
        "posterior_inclusion_probabilities": pips,
        "top_model_weight": float(bma.weights_.max()),
    }

    artifact_path = MODELS_DIR / "bma_model.joblib"
    save_model(bma, artifact_path, metadata=metadata)
    logger.info(f"Saved BMA model artifact to: {artifact_path.name}")
    print(f"\nBMA Summary:")
    print(f"  - Total Models in Space: {len(bma.models_)}")
    print(f"  - Effective Number of Models (N_eff): {n_eff:.2f}")
    print(f"  - Top Feature PIPs: {sorted(pips.items(), key=lambda x: x[1], reverse=True)[:5]}")

    return bma, metadata

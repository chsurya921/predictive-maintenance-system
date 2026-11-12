r"""
src/mars_model.py
=================
GENUINE From-Scratch Multivariate Adaptive Regression Splines (MARS).
Independent implementation containing zero py-earth dependencies.

Mathematical Algorithm & Implementation:
  1. Reflected Hinge Functions:
     h(x, t, s) = \max(0, s(x - t)) where s \in \{+1, -1\}
     yielding complementary pairs:
     h(x - t)_+ = \max(0, x - t)   and   h(t - x)_+ = \max(0, t - x)

  2. Greedy Forward Basis Selection:
     Starts with B_0(x) \equiv 1. Iteratively adds product pairs:
     B_m(x) \cdot \max(0, x_j - t)  and  B_m(x) \cdot \max(0, t - x_j)
     selecting the parent term B_m, predictor j, and knot location t that
     minimizes Residual Sum of Squares (RSS).

  3. Backward Pruning by Generalized Cross-Validation (GCV):
     GCV(M) = \frac{\frac{1}{N} \text{RSS}(M)}{\left(1 - \frac{C(M)}{N}\right)^2}
     where C(M) = \text{trace} + d \cdot \frac{M - 1}{2} is the complexity penalty
     governed by parameter d (default 2.0).

  4. MARS-Basis + Logistic GLM Classification Mode:
     Transforms raw predictors into the optimal pruned basis representation \Phi(X),
     then estimates maximum-likelihood parameters using regularized Logistic Regression.

Author: ML Engineering Team
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression

from config import (
    MODELS_DIR,
    RANDOM_STATE,
    COL_AIR_TEMP,
    COL_PROCESS_TEMP,
    COL_ROT_SPEED,
    COL_TORQUE,
    COL_TOOL_WEAR,
    COL_TYPE,
)
from utils.helpers import (
    assert_no_forbidden_columns,
    ensure_dir,
    get_logger,
    save_model,
)

logger = get_logger("mars_model")


class HingeTerm:
    r"""Represents a single hinge factor \max(0, s * (x_j - t))."""
    def __init__(self, feature_idx: int, knot: float, sign: int, feature_name: str = ""):
        self.feature_idx = feature_idx
        self.knot = float(knot)
        self.sign = int(sign)  # +1 for (x - t)_+, -1 for (t - x)_+
        self.feature_name = feature_name or f"x_{feature_idx}"

    def eval(self, X: np.ndarray) -> np.ndarray:
        val = self.sign * (X[:, self.feature_idx] - self.knot)
        return np.maximum(0.0, val)

    def __repr__(self) -> str:
        s_char = "+" if self.sign == 1 else "-"
        if self.sign == 1:
            return f"max(0, {self.feature_name} - {self.knot:.2f})"
        else:
            return f"max(0, {self.knot:.2f} - {self.feature_name})"


class BasisFunction:
    r"""Represents a product of one or more hinge factors: B_m(x) = \prod_k h_k(x)."""
    def __init__(self, terms: list[HingeTerm] | None = None):
        self.terms: list[HingeTerm] = terms or []

    @property
    def degree(self) -> int:
        return len(self.terms)

    @property
    def feature_indices(self) -> set[int]:
        return {t.feature_idx for t in self.terms}

    def eval(self, X: np.ndarray) -> np.ndarray:
        if not self.terms:
            return np.ones(len(X), dtype=float)
        res = np.ones(len(X), dtype=float)
        for term in self.terms:
            res *= term.eval(X)
        return res

    def __repr__(self) -> str:
        if not self.terms:
            return "1.0 (Intercept)"
        return " * ".join(repr(t) for t in self.terms)


class FromScratchMARSClassifier(BaseEstimator, ClassifierMixin):
    """
    From-Scratch Multivariate Adaptive Regression Splines (MARS) Classifier.
    Employs forward greedy knot selection, backward GCV pruning, and logistic GLM.
    """
    def __init__(
        self,
        max_terms: int = 16,
        max_degree: int = 2,
        penalty: float = 2.0,
        min_samples_leaf: int = 15,
        n_knots_per_feature: int = 15,
        class_weight: str | None = "balanced",
        random_state: int = RANDOM_STATE,
    ):
        self.max_terms = max_terms
        self.max_degree = max_degree
        self.penalty = penalty
        self.min_samples_leaf = min_samples_leaf
        self.n_knots_per_feature = n_knots_per_feature
        self.class_weight = class_weight
        self.random_state = random_state

        self.basis_functions_: list[BasisFunction] = []
        self.glm_model_: LogisticRegression | None = None
        self.feature_names_: list[str] = []
        self.classes_ = np.array([0, 1])

    def _prepare_data(self, X: pd.DataFrame | np.ndarray) -> tuple[np.ndarray, list[str]]:
        if isinstance(X, pd.DataFrame):
            assert_no_forbidden_columns(X, context="FromScratchMARS")
            feature_names = list(X.columns)
            # Map type column if present
            X_copy = X.copy()
            if COL_TYPE in X_copy.columns:
                type_map = {"L": 0.0, "M": 1.0, "H": 2.0}
                X_copy[COL_TYPE] = X_copy[COL_TYPE].map(type_map).fillna(0.0)
            X_mat = X_copy.values.astype(float)
            return X_mat, feature_names
        else:
            X_mat = np.asarray(X, dtype=float)
            feature_names = [f"x_{i}" for i in range(X_mat.shape[1])]
            return X_mat, feature_names

    def fit(self, X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray) -> "FromScratchMARSClassifier":
        X_mat, self.feature_names_ = self._prepare_data(X)
        y_vec = np.asarray(y, dtype=float).ravel()
        n_samples, n_features = X_mat.shape

        logger.info(f"Fitting From-Scratch MARS on shape {X_mat.shape} (max_terms={self.max_terms}, max_degree={self.max_degree})...")

        # -------------------------------------------------------------------
        # 1. Forward Pass: Greedy Addition of Hinge Pairs
        # -------------------------------------------------------------------
        bases = [BasisFunction()]  # B_0 \equiv 1
        design_cols = [bases[0].eval(X_mat)]

        # Precompute candidate knots per feature
        feature_knots: dict[int, np.ndarray] = {}
        for j in range(n_features):
            unique_vals = np.unique(X_mat[:, j])
            if len(unique_vals) <= self.n_knots_per_feature:
                feature_knots[j] = unique_vals
            else:
                qs = np.linspace(0.05, 0.95, self.n_knots_per_feature)
                feature_knots[j] = np.quantile(unique_vals, qs)

        while len(bases) + 2 <= self.max_terms:
            best_rss = float("inf")
            best_pair = None

            # Current design matrix
            Phi = np.column_stack(design_cols)
            # Baseline OLS
            beta, residuals, _, _ = np.linalg.lstsq(Phi, y_vec, rcond=None)
            current_rss = float(np.sum((y_vec - Phi @ beta) ** 2))

            for m_idx, parent_base in enumerate(bases):
                if parent_base.degree >= self.max_degree:
                    continue

                parent_eval = design_cols[m_idx]

                for j in range(n_features):
                    # Disallow repeating same feature in product
                    if j in parent_base.feature_indices:
                        continue

                    fname = self.feature_names_[j]
                    for t in feature_knots[j]:
                        h_pos = HingeTerm(j, t, +1, fname)
                        h_neg = HingeTerm(j, t, -1, fname)

                        b_pos = BasisFunction(parent_base.terms + [h_pos])
                        b_neg = BasisFunction(parent_base.terms + [h_neg])

                        pos_eval = parent_eval * h_pos.eval(X_mat)
                        neg_eval = parent_eval * h_neg.eval(X_mat)

                        # Minimum samples in non-zero region
                        if (pos_eval > 0).sum() < self.min_samples_leaf or (neg_eval > 0).sum() < self.min_samples_leaf:
                            continue

                        # Fit augmented model
                        Phi_cand = np.column_stack([Phi, pos_eval, neg_eval])
                        beta_c, _, _, _ = np.linalg.lstsq(Phi_cand, y_vec, rcond=None)
                        rss_cand = float(np.sum((y_vec - Phi_cand @ beta_c) ** 2))

                        if rss_cand < best_rss:
                            best_rss = rss_cand
                            best_pair = (b_pos, b_neg, pos_eval, neg_eval)

            if best_pair is None or best_rss >= current_rss - 1e-5:
                # No further improvement possible
                break

            # Append best hinge pair
            b_pos, b_neg, pos_eval, neg_eval = best_pair
            bases.extend([b_pos, b_neg])
            design_cols.extend([pos_eval, neg_eval])

        logger.info(f"Forward pass completed. Total bases created: {len(bases)}")

        # -------------------------------------------------------------------
        # 2. Backward Pass: Pruning by Generalized Cross-Validation (GCV)
        # -------------------------------------------------------------------
        best_gcv = float("inf")
        best_bases = list(bases)

        current_bases = list(bases)
        current_cols = list(design_cols)

        while len(current_bases) > 1:
            Phi_cur = np.column_stack(current_cols)
            beta_cur, _, _, _ = np.linalg.lstsq(Phi_cur, y_vec, rcond=None)
            rss_cur = float(np.sum((y_vec - Phi_cur @ beta_cur) ** 2))

            # GCV formula
            M = len(current_bases)
            effective_df = M + self.penalty * (M - 1) / 2.0
            denom = max(1e-4, 1.0 - effective_df / n_samples) ** 2
            gcv = (rss_cur / n_samples) / denom

            if gcv < best_gcv:
                best_gcv = gcv
                best_bases = list(current_bases)

            # Find basis to drop that causes least increase in RSS
            best_drop_idx = None
            min_drop_rss = float("inf")

            for k in range(1, len(current_bases)):  # never drop constant term B_0
                subset_cols = [c for idx, c in enumerate(current_cols) if idx != k]
                Phi_sub = np.column_stack(subset_cols)
                b_sub, _, _, _ = np.linalg.lstsq(Phi_sub, y_vec, rcond=None)
                rss_sub = float(np.sum((y_vec - Phi_sub @ b_sub) ** 2))

                if rss_sub < min_drop_rss:
                    min_drop_rss = rss_sub
                    best_drop_idx = k

            if best_drop_idx is not None:
                current_bases.pop(best_drop_idx)
                current_cols.pop(best_drop_idx)
            else:
                break

        self.basis_functions_ = best_bases
        logger.info(f"Backward GCV pruning completed. Retained {len(self.basis_functions_)} basis terms.")

        # -------------------------------------------------------------------
        # 3. GLM Classification Mode: Logistic Regression over MARS Basis
        # -------------------------------------------------------------------
        Phi_selected = self.transform(X)
        self.glm_model_ = LogisticRegression(
            class_weight=self.class_weight,
            random_state=self.random_state,
            max_iter=500,
        )
        self.glm_model_.fit(Phi_selected, y_vec.astype(int))
        logger.info("Fitted Logistic GLM over optimal MARS basis.")
        return self

    def transform(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Evaluate all retained MARS basis functions over X."""
        X_mat, _ = self._prepare_data(X)
        cols = [b.eval(X_mat) for b in self.basis_functions_]
        return np.column_stack(cols)

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if self.glm_model_ is None:
            raise RuntimeError("Model is not fitted yet.")
        Phi = self.transform(X)
        return self.glm_model_.predict_proba(Phi)

    def predict(self, X: pd.DataFrame | np.ndarray, threshold: float = 0.5) -> np.ndarray:
        probs = self.predict_proba(X)[:, 1]
        return (probs >= threshold).astype(int)

    def get_basis_descriptions(self) -> list[str]:
        """Return human-readable descriptions of all basis functions."""
        return [f"B_{i}: {repr(b)}" for i, b in enumerate(self.basis_functions_)]


def train_mars_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    max_terms: int = 18,
    max_degree: int = 2,
) -> tuple[FromScratchMARSClassifier, dict[str, Any]]:
    """Train From-Scratch MARS Classifier and save artifact."""
    assert_no_forbidden_columns(X_train, context="train_mars_model")
    ensure_dir(MODELS_DIR)

    clf = FromScratchMARSClassifier(
        max_terms=max_terms,
        max_degree=max_degree,
        penalty=2.0,
        class_weight="balanced",
    )
    clf.fit(X_train, y_train)

    basis_desc = clf.get_basis_descriptions()
    metadata = {
        "model_name": "From-Scratch MARS (Multivariate Adaptive Regression Splines)",
        "num_bases": len(clf.basis_functions_),
        "basis_descriptions": basis_desc,
        "max_degree": max_degree,
    }

    artifact_path = MODELS_DIR / "from_scratch_mars_model.joblib"
    save_model(clf, artifact_path, metadata=metadata)
    logger.info(f"Saved from-scratch MARS model artifact to: {artifact_path.name}")
    print(f"\nFrom-Scratch MARS Trained with {len(clf.basis_functions_)} basis terms:")
    for b in basis_desc[:6]:
        print(f"  - {b}")
    return clf, metadata

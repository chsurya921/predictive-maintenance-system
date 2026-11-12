"""
utils/helpers.py
================
Shared utility functions used across all modules.

Conventions:
  - All functions accept and return plain Python / NumPy / Pandas objects.
  - No global side-effects; callers control logging and I/O.
  - Type hints on every public function.

Author: ML Engineering Team
"""

from __future__ import annotations

import json
import logging
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn
import joblib


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a module-level logger with a consistent format."""
    import sys
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        fmt = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def ensure_dir(path: Path) -> Path:
    """Create *path* (and parents) if it does not exist; return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path


# ---------------------------------------------------------------------------
# Model persistence
# ---------------------------------------------------------------------------

def save_model(
    estimator: Any,
    path: Path,
    metadata: dict[str, Any] | None = None,
) -> None:
    """
    Persist *estimator* to *path* (joblib) and write a companion JSON file
    with *metadata* (versions, seed, threshold, feature list, …).

    Parameters
    ----------
    estimator : fitted sklearn-compatible estimator
    path      : target *.joblib* file path
    metadata  : dict of JSON-serialisable values; augmented with package
                versions and a fit timestamp automatically.
    """
    ensure_dir(path.parent)
    joblib.dump(estimator, path, compress=3)

    meta = metadata or {}
    meta.setdefault("fit_timestamp", time.strftime("%Y-%m-%dT%H:%M:%S"))
    meta.setdefault("python_version", platform.python_version())
    meta.setdefault("sklearn_version", sklearn.__version__)
    meta.setdefault("numpy_version", np.__version__)
    meta.setdefault("pandas_version", pd.__version__)

    meta_path = path.with_suffix(".json")
    with meta_path.open("w") as fh:
        json.dump(meta, fh, indent=2, default=str)


def load_model(path: Path) -> Any:
    """Load and return a joblib-persisted estimator."""
    if not path.exists():
        raise FileNotFoundError(f"Model artifact not found: {path}")
    return joblib.load(path)


def load_model_metadata(path: Path) -> dict[str, Any]:
    """Load the companion JSON metadata for a model artifact."""
    meta_path = path.with_suffix(".json")
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata not found: {meta_path}")
    with meta_path.open() as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Split-index helpers
# ---------------------------------------------------------------------------

def save_split_indices(
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    path: Path,
) -> None:
    """Save train/test integer indices to a compressed .npz file."""
    ensure_dir(path.parent)
    np.savez_compressed(path, train=train_idx, test=test_idx)


def load_split_indices(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load train/test integer indices from an .npz file."""
    if not path.exists():
        raise FileNotFoundError(f"Split indices not found: {path}")
    data = np.load(path)
    return data["train"], data["test"]


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def bootstrap_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    metric_fn: Any,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    random_state: int = 42,
) -> tuple[float, float, float]:
    """
    Compute a bootstrap confidence interval for *metric_fn(y_true, y_score)*.

    Returns
    -------
    point_estimate, lower_ci, upper_ci
    """
    rng = np.random.default_rng(random_state)
    n = len(y_true)
    scores: list[float] = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        try:
            s = metric_fn(y_true[idx], y_score[idx])
            scores.append(float(s))
        except Exception:
            pass  # skip degenerate bootstrap samples

    scores_arr = np.array(scores)
    alpha = (1 - ci) / 2
    lower = float(np.quantile(scores_arr, alpha))
    upper = float(np.quantile(scores_arr, 1 - alpha))
    point = metric_fn(y_true, y_score)
    return float(point), lower, upper


# ---------------------------------------------------------------------------
# Data-quality guard
# ---------------------------------------------------------------------------

FORBIDDEN_COLUMNS = {
    "udi", "product_id",         # identifiers
    "twf", "hdf", "pwf", "osf", "rnf",  # failure-mode flags
}


def assert_no_forbidden_columns(
    X: pd.DataFrame | np.ndarray,
    context: str = "",
) -> None:
    """
    Raise ValueError if any forbidden column name appears in X.

    This check enforces Rule 2 (no data leakage) from the project spec.
    It is called by every model's fit wrapper.

    Parameters
    ----------
    X       : feature matrix; must be a DataFrame (column names checked)
              or a numpy array (skipped – no column names to check).
    context : human-readable label for the error message.
    """
    if not isinstance(X, pd.DataFrame):
        return  # numpy arrays have no column names to check
    bad = FORBIDDEN_COLUMNS & set(c.lower() for c in X.columns)
    if bad:
        raise ValueError(
            f"[Leakage detected] Forbidden column(s) {bad} found in features"
            f"{' (' + context + ')' if context else ''}. "
            "Remove identifiers and failure-mode flags before training."
        )


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------

def set_all_seeds(seed: int = 42) -> None:
    """Set random seeds for Python, NumPy (and optionally torch if present)."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch  # noqa: F401
        torch.manual_seed(seed)
    except ImportError:
        pass


def pretty_dict(d: dict[str, Any], title: str = "") -> str:
    """Return a neatly formatted multi-line string representation of *d*."""
    lines = []
    if title:
        lines.append(f"=== {title} ===")
    max_k = max((len(str(k)) for k in d), default=0)
    for k, v in d.items():
        lines.append(f"  {str(k):<{max_k}} : {v}")
    return "\n".join(lines)

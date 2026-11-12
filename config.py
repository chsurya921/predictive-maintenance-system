"""
config.py
=========
Central configuration for the Predictive Maintenance System.

All paths, seeds, thresholds, and hyper-parameter search spaces live here.
Nothing should be hard-coded in model or script files; import from this module.

Author: ML Engineering Team
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Project root (resolved relative to this file so the project is portable)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Data paths
# ---------------------------------------------------------------------------
DATA_DIR        = PROJECT_ROOT / "data"
RAW_DATA_PATH   = DATA_DIR / "ai4i2020.csv"
SPLIT_INDICES   = DATA_DIR / "split_indices.npz"   # saved train/test indices

# ---------------------------------------------------------------------------
# Output paths
# ---------------------------------------------------------------------------
MODELS_DIR      = PROJECT_ROOT / "models" / "trained_models"
RESULTS_DIR     = PROJECT_ROOT / "results"
VIZ_DIR         = RESULTS_DIR / "visualizations"
METRICS_CSV     = RESULTS_DIR / "evaluation_metrics.csv"

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_STATE    = 42
N_CV_FOLDS      = 5          # stratified k-fold for tuning / BMA weights

# ---------------------------------------------------------------------------
# Train / test split
# ---------------------------------------------------------------------------
TEST_SIZE       = 0.20       # 20 % held-out test set

# ---------------------------------------------------------------------------
# Class imbalance
# ---------------------------------------------------------------------------
# Failure prevalence ≈ 3.39 %  →  positive-class weight for XGBoost
# scale_pos_weight ≈ (n_neg / n_pos).  Exact value recomputed from training
# data at fit time; this is the approximate expected value.
APPROX_SCALE_POS_WEIGHT = 29  # (9661 / 339) ≈ 28.5, rounded up

# ---------------------------------------------------------------------------
# Feature engineering column names (after renaming)
# ---------------------------------------------------------------------------
# Raw sensor columns (after safe renaming)
COL_AIR_TEMP        = "air_temperature_k"
COL_PROCESS_TEMP    = "process_temperature_k"
COL_ROT_SPEED       = "rotational_speed_rpm"
COL_TORQUE          = "torque_nm"
COL_TOOL_WEAR       = "tool_wear_min"
COL_TYPE            = "type"
COL_TARGET          = "machine_failure"

# Failure-mode flag columns – NEVER allowed to reach a failure model
FAILURE_MODE_COLS   = ["twf", "hdf", "pwf", "osf", "rnf"]

# Identifier columns – NEVER allowed to reach a model
IDENTIFIER_COLS     = ["udi", "product_id"]

# Engineered features
COL_TEMP_DIFF       = "temp_diff_k"            # process_temp - air_temp
COL_MECH_POWER      = "mechanical_power_w"      # torque * rot_speed (rad/s)
COL_TORQUE_WEAR     = "torque_wear_proxy"        # torque * tool_wear

# ---------------------------------------------------------------------------
# Recommendation engine thresholds
# (computed from training-data percentiles; updated in Stage 2 and stored here)
# ---------------------------------------------------------------------------
# These will be overwritten by compute_recommendation_thresholds() in Stage 2.
# Defaults below are approximate domain values for documentation purposes.
RECOMMENDATION_THRESHOLDS: dict = {
    "tool_wear_warning_pct":  75,    # percentile above which we warn
    "tool_wear_critical_pct": 95,
    "torque_warning_pct":     90,
    "process_temp_warning_pct": 90,
    "rot_speed_low_pct":      5,
    "rot_speed_high_pct":     95,
}

# ---------------------------------------------------------------------------
# Probability bands for Normal / Warning / Critical classification
# (thresholds set from CV/validation predictions in Stage 5)
# ---------------------------------------------------------------------------
PROB_WARNING_THRESHOLD  = 0.30   # will be updated after Stage 5 tuning
PROB_CRITICAL_THRESHOLD = 0.60   # will be updated after Stage 5 tuning

# ---------------------------------------------------------------------------
# Model metadata keys (stored alongside .joblib artifacts as JSON)
# ---------------------------------------------------------------------------
MODEL_METADATA_KEYS = [
    "model_name",
    "sklearn_version",
    "numpy_version",
    "pandas_version",
    "fit_timestamp",
    "random_state",
    "train_indices_path",
    "features_used",
    "threshold_used",
    "pr_auc_cv_mean",
    "pr_auc_cv_std",
]

# ---------------------------------------------------------------------------
# Logging format
# ---------------------------------------------------------------------------
LOG_FORMAT  = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
LOG_DATEFMT = "%Y-%m-%d %H:%M:%S"

"""
tests/test_stage2_preprocessing.py
==================================
Automated unit tests for Stage 2: Data Preprocessing & Feature Engineering.

Acceptance Criteria Verified:
  1. No Data Leakage: UDI, Product ID, and the 5 failure mode flags (TWF, HDF, PWF, OSF, RNF)
     never appear in X_train or X_test.
  2. Split Reproducibility: Saved indices in split_indices.npz produce identical splits.
  3. Stratification Parity: Failure rate in train and test sets both equal ~3.39%.
  4. No NaNs: No null values in raw or engineered feature matrices.
  5. Mathematical Correctness of Physics-Derived Features:
     - temp_diff_k = process_temperature_k - air_temperature_k
     - mechanical_power_w = torque_nm * (rotational_speed_rpm * 2 * pi / 60)
     - torque_wear_proxy = torque_nm * tool_wear_min
  6. Preprocessor Compatibility: ColumnTransformer works inside an sklearn Pipeline.

Author: ML Engineering Team
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from config import (
    RAW_DATA_PATH,
    SPLIT_INDICES,
    FAILURE_MODE_COLS,
    IDENTIFIER_COLS,
    COL_TARGET,
    COL_AIR_TEMP,
    COL_PROCESS_TEMP,
    COL_ROT_SPEED,
    COL_TORQUE,
    COL_TOOL_WEAR,
    COL_TEMP_DIFF,
    COL_MECH_POWER,
    COL_TORQUE_WEAR,
)
from src.data_preprocessing import (
    load_and_clean_data,
    create_and_save_split_indices,
    get_train_test_data,
    build_preprocessor,
)
from src.feature_engineering import (
    engineer_features,
    compute_recommendation_thresholds,
)
from utils.helpers import assert_no_forbidden_columns


# ===========================================================================
# 1. Cleaning & Safe Column Names
# ===========================================================================

def test_load_and_clean_data_columns():
    """Verify clean dataset column names and data shape."""
    df = load_and_clean_data(RAW_DATA_PATH)
    assert len(df) == 10000
    assert df.isnull().sum().sum() == 0

    expected_cols = {
        "udi", "product_id", "type", "air_temperature_k", "process_temperature_k",
        "rotational_speed_rpm", "torque_nm", "tool_wear_min", "machine_failure",
        "twf", "hdf", "pwf", "osf", "rnf"
    }
    assert expected_cols.issubset(set(df.columns))


# ===========================================================================
# 2. Rule 2: Strict No Data Leakage
# ===========================================================================

def test_no_forbidden_columns_in_train_test():
    """Verify that neither X_train nor X_test contain any identifier or failure mode flag."""
    X_train, X_test, y_train, y_test = get_train_test_data(include_engineered=True)

    forbidden = set(IDENTIFIER_COLS + FAILURE_MODE_COLS + [COL_TARGET])
    train_overlap = forbidden & set(X_train.columns)
    test_overlap = forbidden & set(X_test.columns)

    assert not train_overlap, f"Data leakage! Forbidden columns found in X_train: {train_overlap}"
    assert not test_overlap, f"Data leakage! Forbidden columns found in X_test: {test_overlap}"

    # Verify via helper assertion
    assert_no_forbidden_columns(X_train, context="test X_train")
    assert_no_forbidden_columns(X_test, context="test X_test")


# ===========================================================================
# 3. Rule 3: Fixed Stratified Split Reproducibility & Parity
# ===========================================================================

def test_split_indices_reproducibility():
    """Verify that loading split indices multiple times returns identical rows."""
    X_train1, X_test1, y_train1, y_test1 = get_train_test_data(include_engineered=False)
    X_train2, X_test2, y_train2, y_test2 = get_train_test_data(include_engineered=False)

    pd.testing.assert_frame_equal(X_train1, X_train2)
    pd.testing.assert_frame_equal(X_test1, X_test2)
    pd.testing.assert_series_equal(y_train1, y_train2)
    pd.testing.assert_series_equal(y_test1, y_test2)


def test_stratification_parity():
    """Verify 80/20 split ratio and exact stratified failure rates."""
    X_train, X_test, y_train, y_test = get_train_test_data(include_engineered=False)

    assert len(X_train) == 8000, f"Expected 8000 train rows, got {len(X_train)}"
    assert len(X_test) == 2000, f"Expected 2000 test rows, got {len(X_test)}"

    train_rate = y_train.mean()
    test_rate = y_test.mean()

    # Exact failures count
    assert y_train.sum() == 271, f"Expected 271 train failures, got {y_train.sum()}"
    assert y_test.sum() == 68, f"Expected 68 test failures, got {y_test.sum()}"
    assert (y_train.sum() + y_test.sum()) == 339

    # Both failure rates must equal ~3.39% within 0.05% margin
    assert abs(train_rate - 0.033875) < 1e-4
    assert abs(test_rate - 0.0340) < 1e-4


# ===========================================================================
# 4. Feature Engineering Mathematical Correctness & No NaNs
# ===========================================================================

def test_feature_engineering_math():
    """Verify the exact physical calculation of all 3 engineered features."""
    df = load_and_clean_data(RAW_DATA_PATH)
    raw_features = df.drop(columns=[COL_TARGET] + IDENTIFIER_COLS + FAILURE_MODE_COLS)
    eng_features = engineer_features(raw_features)

    assert eng_features.isnull().sum().sum() == 0, "Engineered features contain NaNs"

    # 1. Delta T check
    expected_delta_t = raw_features[COL_PROCESS_TEMP] - raw_features[COL_AIR_TEMP]
    np.testing.assert_allclose(eng_features[COL_TEMP_DIFF], expected_delta_t)

    # 2. Mechanical Power check: P = \tau * (RPM * 2 * \pi / 60)
    expected_power = raw_features[COL_TORQUE] * (raw_features[COL_ROT_SPEED] * 2.0 * np.pi / 60.0)
    np.testing.assert_allclose(eng_features[COL_MECH_POWER], expected_power)

    # 3. Torque * Tool Wear check
    expected_strain = raw_features[COL_TORQUE] * raw_features[COL_TOOL_WEAR]
    np.testing.assert_allclose(eng_features[COL_TORQUE_WEAR], expected_strain)


def test_feature_engineering_without_leakage():
    """Ensure feature engineering raises ValueError if bad columns reach it."""
    bad_df = pd.DataFrame({
        COL_PROCESS_TEMP: [310.0],
        COL_AIR_TEMP: [300.0],
        COL_TORQUE: [40.0],
        COL_ROT_SPEED: [1500],
        COL_TOOL_WEAR: [50],
        "twf": [0]  # forbidden!
    })
    with pytest.raises(ValueError, match="Forbidden column"):
        engineer_features(bad_df)


# ===========================================================================
# 5. Preprocessor Integration inside Pipeline
# ===========================================================================

@pytest.mark.parametrize("encoding", ["onehot", "ordinal"])
def test_preprocessor_inside_pipeline(encoding):
    """Verify that build_preprocessor fits and transforms properly within an sklearn Pipeline."""
    X_train, X_test, y_train, y_test = get_train_test_data(include_engineered=True)

    preprocessor = build_preprocessor(include_engineered=True, encoding=encoding)
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=200, random_state=42))
    ])

    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)
    probs = pipeline.predict_proba(X_test)

    assert len(preds) == len(X_test)
    assert probs.shape == (len(X_test), 2)
    assert not np.isnan(probs).any()


def test_training_only_thresholds():
    """Verify that recommendation thresholds are calculated strictly on training data."""
    X_train, X_test, y_train, y_test = get_train_test_data(include_engineered=True)
    thresholds = compute_recommendation_thresholds(X_train)

    assert "tool_wear_warning" in thresholds
    assert "tool_wear_critical" in thresholds
    assert "torque_warning" in thresholds
    assert thresholds["tool_wear_critical"] > thresholds["tool_wear_warning"]

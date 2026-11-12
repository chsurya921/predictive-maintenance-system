"""
tests/test_stage1_eda.py
========================
Automated test suite for Stage 1: Setup, Data Integrity, and Exploratory Data Analysis.

Verifies:
  - Exact dataset properties and integrity (10,000 rows, 14 columns, no nulls, no duplicates)
  - Target distribution (339 failures, 9661 normal)
  - Exact failure mode counts and anomalous overlap counts
  - Verification of saved visualizations and summary artifacts
  - Validation of the executed EDA notebook
  - Initial test of the anti-leakage column guard

Author: ML Engineering Team
"""

import os
from pathlib import Path
import pytest
import pandas as pd
import numpy as np
import nbformat as nbf

from config import (
    RAW_DATA_PATH,
    VIZ_DIR,
    PROJECT_ROOT,
    FAILURE_MODE_COLS,
    IDENTIFIER_COLS,
)
from utils.helpers import assert_no_forbidden_columns


@pytest.fixture(scope="module")
def raw_df():
    """Load the raw AI4I 2020 dataset once for the test module."""
    assert RAW_DATA_PATH.exists(), f"Raw dataset missing at {RAW_DATA_PATH}"
    return pd.read_csv(RAW_DATA_PATH)


# ===========================================================================
# 1. Dataset Integrity & Shape Tests
# ===========================================================================

def test_dataset_shape_and_completeness(raw_df):
    """Verify exact dimensions, zero missing values, and zero duplicates."""
    assert raw_df.shape == (10000, 14), f"Expected shape (10000, 14), got {raw_df.shape}"
    assert raw_df.isnull().sum().sum() == 0, "Dataset contains missing values"
    assert raw_df.duplicated().sum() == 0, "Dataset contains duplicate rows"


def test_expected_columns_present(raw_df):
    """Verify all 14 expected raw columns exist."""
    expected_cols = [
        "UDI", "Product ID", "Type", "Air temperature [K]", "Process temperature [K]",
        "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]", "Machine failure",
        "TWF", "HDF", "PWF", "OSF", "RNF"
    ]
    assert list(raw_df.columns) == expected_cols, f"Column mismatch: {list(raw_df.columns)}"


# ===========================================================================
# 2. Target & Categorical Distribution Tests
# ===========================================================================

def test_type_counts(raw_df):
    """Verify product Type distribution matches known UCI facts (L: 6000, M: 2997, H: 1003)."""
    counts = raw_df["Type"].value_counts().to_dict()
    assert counts == {"L": 6000, "M": 2997, "H": 1003}, f"Unexpected Type counts: {counts}"


def test_target_imbalance_counts(raw_df):
    """Verify Machine failure distribution matches 339 failures (3.39%) and 9661 normal (96.61%)."""
    counts = raw_df["Machine failure"].value_counts().to_dict()
    assert counts == {0: 9661, 1: 339}, f"Unexpected target counts: {counts}"
    prevalence = counts[1] / len(raw_df)
    assert abs(prevalence - 0.0339) < 1e-4, f"Prevalence is {prevalence}, expected 0.0339"


# ===========================================================================
# 3. Failure Mode & Anomaly Overlap Tests
# ===========================================================================

def test_individual_failure_mode_counts(raw_df):
    """Verify individual failure flags match exact UCI totals."""
    expected_flags = {"TWF": 46, "HDF": 115, "PWF": 95, "OSF": 98, "RNF": 19}
    for flag, expected in expected_flags.items():
        actual = int(raw_df[flag].sum())
        assert actual == expected, f"Flag {flag} count: {actual}, expected {expected}"


def test_failed_rows_overlap_breakdown(raw_df):
    """
    Verify that out of 339 failed rows:
      - 306 have exactly 1 failure mode flag
      - 24 have 2 or more failure mode flags
      - 9 have 0 failure mode flags (Anomaly A)
    """
    flags = ["TWF", "HDF", "PWF", "OSF", "RNF"]
    failed = raw_df[raw_df["Machine failure"] == 1]
    flag_sums = failed[flags].sum(axis=1)

    n_zero = int((flag_sums == 0).sum())
    n_one = int((flag_sums == 1).sum())
    n_multi = int((flag_sums >= 2).sum())

    assert n_one == 306, f"Expected 306 single-flag failures, got {n_one}"
    assert n_multi == 24, f"Expected 24 multi-flag failures, got {n_multi}"
    assert n_zero == 9, f"Expected 9 unlabeled failures (Anomaly A), got {n_zero}"
    assert (n_zero + n_one + n_multi) == 339, "Sum of breakdown does not equal 339"


def test_non_failed_flagged_rows_anomaly(raw_df):
    """
    Verify that exactly 18 rows have Machine failure == 0 but at least one failure flag set,
    and all 18 of those rows are RNF == 1 (Anomaly B).
    """
    flags = ["TWF", "HDF", "PWF", "OSF", "RNF"]
    non_failed = raw_df[raw_df["Machine failure"] == 0]
    flagged = non_failed[non_failed[flags].sum(axis=1) > 0]

    assert len(flagged) == 18, f"Expected 18 flagged non-failure rows, got {len(flagged)}"
    assert int(flagged["RNF"].sum()) == 18, "Expected all 18 anomalous rows to be RNF"
    for other in ["TWF", "HDF", "PWF", "OSF"]:
        assert int(flagged[other].sum()) == 0, f"Expected 0 {other} flags in non-failed rows"


# ===========================================================================
# 4. Sensor Value Bounds Tests
# ===========================================================================

def test_sensor_physical_ranges(raw_df):
    """Sanity check that sensor readings are physically plausible."""
    # Temperatures in Kelvin (milling machine operating range ~295K to 315K)
    assert 290.0 < raw_df["Air temperature [K]"].min() < raw_df["Air temperature [K]"].max() < 310.0
    assert 300.0 < raw_df["Process temperature [K]"].min() < raw_df["Process temperature [K]"].max() < 320.0
    # Process temp is always higher than air temp
    assert (raw_df["Process temperature [K]"] > raw_df["Air temperature [K]"]).all()
    # Rotational speed (RPM > 1000)
    assert raw_df["Rotational speed [rpm]"].min() >= 1100
    assert raw_df["Rotational speed [rpm]"].max() <= 3000
    # Torque (Nm > 0)
    assert raw_df["Torque [Nm]"].min() > 0
    # Tool wear (minutes >= 0)
    assert raw_df["Tool wear [min]"].min() >= 0


# ===========================================================================
# 5. Visualizations & Outputs Artifact Tests
# ===========================================================================

def test_visualizations_exist_and_non_empty():
    """Verify that all required publication plots were generated in results/visualizations/."""
    expected_plots = [
        "01_class_and_type_distribution.png",
        "02_failure_modes_and_cooccurrence.png",
        "03_sensor_distributions_by_type.png",
        "04_sensor_distributions_by_failure.png",
        "05_correlation_heatmap.png",
        "06_failure_rate_by_type_and_tool_wear.png",
        "07_physics_sensor_regimes.png",
        "sensor_summary_and_outliers.csv",
        "failure_rate_by_type.csv",
        "failure_rate_by_tool_wear.csv",
    ]
    for filename in expected_plots:
        path = VIZ_DIR / filename
        assert path.exists(), f"Expected artifact missing: {path}"
        assert path.stat().st_size > 0, f"Artifact is empty: {path}"


# ===========================================================================
# 6. Notebook Acceptance Test
# ===========================================================================

def test_eda_notebook_is_executed_top_to_bottom():
    """Verify that exploratory_analysis.ipynb exists and has outputs in code cells."""
    nb_path = PROJECT_ROOT / "notebooks" / "exploratory_analysis.ipynb"
    assert nb_path.exists(), f"EDA Notebook missing at {nb_path}"

    with open(nb_path, "r", encoding="utf-8") as f:
        nb = nbf.read(f, as_version=4)

    code_cells = [c for c in nb.cells if c.cell_type == "code"]
    assert len(code_cells) >= 8, f"Expected at least 8 code cells, found {len(code_cells)}"

    # Check that all code cells have execution outputs
    for i, cell in enumerate(code_cells):
        assert len(cell.outputs) > 0, f"Code cell {i+1} has no outputs recorded"


# ===========================================================================
# 7. Data Leakage Prevention Unit Test
# ===========================================================================

def test_leakage_guard_catches_forbidden_columns():
    """
    Enforces Rule 2: assert_no_forbidden_columns MUST raise a ValueError
    whenever any identifier or failure mode flag appears in a DataFrame.
    """
    # Safe feature frame
    safe_df = pd.DataFrame({
        "air_temperature_k": [300.0],
        "torque_nm": [40.0],
        "type": ["L"]
    })
    # Should not raise
    assert_no_forbidden_columns(safe_df, context="test_safe")

    # Forbidden frames
    for bad_col in ["udi", "product_id", "twf", "hdf", "pwf", "osf", "rnf"]:
        leaked_df = safe_df.copy()
        leaked_df[bad_col] = 1
        with pytest.raises(ValueError, match="Forbidden column"):
            assert_no_forbidden_columns(leaked_df, context=f"test_leak_{bad_col}")

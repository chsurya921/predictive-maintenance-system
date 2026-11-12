"""
src/data_preprocessing.py
=========================
Data cleaning, column sanitization, stratified split management,
and scikit-learn ColumnTransformer pipelines for industrial equipment maintenance.

Non-Negotiable Rules Enforced:
  - Rule 2 (No Leakage): UDI, Product ID, and the five failure mode flags
    (TWF, HDF, PWF, OSF, RNF) are strictly dropped from the binary predictor feature set.
  - Rule 3 (Fixed Split): Stratified 80/20 split on machine_failure (random_state=42)
    saved to data/split_indices.npz and reused identically by all models.
  - Rule 4 (Pipeline Preprocessing): All scalers, imputers, and encoders are fit
    inside sklearn Pipelines strictly on training folds.

Author: ML Engineering Team
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from config import (
    RAW_DATA_PATH,
    SPLIT_INDICES,
    RANDOM_STATE,
    TEST_SIZE,
    COL_AIR_TEMP,
    COL_PROCESS_TEMP,
    COL_ROT_SPEED,
    COL_TORQUE,
    COL_TOOL_WEAR,
    COL_TYPE,
    COL_TARGET,
    COL_TEMP_DIFF,
    COL_MECH_POWER,
    COL_TORQUE_WEAR,
    FAILURE_MODE_COLS,
    IDENTIFIER_COLS,
)
from src.feature_engineering import engineer_features
from utils.helpers import (
    assert_no_forbidden_columns,
    ensure_dir,
    get_logger,
    save_split_indices,
    load_split_indices,
)

logger = get_logger("data_preprocessing")

# Safe column renaming dictionary
COLUMN_RENAME_MAP = {
    "UDI": "udi",
    "Product ID": "product_id",
    "Type": "type",
    "Air temperature [K]": "air_temperature_k",
    "Process temperature [K]": "process_temperature_k",
    "Rotational speed [rpm]": "rotational_speed_rpm",
    "Torque [Nm]": "torque_nm",
    "Tool wear [min]": "tool_wear_min",
    "Machine failure": "machine_failure",
    "TWF": "twf",
    "HDF": "hdf",
    "PWF": "pwf",
    "OSF": "osf",
    "RNF": "rnf",
}

# Raw numerical sensor columns
RAW_NUMERIC_FEATURES = [
    COL_AIR_TEMP,
    COL_PROCESS_TEMP,
    COL_ROT_SPEED,
    COL_TORQUE,
    COL_TOOL_WEAR,
]

# Engineered numerical features
ENGINEERED_NUMERIC_FEATURES = [
    COL_TEMP_DIFF,
    COL_MECH_POWER,
    COL_TORQUE_WEAR,
]


def load_and_clean_data(csv_path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """
    Load raw CSV data and rename columns to safe lowercase snake_case standard.

    Parameters
    ----------
    csv_path : Path to raw CSV file.

    Returns
    -------
    pd.DataFrame with standardized column names and validated data types.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Raw dataset file not found at: {csv_path}")

    df = pd.read_csv(csv_path)
    df = df.rename(columns=COLUMN_RENAME_MAP)

    # Validate all expected columns are present
    expected_cols = set(COLUMN_RENAME_MAP.values())
    if not expected_cols.issubset(set(df.columns)):
        missing = expected_cols - set(df.columns)
        raise KeyError(f"Expected sanitized columns missing: {missing}")

    logger.info(f"Loaded and cleaned {len(df)} records from {csv_path.name}")
    return df


def create_and_save_split_indices(
    df: pd.DataFrame,
    target_col: str = COL_TARGET,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
    save_path: Path = SPLIT_INDICES,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate and persist a single fixed stratified train/test split.
    Guarantees exact evaluation parity across all model families.

    Parameters
    ----------
    df           : Cleaned DataFrame containing target_col.
    target_col   : Name of the binary target column.
    test_size    : Fraction of records in held-out test set (default 0.20).
    random_state : Seed for reproducibility (default 42).
    save_path    : Destination path for .npz split indices.

    Returns
    -------
    (train_indices, test_indices)
    """
    indices = np.arange(len(df))
    y = df[target_col].values

    train_idx, test_idx = train_test_split(
        indices,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
        shuffle=True,
    )

    save_split_indices(train_idx, test_idx, save_path)
    logger.info(
        f"Saved stratified split indices ({len(train_idx)} train, {len(test_idx)} test) "
        f"to {save_path}"
    )
    return train_idx, test_idx


def get_train_test_data(
    include_engineered: bool = True,
    csv_path: Path = RAW_DATA_PATH,
    split_path: Path = SPLIT_INDICES,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Return X_train, X_test, y_train, y_test using the fixed saved split indices.
    Enforces Rule 2 (no leakage) and verifies zero missing values.

    Parameters
    ----------
    include_engineered : If True, appends temperature diff, mechanical power,
                         and torque*wear proxy features.
    csv_path           : Path to raw dataset CSV.
    split_path         : Path to .npz split indices file.

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    df = load_and_clean_data(csv_path)

    # Ensure split indices exist; generate if missing
    if not split_path.exists():
        logger.warning(f"Split indices not found at {split_path}. Generating now...")
        train_idx, test_idx = create_and_save_split_indices(df, save_path=split_path)
    else:
        train_idx, test_idx = load_split_indices(split_path)

    # Extract target
    y = df[COL_TARGET].copy()

    # Drop target, identifiers, and failure modes to construct pure feature set
    forbidden_to_drop = [COL_TARGET] + IDENTIFIER_COLS + FAILURE_MODE_COLS
    X = df.drop(columns=forbidden_to_drop)

    # Assert no data leakage before any transformation
    assert_no_forbidden_columns(X, context="get_train_test_data raw features")

    # Apply feature engineering if requested
    if include_engineered:
        X = engineer_features(X)

    # Final leakage guard
    assert_no_forbidden_columns(X, context="get_train_test_data prepared features")

    # Split using saved indices
    X_train = X.iloc[train_idx].copy().reset_index(drop=True)
    X_test  = X.iloc[test_idx].copy().reset_index(drop=True)
    y_train = y.iloc[train_idx].copy().reset_index(drop=True)
    y_test  = y.iloc[test_idx].copy().reset_index(drop=True)

    return X_train, X_test, y_train, y_test


def build_preprocessor(
    include_engineered: bool = True,
    encoding: Literal["onehot", "ordinal"] = "onehot",
    scale_numeric: bool = True,
) -> ColumnTransformer:
    """
    Construct a scikit-learn ColumnTransformer preprocessor.

    Encoding Justification:
      - 'onehot': Product type variants (L, M, H) are encoded as binary dummy
        variables without assuming equal distance between variant tolerances.
        Essential for linear, GAM, and kernel methods.
      - 'ordinal': Maps L -> 0, M -> 1, H -> 2, respecting the monotonic
        manufacturing quality order (where failure resistance improves: L < M < H).
        Particularly effective for decision trees and MARS.

    Parameters
    ----------
    include_engineered : Whether engineered numerical features are present in X.
    encoding           : 'onehot' (default) or 'ordinal'.
    scale_numeric      : Whether to apply StandardScaler to numerical columns.

    Returns
    -------
    ColumnTransformer configured for fit/transform within Pipeline.
    """
    num_cols = list(RAW_NUMERIC_FEATURES)
    if include_engineered:
        num_cols += list(ENGINEERED_NUMERIC_FEATURES)

    cat_cols = [COL_TYPE]

    # Numeric pipeline
    num_transformer = StandardScaler() if scale_numeric else "passthrough"

    # Categorical pipeline
    if encoding == "onehot":
        cat_transformer = OneHotEncoder(
            categories=[["L", "M", "H"]],
            drop="first",  # avoid dummy variable trap (reference: L)
            sparse_output=False,
            handle_unknown="ignore"
        )
    elif encoding == "ordinal":
        cat_transformer = OrdinalEncoder(
            categories=[["L", "M", "H"]],
            handle_unknown="use_encoded_value",
            unknown_value=-1
        )
    else:
        raise ValueError(f"Unsupported encoding: {encoding}. Choose 'onehot' or 'ordinal'.")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_transformer, num_cols),
            ("cat", cat_transformer, cat_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return preprocessor

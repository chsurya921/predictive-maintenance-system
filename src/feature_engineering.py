r"""
src/feature_engineering.py
==========================
Physics-derived feature engineering for industrial predictive maintenance.

Features engineered:
  1. Temperature Difference (temp_diff_k):
     Process temperature minus Air temperature: \Delta T = T_{process} - T_{air}.
     Directly models Heat Dissipation Failure (HDF) which triggers when \Delta T < 8.6 K
     in conjunction with low rotational speed (< 1380 rpm).

  2. Mechanical Power in Watts (mechanical_power_w):
     Power = Torque \times Angular Speed = \tau \cdot \left(\frac{2 \pi \cdot \text{RPM}}{60}\right).
     Directly models Power Failure (PWF), which occurs when mechanical power falls
     below 3500 W or exceeds 9000 W.

  3. Torque \times Tool Wear (torque_wear_proxy):
     Strain proxy = \tau \cdot \text{tool\_wear}.
     Directly models Overstrain Failure (OSF), which triggers when the cumulative
     mechanical stress exceeds product-type dependent endurance limits
     (11,000 for L, 12,000 for M, 13,000 for H).

Author: ML Engineering Team
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Any

from config import (
    COL_AIR_TEMP,
    COL_PROCESS_TEMP,
    COL_ROT_SPEED,
    COL_TORQUE,
    COL_TOOL_WEAR,
    COL_TEMP_DIFF,
    COL_MECH_POWER,
    COL_TORQUE_WEAR,
)
from utils.helpers import assert_no_forbidden_columns, get_logger

logger = get_logger("feature_engineering")


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate physics-derived maintenance features from raw sensor readings.

    Parameters
    ----------
    df : pd.DataFrame containing cleaned sensor columns.

    Returns
    -------
    pd.DataFrame: A copy of df with engineered features appended.
    """
    assert_no_forbidden_columns(df, context="feature_engineering input")
    X = df.copy()

    # 1. Temperature Difference: \Delta T = T_process - T_air
    if COL_PROCESS_TEMP in X.columns and COL_AIR_TEMP in X.columns:
        X[COL_TEMP_DIFF] = X[COL_PROCESS_TEMP] - X[COL_AIR_TEMP]
    else:
        raise KeyError(f"Missing temperature columns: {COL_PROCESS_TEMP}, {COL_AIR_TEMP}")

    # 2. Mechanical Power: P = \tau \times \omega (W), \omega = 2 \pi RPM / 60
    if COL_TORQUE in X.columns and COL_ROT_SPEED in X.columns:
        angular_speed_rad_s = X[COL_ROT_SPEED] * (2.0 * np.pi / 60.0)
        X[COL_MECH_POWER] = X[COL_TORQUE] * angular_speed_rad_s
    else:
        raise KeyError(f"Missing torque or speed columns: {COL_TORQUE}, {COL_ROT_SPEED}")

    # 3. Torque \times Tool Wear: Overstrain proxy
    if COL_TORQUE in X.columns and COL_TOOL_WEAR in X.columns:
        X[COL_TORQUE_WEAR] = X[COL_TORQUE] * X[COL_TOOL_WEAR]
    else:
        raise KeyError(f"Missing torque or wear columns: {COL_TORQUE}, {COL_TOOL_WEAR}")

    # Validate output integrity
    if X.isnull().any().any():
        null_cols = X.columns[X.isnull().any()].tolist()
        raise ValueError(f"Feature engineering generated NaNs in columns: {null_cols}")

    assert_no_forbidden_columns(X, context="feature_engineering output")
    return X


def compute_recommendation_thresholds(X_train: pd.DataFrame) -> dict[str, float]:
    """
    Compute operational threshold percentiles on the TRAINING partition only.
    Prevents leakage of test set distributions into recommendation rules.

    Parameters
    ----------
    X_train : pd.DataFrame of training features.

    Returns
    -------
    dict mapping threshold names to empirical training cutoff values.
    """
    thresholds = {
        "tool_wear_warning": float(X_train[COL_TOOL_WEAR].quantile(0.75)),
        "tool_wear_critical": float(X_train[COL_TOOL_WEAR].quantile(0.95)),
        "torque_warning": float(X_train[COL_TORQUE].quantile(0.90)),
        "process_temp_warning": float(X_train[COL_PROCESS_TEMP].quantile(0.90)),
        "rot_speed_low": float(X_train[COL_ROT_SPEED].quantile(0.05)),
        "rot_speed_high": float(X_train[COL_ROT_SPEED].quantile(0.95)),
    }
    if COL_TEMP_DIFF in X_train.columns:
        thresholds["temp_diff_low"] = float(X_train[COL_TEMP_DIFF].quantile(0.05))
    if COL_MECH_POWER in X_train.columns:
        thresholds["power_low"] = float(X_train[COL_MECH_POWER].quantile(0.05))
        thresholds["power_high"] = float(X_train[COL_MECH_POWER].quantile(0.95))

    logger.info("Computed training-only recommendation thresholds successfully.")
    return thresholds

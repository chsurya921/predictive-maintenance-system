"""
src/recommendation_engine.py
============================
Industrial Recommendation and Decision Support Engine for Predictive Maintenance.

System Architecture:
  Separates stochastic Machine Learning risk prediction from deterministic,
  rule-based mechanical action recommendations.

Decision Rules & Threshold Sources:
  All thresholds are derived from training-data percentiles (Rule 3) and documented
  domain boundaries from UCI AI4I physics definitions:
  - Tool Wear: 75th percentile warning (~140 min), 95th percentile critical (>200 min).
  - Torque: 90th percentile (~53 Nm) indicates mechanical binding or overload.
  - Temperature: Process temp > 311.5 K or \Delta T < 8.6 K with speed < 1380 rpm indicates cooling fault.
  - Rotational Speed: Extreme low (< 1380 rpm) or extreme high (> 2000 rpm) indicates drive fault.
  - Calibrated Probability Bands:
    * Normal:    P(y=1) < 0.25
    * Warning:   0.25 <= P(y=1) < 0.55
    * Critical:  P(y=1) >= 0.55

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
    COL_TYPE,
    COL_TEMP_DIFF,
    COL_MECH_POWER,
    COL_TORQUE_WEAR,
    PROB_WARNING_THRESHOLD,
    PROB_CRITICAL_THRESHOLD,
)
from utils.helpers import get_logger

logger = get_logger("recommendation_engine")


class MaintenanceRecommendationEngine:
    """
    Transparent condition-based maintenance decision rules engine.
    """
    def __init__(self, thresholds: dict[str, float] | None = None):
        # Default empirical cutoffs derived from Stage 1 & 2 training percentiles
        self.thresholds = thresholds or {
            "tool_wear_warning": 142.0,     # 75th percentile
            "tool_wear_critical": 202.0,    # 95th percentile (HDF/TWF boundary)
            "torque_warning": 52.8,         # 90th percentile
            "torque_critical": 62.0,        # 98th percentile
            "process_temp_warning": 311.9,  # 90th percentile
            "rot_speed_low": 1380.0,        # HDF physics boundary
            "rot_speed_high": 2000.0,       # 95th percentile
            "temp_diff_low": 8.6,           # HDF physics boundary
            "power_low": 3500.0,            # PWF lower boundary
            "power_high": 9000.0,           # PWF upper boundary
        }

    def assess_machine_condition(
        self,
        sensor_readings: dict[str, float | str],
        ml_failure_prob: float | None = None,
    ) -> dict[str, Any]:
        """
        Evaluate physical telemetry readings and return structured maintenance advice.

        Parameters
        ----------
        sensor_readings : Dict of sensor inputs (air_temp, proc_temp, rpm, torque, tool_wear, type).
        ml_failure_prob : Predicted probability from ensemble / ML pipeline (optional).

        Returns
        -------
        Dict with status band, severity, specific root-cause warnings, and actions.
        """
        air_temp = float(sensor_readings.get(COL_AIR_TEMP, 300.0))
        proc_temp = float(sensor_readings.get(COL_PROCESS_TEMP, 310.0))
        speed = float(sensor_readings.get(COL_ROT_SPEED, 1500.0))
        torque = float(sensor_readings.get(COL_TORQUE, 40.0))
        wear = float(sensor_readings.get(COL_TOOL_WEAR, 100.0))
        prod_type = str(sensor_readings.get(COL_TYPE, "L"))

        delta_t = proc_temp - air_temp
        power = torque * (speed * 2.0 * np.pi / 60.0)
        strain = torque * wear

        diagnostics = []
        action_items = []
        severity = "NORMAL"

        # 1. Tool Wear Evaluation
        if wear >= self.thresholds["tool_wear_critical"]:
            severity = "CRITICAL"
            diagnostics.append(f"Tool wear ({wear:.0f} min) exceeds critical threshold ({self.thresholds['tool_wear_critical']:.0f} min).")
            action_items.append("Immediate tool insert replacement required before next machining operation.")
        elif wear >= self.thresholds["tool_wear_warning"]:
            if severity != "CRITICAL":
                severity = "WARNING"
            diagnostics.append(f"Tool wear ({wear:.0f} min) is in warning zone ({self.thresholds['tool_wear_warning']:.0f} min).")
            action_items.append("Schedule tool replacement at nearest planned cycle break.")

        # 2. Thermal & Heat Dissipation Evaluation
        if delta_t < self.thresholds["temp_diff_low"] and speed < self.thresholds["rot_speed_low"]:
            severity = "CRITICAL"
            diagnostics.append(f"Heat Dissipation Fault: Temp difference \Delta T = {delta_t:.1f} K (< 8.6 K) at low RPM ({speed:.0f}).")
            action_items.append("Inspect coolant pump flow rate and clear heat exchanger ventilation passages.")
        elif proc_temp >= self.thresholds["process_temp_warning"]:
            if severity != "CRITICAL":
                severity = "WARNING"
            diagnostics.append(f"Process temperature ({proc_temp:.1f} K) exceeds 90th percentile.")
            action_items.append("Verify chiller circuit setpoints and check for lubrication degradation.")

        # 3. Mechanical Power & Drive Regime
        if power < self.thresholds["power_low"] or power > self.thresholds["power_high"]:
            severity = "CRITICAL"
            diagnostics.append(f"Power Failure Condition: Mechanical power P = {power:.0f} W (outside nominal 3500 - 9000 W).")
            action_items.append("Perform spindle drive electrical diagnostics and inspect variable frequency drive (VFD).")

        # 4. Overstrain Stress Proxy
        overstrain_limit = 11000.0 if "L" in prod_type else (12000.0 if "M" in prod_type else 13000.0)
        if strain > overstrain_limit:
            severity = "CRITICAL"
            diagnostics.append(f"Overstrain Hazard: Torque \times Tool Wear = {strain:.0f} exceeds variant limit ({overstrain_limit:.0f}).")
            action_items.append("Reduce feed rate and axial cut depth; verify material hardness tolerances.")
        elif torque >= self.thresholds["torque_warning"]:
            if severity != "CRITICAL":
                severity = "WARNING"
            diagnostics.append(f"Torque reading ({torque:.1f} Nm) is abnormally elevated.")
            action_items.append("Inspect mechanical transmission, bearing preloads, and spindle alignment.")

        # 5. Integrate ML Prediction (if available)
        if ml_failure_prob is not None:
            if ml_failure_prob >= PROB_CRITICAL_THRESHOLD:
                severity = "CRITICAL"
            elif ml_failure_prob >= PROB_WARNING_THRESHOLD and severity == "NORMAL":
                severity = "WARNING"

        if not action_items:
            action_items.append("Operating within safe nominal design envelope. Continue standard shift schedule.")

        return {
            "severity": severity,
            "probability_band": (
                "Critical Risk" if severity == "CRITICAL"
                else ("Elevated Warning" if severity == "WARNING" else "Nominal / Normal")
            ),
            "diagnostics": diagnostics,
            "recommendations": action_items,
            "calculated_physics": {
                "delta_t_k": round(delta_t, 2),
                "mechanical_power_w": round(power, 1),
                "torque_wear_proxy": round(strain, 1),
            }
        }

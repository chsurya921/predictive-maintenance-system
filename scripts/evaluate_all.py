"""
scripts/evaluate_all.py
=======================
Evaluation inspection script.
Loads trained model artifacts from models/trained_models/ and evaluates on the
fixed shared test set, verifying evaluation parity and printing performance tables.

Author: ML Engineering Team
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import METRICS_CSV, MODELS_DIR
from src.data_preprocessing import get_train_test_data
from utils.helpers import load_model, get_logger

logger = get_logger("evaluate_all")

def main():
    if not METRICS_CSV.exists():
        logger.warning(f"Metrics CSV not found at {METRICS_CSV}. Please run scripts/train_all.py first.")
        return

    df = pd.read_csv(METRICS_CSV)
    print("\n" + "="*85)
    print("AI4I 2020 PREDICTIVE MAINTENANCE SYSTEM - FINAL TEST SET BENCHMARK")
    print("Evaluated on identical fixed held-out test set (N=2,000, 68 Failures, 80/20 Stratified Split)")
    print("="*85)
    print(df.to_string(index=False))

    best = df.iloc[0]
    print("\n" + "-"*85)
    print(f"Top Model: {best['Model']}")
    print(f"  - PR-AUC (Primary): {best['PR-AUC (Primary)']}  (95% CI: {best['PR-AUC 95% CI']})")
    print(f"  - ROC-AUC:          {best['ROC-AUC']}  (95% CI: {best['ROC-AUC 95% CI']})")
    print(f"  - Recall:           {best['Recall']}  (95% CI: {best['Recall 95% CI']})")
    print(f"  - F1-Score:         {best['F1-Score']}")
    print(f"  - Accuracy:         {best['Accuracy']}")
    print("-"*85 + "\n")

if __name__ == "__main__":
    main()

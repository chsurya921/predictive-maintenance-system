"""
scripts/run_eda.py
==================
Complete Exploratory Data Analysis (EDA) execution script for AI4I 2020 dataset.
Generates all required statistical summaries, outlier audits, failure-overlap matrices,
and saves high-resolution figures to results/visualizations/.

Author: ML Engineering Team
"""

import os
import sys
from pathlib import Path

# Ensure writable matplotlib config directory in all execution environments
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from scipy import stats

# Path setup
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import RAW_DATA_PATH, VIZ_DIR, FAILURE_MODE_COLS, IDENTIFIER_COLS
from utils.helpers import get_logger, ensure_dir

logger = get_logger("run_eda")

def main():
    logger.info("Starting Stage 1: Exploratory Data Analysis...")
    ensure_dir(VIZ_DIR)
    
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(f"Raw data file missing at: {RAW_DATA_PATH}")

    df = pd.read_csv(RAW_DATA_PATH)
    logger.info(f"Loaded dataset successfully. Shape: {df.shape}")

    # 1. Basic Inspection
    n_rows, n_cols = df.shape
    n_missing = df.isnull().sum().sum()
    n_duplicates = df.duplicated().sum()
    dtypes = df.dtypes.to_dict()

    print("\n" + "="*70)
    print("1. DATASET OVERVIEW & INTEGRITY AUDIT")
    print("="*70)
    print(f"Total Rows:       {n_rows}")
    print(f"Total Columns:    {n_cols}")
    print(f"Missing Values:   {n_missing}")
    print(f"Duplicate Rows:   {n_duplicates}")
    print("\nColumns and Data Types:")
    for col, dt in dtypes.items():
        print(f"  - {col:<26}: {dt}")

    # 2. Class Imbalance
    target_counts = df["Machine failure"].value_counts()
    target_pct = df["Machine failure"].value_counts(normalize=True) * 100
    print("\n" + "="*70)
    print("2. TARGET CLASS IMBALANCE (Machine failure)")
    print("="*70)
    print(f"Class 0 (Non-failure): {target_counts[0]:>5} ({target_pct[0]:.2f}%)")
    print(f"Class 1 (Failure):     {target_counts[1]:>5} ({target_pct[1]:.2f}%)")
    print(f"Imbalance Ratio:       {target_counts[0] / target_counts[1]:.2f} : 1")

    # 3. Type Distribution
    type_counts = df["Type"].value_counts()
    type_pct = df["Type"].value_counts(normalize=True) * 100
    print("\n" + "="*70)
    print("3. PRODUCT TYPE DISTRIBUTION")
    print("="*70)
    for t, cnt in type_counts.items():
        print(f"Type {t}: {cnt:>5} ({type_pct[t]:.2f}%)")

    # 4. Failure Mode Overlap & Anomalous Rows
    flags = ["TWF", "HDF", "PWF", "OSF", "RNF"]
    flag_counts = {f: int(df[f].sum()) for f in flags}
    print("\n" + "="*70)
    print("4. FAILURE MODES & ANOMALOUS OVERLAP ANALYSIS")
    print("="*70)
    print("Individual Failure Flag Totals:")
    for f, c in flag_counts.items():
        print(f"  - {f}: {c}")

    # Rows where Machine failure == 1
    failed_df = df[df["Machine failure"] == 1]
    sum_flags_failed = failed_df[flags].sum(axis=1)
    failed_0_flags = int((sum_flags_failed == 0).sum())
    failed_1_flag = int((sum_flags_failed == 1).sum())
    failed_multi_flags = int((sum_flags_failed >= 2).sum())

    print(f"\nOf the 339 Failed Rows (Machine failure == 1):")
    print(f"  - Exactly 1 flag set:      {failed_1_flag:>3} ({failed_1_flag/339*100:.1f}%)")
    print(f"  - Multiple (>=2) flags:    {failed_multi_flags:>3} ({failed_multi_flags/339*100:.1f}%)")
    print(f"  - Zero failure flags set:  {failed_0_flags:>3} ({failed_0_flags/339*100:.1f}%) [ANOMALY A]")

    # Rows where Machine failure == 0 but flag == 1
    non_failed_df = df[df["Machine failure"] == 0]
    flagged_non_failed = non_failed_df[non_failed_df[flags].sum(axis=1) > 0]
    n_flagged_non_failed = len(flagged_non_failed)

    print(f"\nFlagged but Non-Failure Rows (Machine failure == 0 but flags > 0) [ANOMALY B]:")
    print(f"  - Total anomalous rows:    {n_flagged_non_failed}")
    for f in flags:
        print(f"    * {f}: {flagged_non_failed[f].sum()}")

    # 5. Outlier Detection: IQR & Z-score
    sensor_cols = [
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]"
    ]
    print("\n" + "="*70)
    print("5. SENSOR DISTRIBUTIONS & OUTLIER AUDIT (DO NOT BLINDLY REMOVE)")
    print("="*70)
    outlier_summary = []
    for col in sensor_cols:
        series = df[col]
        # IQR method
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        iqr_lower = q1 - 1.5 * iqr
        iqr_upper = q3 + 1.5 * iqr
        iqr_outliers = series[(series < iqr_lower) | (series > iqr_upper)]

        # Z-score method (|z| > 3)
        z_scores = np.abs(stats.zscore(series))
        z_outliers = series[z_scores > 3]

        outlier_summary.append({
            "Sensor": col,
            "Mean": series.mean(),
            "Std": series.std(),
            "Min": series.min(),
            "Q1": q1,
            "Median": series.median(),
            "Q3": q3,
            "Max": series.max(),
            "IQR": iqr,
            "IQR Outliers": len(iqr_outliers),
            "IQR Outlier %": len(iqr_outliers) / len(series) * 100,
            "Z-Score (|z|>3)": len(z_outliers),
            "Z-Score %": len(z_outliers) / len(series) * 100,
        })

    outlier_df = pd.DataFrame(outlier_summary)
    for row in outlier_summary:
        print(f"{row['Sensor']}:")
        print(f"  Range: [{row['Min']:.2f}, {row['Max']:.2f}], Mean: {row['Mean']:.2f} ± {row['Std']:.2f}")
        print(f"  IQR [1.5x] Outliers:  {row['IQR Outliers']:>4} ({row['IQR Outlier %']:.2f}%)")
        print(f"  Z-Score [|z|>3] Outliers: {row['Z-Score (|z|>3)']:>4} ({row['Z-Score %']:.2f}%)")

    # 6. Failure Rate by Type
    print("\n" + "="*70)
    print("6. FAILURE RATE BY PRODUCT TYPE")
    print("="*70)
    type_fail = df.groupby("Type")["Machine failure"].agg(["count", "sum", "mean"])
    type_fail["failure_rate_%"] = type_fail["mean"] * 100
    type_fail = type_fail.rename(columns={"count": "total_machines", "sum": "failures"})
    for idx, r in type_fail.iterrows():
        print(f"Type {idx}: {int(r['total_machines']):>5} total | {int(r['failures']):>3} failures | Rate: {r['failure_rate_%']:.2f}%")

    # 7. Failure Rate by Tool Wear Bins
    print("\n" + "="*70)
    print("7. FAILURE RATE BY TOOL WEAR BINS")
    print("="*70)
    wear_bins = [0, 50, 100, 150, 200, 250, 300]
    wear_labels = ["0-50m", "51-100m", "101-150m", "151-200m", "201-250m", "251+m"]
    df["wear_bin"] = pd.cut(df["Tool wear [min]"], bins=wear_bins, labels=wear_labels, include_lowest=True)
    wear_fail = df.groupby("wear_bin", observed=True)["Machine failure"].agg(["count", "sum", "mean"])
    wear_fail["failure_rate_%"] = wear_fail["mean"] * 100
    wear_fail = wear_fail.rename(columns={"count": "total_machines", "sum": "failures"})
    for idx, r in wear_fail.iterrows():
        print(f"Bin {idx:<10}: {int(r['total_machines']):>5} total | {int(r['failures']):>3} failures | Rate: {r['failure_rate_%']:.2f}%")

    # ==========================================
    # VISUALIZATION GENERATION
    # ==========================================
    logger.info("Generating publication-quality visualizations...")
    sns.set_theme(style="whitegrid", font="sans-serif")

    # Plot 1: Target Imbalance & Type Breakdown
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    sns.countplot(ax=axes[0], data=df, x="Machine failure", palette=["#2ecc71", "#e74c3c"])
    axes[0].set_title("Target Class Distribution (Machine failure)", fontsize=13, fontweight="bold")
    axes[0].set_xticklabels(["Normal (0)\n96.61%", "Failure (1)\n3.39%"])
    axes[0].set_ylabel("Count")
    for p in axes[0].patches:
        axes[0].annotate(f"{int(p.get_height())}", (p.get_x() + p.get_width() / 2., p.get_height() + 100),
                         ha='center', va='baseline', fontsize=11, fontweight="bold")

    sns.countplot(ax=axes[1], data=df, x="Type", palette="Blues_d", order=["L", "M", "H"])
    axes[1].set_title("Product Type Distribution (L / M / H)", fontsize=13, fontweight="bold")
    axes[1].set_ylabel("Count")
    for p in axes[1].patches:
        pct = p.get_height() / len(df) * 100
        axes[1].annotate(f"{int(p.get_height())}\n({pct:.1f}%)", (p.get_x() + p.get_width() / 2., p.get_height() / 2),
                         ha='center', va='center', fontsize=11, color="white", fontweight="bold")
    plt.tight_layout()
    p1_path = VIZ_DIR / "01_class_and_type_distribution.png"
    plt.savefig(p1_path, dpi=300)
    plt.close()
    logger.info(f"Saved: {p1_path.name}")

    # Plot 2: Failure Modes Breakdown & Overlap Matrix
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    flag_series = pd.Series(flag_counts)
    sns.barplot(ax=axes[0], x=flag_series.index, y=flag_series.values, palette="rocket")
    axes[0].set_title("Individual Failure Mode Counts", fontsize=13, fontweight="bold")
    axes[0].set_ylabel("Number of Occurrences")
    for p in axes[0].patches:
        axes[0].annotate(f"{int(p.get_height())}", (p.get_x() + p.get_width() / 2., p.get_height() + 2),
                         ha='center', va='baseline', fontsize=11, fontweight="bold")

    # Co-occurrence matrix among failure modes
    co_matrix = df[flags].T.dot(df[flags])
    sns.heatmap(ax=axes[1], data=co_matrix, annot=True, fmt="d", cmap="YlOrRd", cbar=True)
    axes[1].set_title("Failure Mode Co-Occurrence Matrix", fontsize=13, fontweight="bold")
    plt.tight_layout()
    p2_path = VIZ_DIR / "02_failure_modes_and_cooccurrence.png"
    plt.savefig(p2_path, dpi=300)
    plt.close()
    logger.info(f"Saved: {p2_path.name}")

    # Plot 3: Sensor Distributions per Product Type (Boxplots & KDE)
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.flatten()
    colors = {"L": "#3498db", "M": "#f39c12", "H": "#9b59b6"}
    for idx, col in enumerate(sensor_cols):
        sns.boxplot(ax=axes[idx], data=df, x="Type", y=col, order=["L", "M", "H"], palette=colors)
        axes[idx].set_title(f"{col} by Product Type", fontsize=11, fontweight="bold")
        axes[idx].set_xlabel("Product Type")
    # Hide the 6th unused subplot
    axes[5].axis("off")
    plt.tight_layout()
    p3_path = VIZ_DIR / "03_sensor_distributions_by_type.png"
    plt.savefig(p3_path, dpi=300)
    plt.close()
    logger.info(f"Saved: {p3_path.name}")

    # Plot 4: Sensor Distributions Conditioned on Machine Failure (0 vs 1)
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.flatten()
    fail_colors = {0: "#2ecc71", 1: "#e74c3c"}
    for idx, col in enumerate(sensor_cols):
        sns.kdeplot(ax=axes[idx], data=df, x=col, hue="Machine failure", palette=fail_colors,
                    common_norm=False, fill=True, alpha=0.3, linewidth=2)
        axes[idx].set_title(f"{col}: Normal vs Failure", fontsize=11, fontweight="bold")
        axes[idx].set_xlabel(col)
    axes[5].axis("off")
    plt.tight_layout()
    p4_path = VIZ_DIR / "04_sensor_distributions_by_failure.png"
    plt.savefig(p4_path, dpi=300)
    plt.close()
    logger.info(f"Saved: {p4_path.name}")

    # Plot 5: Correlation Heatmap
    corr_cols = sensor_cols + ["Machine failure"] + flags
    corr_matrix = df[corr_cols].corr()
    plt.figure(figsize=(11, 9))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1,
                linewidths=0.5, cbar_kws={"shrink": 0.8})
    plt.title("Correlation Heatmap: Sensors, Failure Target & Failure Modes", fontsize=13, fontweight="bold", pad=15)
    plt.tight_layout()
    p5_path = VIZ_DIR / "05_correlation_heatmap.png"
    plt.savefig(p5_path, dpi=300)
    plt.close()
    logger.info(f"Saved: {p5_path.name}")

    # Plot 6: Failure Rate by Type & by Tool Wear Bins
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.barplot(ax=axes[0], x=type_fail.index, y=type_fail["failure_rate_%"], palette="Blues_r")
    axes[0].set_title("Failure Rate by Product Type", fontsize=13, fontweight="bold")
    axes[0].set_ylabel("Failure Rate (%)")
    axes[0].set_xlabel("Product Type")
    for p in axes[0].patches:
        axes[0].annotate(f"{p.get_height():.2f}%", (p.get_x() + p.get_width() / 2., p.get_height() + 0.1),
                         ha='center', va='baseline', fontsize=11, fontweight="bold")

    sns.barplot(ax=axes[1], x=wear_fail.index, y=wear_fail["failure_rate_%"], palette="Reds")
    axes[1].set_title("Failure Rate by Tool Wear Exposure (Bins)", fontsize=13, fontweight="bold")
    axes[1].set_ylabel("Failure Rate (%)")
    axes[1].set_xlabel("Tool Wear Interval")
    plt.setp(axes[1].get_xticklabels(), rotation=30, ha="right")
    for p in axes[1].patches:
        axes[1].annotate(f"{p.get_height():.1f}%", (p.get_x() + p.get_width() / 2., p.get_height() + 0.2),
                         ha='center', va='baseline', fontsize=10, fontweight="bold")
    plt.tight_layout()
    p6_path = VIZ_DIR / "06_failure_rate_by_type_and_tool_wear.png"
    plt.savefig(p6_path, dpi=300)
    plt.close()
    logger.info(f"Saved: {p6_path.name}")

    # Plot 7: Bivariate Sensor Scatter Highlighting Physics-Based Failure Clusters
    # (Torque vs Rotational Speed, Air Temp vs Process Temp)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    sns.scatterplot(ax=axes[0], data=df, x="Rotational speed [rpm]", y="Torque [Nm]",
                    hue="Machine failure", palette=fail_colors, alpha=0.6, s=25)
    axes[0].set_title("Torque vs Rotational Speed (Mechanical Power Regime)", fontsize=12, fontweight="bold")

    sns.scatterplot(ax=axes[1], data=df, x="Air temperature [K]", y="Process temperature [K]",
                    hue="Machine failure", palette=fail_colors, alpha=0.6, s=25)
    axes[1].set_title("Process Temp vs Air Temp (Heat Dissipation Regime)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    p7_path = VIZ_DIR / "07_physics_sensor_regimes.png"
    plt.savefig(p7_path, dpi=300)
    plt.close()
    logger.info(f"Saved: {p7_path.name}")

    # Save summary tables to CSV
    outlier_df.to_csv(VIZ_DIR / "sensor_summary_and_outliers.csv", index=False)
    type_fail.to_csv(VIZ_DIR / "failure_rate_by_type.csv")
    wear_fail.to_csv(VIZ_DIR / "failure_rate_by_tool_wear.csv")
    logger.info("Stage 1 EDA run complete. All artifacts and tables generated successfully.")

if __name__ == "__main__":
    main()

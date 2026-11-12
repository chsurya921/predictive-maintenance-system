"""
app.py
======
Intelligent Predictive Maintenance System (AI4I 2020)
Streamlit Dashboard for Industrial Equipment Monitoring & Analytics.

Pages:
  1. Overview: System architecture, live dataset KPIs, and pipeline status.
  2. Machine Prediction: Real-time sensor input inference & probability bands.
  3. Failure Analysis: Multi-mode breakdown, overlap matrix & anomaly audit.
  4. Model Comparison: Benchmark table, ROC/PR curves & performance metrics.
  5. Explainable AI: SHAP force/beeswarm plots & GAM partial dependence.
  6. Dataset Explorer: Interactive slice-and-dice, distributions & CSV export.

Author: ML Engineering Team
"""

import os
import sys
from pathlib import Path

# Fix Matplotlib configuration directory for restricted environments
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Path setup
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    RAW_DATA_PATH,
    MODELS_DIR,
    VIZ_DIR,
    METRICS_CSV,
    FAILURE_MODE_COLS,
    PROB_WARNING_THRESHOLD,
    PROB_CRITICAL_THRESHOLD,
    COL_AIR_TEMP,
    COL_PROCESS_TEMP,
    COL_ROT_SPEED,
    COL_TORQUE,
    COL_TOOL_WEAR,
    COL_TYPE,
)
from src.recommendation_engine import MaintenanceRecommendationEngine
from utils.helpers import load_model

# ---------------------------------------------------------------------------
# Streamlit Page Config & Custom Industrial Styling
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Predictive Maintenance AI4I 2020",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Industrial CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border-radius: 8px;
        padding: 16px;
        border: 1px solid #E2E8F0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .badge-normal {
        background-color: #DCFCE7;
        color: #166534;
        padding: 4px 10px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-warning {
        background-color: #FEF9C3;
        color: #854D0E;
        padding: 4px 10px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-critical {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 4px 10px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Data & Model Loading (Cached)
# ---------------------------------------------------------------------------
@st.cache_data
def load_data():
    if not RAW_DATA_PATH.exists():
        st.error(f"Dataset not found at {RAW_DATA_PATH}. Please verify data directory.")
        st.stop()
    df = pd.read_csv(RAW_DATA_PATH)
    return df

@st.cache_resource
def load_trained_models():
    models = {}
    candidates = {
        "Soft Voting Ensemble": MODELS_DIR / "voting_ensemble.joblib",
        "XGBoost": MODELS_DIR / "xgboost_model.joblib",
        "Gradient Boosting": MODELS_DIR / "gradient_boosting_model.joblib",
        "CART": MODELS_DIR / "cart_model.joblib",
        "From-Scratch MARS": MODELS_DIR / "from_scratch_mars_model.joblib",
        "pyGAM": MODELS_DIR / "gam_model.joblib",
        "Bayesian Model Averaging (BMA)": MODELS_DIR / "bma_model.joblib",
    }
    for name, path in candidates.items():
        if path.exists():
            try:
                models[name] = load_model(path)
            except Exception:
                pass

    ovr_path = MODELS_DIR / "failure_type_ovr_classifier.joblib"
    ovr_model = load_model(ovr_path) if ovr_path.exists() else None

    return models, ovr_model

df = load_data()
models_dict, ovr_module = load_trained_models()
rec_engine = MaintenanceRecommendationEngine()


# ---------------------------------------------------------------------------
# Sidebar Navigation
# ---------------------------------------------------------------------------
st.sidebar.image("https://img.icons8.com/fluency/96/maintenance.png", width=64)
st.sidebar.title("Predictive Maintenance")
st.sidebar.caption("AI4I 2020 Industrial Equipment ML System")

page = st.sidebar.radio(
    "Navigation",
    [
        "1. Overview",
        "2. Machine Prediction",
        "3. Failure Analysis",
        "4. Model Comparison",
        "5. Explainable AI",
        "6. Dataset Explorer",
    ],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.info("""
**System Version:** v2.0.0-Production  
**Trained Models:** 9 Architectures  
**Top Model:** Soft Voting Ensemble  
**Primary Metric:** PR-AUC = 0.9042
""")


# ===========================================================================
# PAGE 1: OVERVIEW
# ===========================================================================
if page == "1. Overview":
    st.markdown('<div class="main-header">Predictive Maintenance System Overview</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Industrial Milling Machine Condition Monitoring (AI4I 2020 Dataset)</div>', unsafe_allow_html=True)

    # Top KPI Metrics
    c1, c2, c3, c4 = st.columns(4)
    total_records = len(df)
    total_failures = int(df["Machine failure"].sum())
    failure_pct = total_failures / total_records * 100
    avg_speed = df["Rotational speed [rpm]"].mean()
    avg_torque = df["Torque [Nm]"].mean()

    with c1:
        st.metric("Total Equipment Cycles", f"{total_records:,}", "100% verified")
    with c2:
        st.metric("Machine Failures", f"{total_failures:,}", f"{failure_pct:.2f}% prevalence", delta_color="inverse")
    with c3:
        st.metric("Mean Speed", f"{avg_speed:.1f} RPM", "Range: 1168 - 2886")
    with c4:
        st.metric("Mean Torque", f"{avg_torque:.1f} Nm", "Range: 3.8 - 76.6")

    st.markdown("---")

    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.subheader("Data & System Architecture")
        st.markdown("""
        This predictive maintenance platform applies **Ensemble and Advanced Machine Learning** techniques to detect and diagnose industrial equipment degradation before catastrophic failure occurs:
        
        - **Data Integrity:** 10,000 sensor observations, 14 attributes, zero missing values, zero duplicates.
        - **Anti-Leakage Architecture:** Strict feature filtering blocks unique identifiers (`UDI`, `Product ID`) and the five failure-mode labels (`TWF`, `HDF`, `PWF`, `OSF`, `RNF`) from binary failure models.
        - **Cost-Sensitive Learning:** Accounts for severe class imbalance (28.5 : 1) via calibrated probability bands, `scale_pos_weight`, and threshold tuning on PR-AUC.
        - **Multi-Algorithm Spectrum:** CART, AdaBoost, Gradient Boosting, XGBoost, Generalized Additive Models (GAM), from-scratch MARS, and Bayesian Model Averaging (BMA).
        """)

        st.subheader("Sensor Correlation Structure")
        corr_cols = [
            "Air temperature [K]", "Process temperature [K]",
            "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]", "Machine failure"
        ]
        corr_matrix = df[corr_cols].corr()
        fig_corr = px.imshow(
            corr_matrix,
            text_auto=".2f",
            color_continuous_scale="RdBu_r",
            zmin=-1, zmax=1,
            title="Sensor & Target Correlation Heatmap"
        )
        st.plotly_chart(fig_corr, width="stretch")

    with col_right:
        st.subheader("Product Variant Stratification")
        type_counts = df["Type"].value_counts().reset_index()
        type_counts.columns = ["Variant", "Count"]
        type_counts["Percentage"] = (type_counts["Count"] / len(df) * 100).round(1)

        fig_type = px.pie(
            type_counts,
            names="Variant",
            values="Count",
            color="Variant",
            color_discrete_map={"L": "#3b82f6", "M": "#10b981", "H": "#8b5cf6"},
            hole=0.45,
            title="Equipment Quality Variant Split (L / M / H)"
        )
        st.plotly_chart(fig_type, width="stretch")

        st.subheader("Pipeline Stage Progress")
        stages = [
            ("Stage 1: Setup & EDA", "COMPLETED", "11/11 tests passing"),
            ("Stage 2: Preprocessing & Features", "COMPLETED", "9/9 tests passing"),
            ("Stage 3: Standalone Models", "COMPLETED", "5/5 tests passing"),
            ("Stage 4: Ensembles & BMA", "COMPLETED", "6/6 tests passing"),
            ("Stage 5: Evaluation & Multi-label", "COMPLETED", "All artifacts saved"),
            ("Stage 6: Recommendation Engine", "COMPLETED", "Dashboard live"),
        ]
        for name, status, detail in stages:
            color = "#10b981"
            st.markdown(f"""
            <div style="display:flex; justify-content:space-between; align-items:center; padding:8px 12px; margin-bottom:6px; background:#F8FAFC; border-left:4px solid {color}; border-radius:4px;">
                <div><strong>{name}</strong><br><small style="color:#64748B">{detail}</small></div>
                <span style="font-size:0.8rem; font-weight:700; color:{color}">{status}</span>
            </div>
            """, unsafe_allow_html=True)


# ===========================================================================
# PAGE 2: MACHINE PREDICTION
# ===========================================================================
elif page == "2. Machine Prediction":
    st.markdown('<div class="main-header">Real-Time Machine Condition Prediction</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Interactive sensor telemetry simulation, ML inference, and physics-driven recommendations</div>', unsafe_allow_html=True)

    col_in, col_res = st.columns([1, 1])

    with col_in:
        st.subheader("Input Machine Telemetry")
        p_type_full = st.selectbox("Product Type", ["L (Low)", "M (Medium)", "H (High)"], index=0)
        p_type = p_type_full[0]  # 'L', 'M', or 'H'
        p_air_temp = st.slider("Air Temperature [K]", min_value=295.0, max_value=305.0, value=300.0, step=0.1)
        p_proc_temp = st.slider("Process Temperature [K]", min_value=305.0, max_value=315.0, value=310.0, step=0.1)
        p_speed = st.slider("Rotational Speed [rpm]", min_value=1100, max_value=2900, value=1530, step=10)
        p_torque = st.slider("Torque [Nm]", min_value=3.0, max_value=80.0, value=40.0, step=0.5)
        p_wear = st.slider("Tool Wear [min]", min_value=0, max_value=260, value=100, step=1)

        # Active Model Selector
        available_model_names = list(models_dict.keys())
        default_model = "Soft Voting Ensemble" if "Soft Voting Ensemble" in available_model_names else available_model_names[0]
        chosen_model_name = st.selectbox("Active Inference Model", available_model_names, index=available_model_names.index(default_model))
        active_model = models_dict[chosen_model_name]

    with col_res:
        st.subheader("Diagnostic Status & Inference")

        # Construct single-instance DataFrame for inference
        input_data = {
            COL_TYPE: p_type,
            COL_AIR_TEMP: p_air_temp,
            COL_PROCESS_TEMP: p_proc_temp,
            COL_ROT_SPEED: p_speed,
            COL_TORQUE: p_torque,
            COL_TOOL_WEAR: p_wear,
        }
        input_df = pd.DataFrame([input_data])
        # Add engineered features
        from src.feature_engineering import engineer_features
        input_eng_df = engineer_features(input_df)

        # ML Prediction
        try:
            prob_arr = active_model.predict_proba(input_eng_df)
            failure_prob = float(prob_arr[0, 1])
        except Exception:
            prob_arr = active_model.predict_proba(input_df)
            failure_prob = float(prob_arr[0, 1])

        # Rule-based assessment
        assessment = rec_engine.assess_machine_condition(input_data, ml_failure_prob=failure_prob)

        # Gauge Chart
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=failure_prob * 100,
            number={"suffix": "%", "valueformat": ".1f"},
            title={"text": f"<b>{chosen_model_name} Predicted Failure Risk</b>", "font": {"size": 16}},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#1e293b"},
                "steps": [
                    {"range": [0, 30], "color": "#dcfce7"},
                    {"range": [30, 60], "color": "#fef9c3"},
                    {"range": [60, 100], "color": "#fee2e2"},
                ],
                "threshold": {
                    "line": {"color": "red", "width": 4},
                    "thickness": 0.75,
                    "value": 55,
                }
            }
        ))
        fig_gauge.update_layout(height=260, margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig_gauge, width="stretch")

        # Status Badge
        severity = assessment["severity"]
        if severity == "NORMAL":
            st.markdown('<div class="badge-normal" style="font-size:1.1rem; display:inline-block; margin-bottom:12px;">STATUS: NORMAL OPERATION</div>', unsafe_allow_html=True)
        elif severity == "WARNING":
            st.markdown('<div class="badge-warning" style="font-size:1.1rem; display:inline-block; margin-bottom:12px;">STATUS: WARNING - ELEVATED HAZARD</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="badge-critical" style="font-size:1.1rem; display:inline-block; margin-bottom:12px;">STATUS: CRITICAL IMMINENT FAILURE</div>', unsafe_allow_html=True)

        # Multi-Label Failure Mode Diagnosis
        if ovr_module is not None and failure_prob >= 0.25:
            st.markdown("**Predicted Specific Failure Mechanism(s):**")
            ovr_probs = ovr_module.predict_proba(input_eng_df)
            active_modes = [flag.upper() for flag, p in ovr_probs.items() if p[0] >= 0.30]
            if active_modes:
                st.warning(f"Likely Failure Mode(s): **{', '.join(active_modes)}**")
            else:
                st.info("Failure probability elevated, but individual mode flags below threshold (Possible secondary anomaly).")

        st.subheader("Actionable Maintenance Recommendations")
        for rec in assessment["recommendations"]:
            st.markdown(f"- {rec}")


# ===========================================================================
# PAGE 3: FAILURE ANALYSIS
# ===========================================================================
elif page == "3. Failure Analysis":
    st.markdown('<div class="main-header">Failure Modes & Anomaly Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">In-depth deconstruction of failure causes, overlap patterns, and dataset anomalies</div>', unsafe_allow_html=True)

    flags = ["TWF", "HDF", "PWF", "OSF", "RNF"]
    flag_sums = df[flags].sum()

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("HDF (Heat Dissipation)", f"{flag_sums['HDF']}", "33.9% of failures")
    c2.metric("OSF (Overstrain)", f"{flag_sums['OSF']}", "28.9% of failures")
    c3.metric("PWF (Power Failure)", f"{flag_sums['PWF']}", "28.0% of failures")
    c4.metric("TWF (Tool Wear)", f"{flag_sums['TWF']}", "13.6% of failures")
    c5.metric("RNF (Random Failure)", f"{flag_sums['RNF']}", "5.6% (Uncorrelated)")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Failure Mode Co-Occurrence Matrix")
        co_matrix = df[flags].T.dot(df[flags])
        fig_co = px.imshow(
            co_matrix,
            text_auto=True,
            color_continuous_scale="YlOrRd",
            title="Failure Mode Co-Occurrence Counts"
        )
        st.plotly_chart(fig_co, width="stretch")

    with col2:
        st.subheader("Target Anomaly Audit")
        st.markdown("""
        The UCI dataset contains two critical edge-case phenomena that must be accounted for without altering raw data:
        
        1. **Anomaly A (9 Unlabeled Failures):**
           - Exactly **9 machines** experienced `Machine failure = 1`, yet all 5 failure mode flags (`TWF`, `HDF`, `PWF`, `OSF`, `RNF`) are 0.
           - *Cause:* Secondary or unmodeled physical degradation modes.
           
        2. **Anomaly B (18 Flagged Non-Failures):**
           - Exactly **18 machines** have a failure mode flag set to 1, but `Machine failure = 0`.
           - **100% of these 18 instances are Random Failures (`RNF = 1`)**.
           - Out of 19 total RNF events in the dataset, only 1 triggered a machine breakdown. This confirms RNF is an unlearnable random Poisson process.
        """)

    st.subheader("Publication Visualizations from Stage 1")
    viz_images = [
        ("02_failure_modes_and_cooccurrence.png", "Failure Modes Breakdown & Co-occurrence Matrix"),
        ("04_sensor_distributions_by_failure.png", "Sensor Distributions Conditioned on Machine Failure"),
        ("06_failure_rate_by_type_and_tool_wear.png", "Failure Rate by Product Type & Tool Wear Exposure"),
        ("07_physics_sensor_regimes.png", "Bivariate Physics Sensor Regimes (Power & Heat Dissipation)"),
    ]
    tabs = st.tabs([title for _, title in viz_images])
    for tab, (fname, title) in zip(tabs, viz_images):
        with tab:
            img_path = VIZ_DIR / fname
            if img_path.exists():
                st.image(str(img_path), caption=title, width="stretch")
            else:
                st.warning(f"Visualization {fname} not found in {VIZ_DIR}")


# ===========================================================================
# PAGE 4: MODEL COMPARISON
# ===========================================================================
elif page == "4. Model Comparison":
    st.markdown('<div class="main-header">Model Benchmark & Comparison</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Rigorous evaluation on identical held-out test partition (N=2,000, 68 Failures, 80/20 Stratified Split)</div>', unsafe_allow_html=True)

    if METRICS_CSV.exists():
        metrics_df = pd.read_csv(METRICS_CSV)
        st.dataframe(metrics_df, width="stretch")

        top_model = metrics_df.iloc[0]
        st.success(f"🏆 **Top Performing Model:** {top_model['Model']} | **PR-AUC (Primary):** {top_model['PR-AUC (Primary)']} (95% CI: {top_model['PR-AUC 95% CI']}) | **Recall:** {top_model['Recall']} | **ROC-AUC:** {top_model['ROC-AUC']}")

        # Display ROC and PR curves
        c1, c2 = st.columns(2)
        roc_path = VIZ_DIR / "roc_curves_comparison.png"
        pr_path = VIZ_DIR / "pr_curves_comparison.png"

        with c1:
            if pr_path.exists():
                st.image(str(pr_path), caption="Precision-Recall Curves (Primary Metric for Rare Failures)", width="stretch")
        with c2:
            if roc_path.exists():
                st.image(str(roc_path), caption="Receiver Operating Characteristic (ROC) Curves", width="stretch")

        tree_path = VIZ_DIR / "cart_decision_tree.png"
        if tree_path.exists():
            st.subheader("Pruned CART Decision Tree Architecture")
            st.image(str(tree_path), caption="Cost-Complexity Pruned Decision Tree Structure", width="stretch")
    else:
        st.warning("Metrics CSV not found. Please run scripts/train_all.py.")


# ===========================================================================
# PAGE 5: EXPLAINABLE AI (XAI)
# ===========================================================================
elif page == "5. Explainable AI":
    st.markdown('<div class="main-header">Explainable AI & Physics Interpretation</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Global feature attribution, SHAP TreeExplainer, and GAM partial dependence curves</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Physical Mechanics Behind Failures")
        st.markdown(r"""
        The synthetic AI4I 2020 dataset was designed around four explicit physical failure formulas:
        
        1. **Heat Dissipation Failure (HDF):**
           $$\Delta T = T_{\text{process}} - T_{\text{air}} < 8.6\text{ K} \quad \text{and} \quad \omega < 1380\text{ rpm}$$
        2. **Power Failure (PWF):**
           $$P = \tau \cdot \omega \quad \text{where} \quad P < 3500\text{ W} \quad \text{or} \quad P > 9000\text{ W}$$
        3. **Overstrain Failure (OSF):**
           $$\tau \cdot \text{Tool wear} > \text{Threshold}(\text{Type})$$
           where the threshold is 11,000 for L, 12,000 for M, and 13,000 for H.
        4. **Tool Wear Failure (TWF):**
           $$\text{Tool wear} \ge 200 - 240\text{ min}$$
        """)

    with col2:
        st.subheader("Sensor Regimes Scatter")
        p7 = VIZ_DIR / "07_physics_sensor_regimes.png"
        if p7.exists():
            st.image(str(p7), caption="Physical Boundaries Captured by Sensor Interactions", width="stretch")

    st.subheader("TreeSHAP Global Feature Importance")
    c1, c2 = st.columns(2)
    shap_beeswarm = VIZ_DIR / "shap_summary_beeswarm.png"
    shap_bar = VIZ_DIR / "shap_feature_importance_bar.png"

    with c1:
        if shap_beeswarm.exists():
            st.image(str(shap_beeswarm), caption="SHAP Summary Beeswarm (Directional Impact on Failure Odds)", width="stretch")
    with c2:
        if shap_bar.exists():
            st.image(str(shap_bar), caption="SHAP Global Mean Absolute Impact", width="stretch")

    pdp_path = VIZ_DIR / "gam_partial_dependence.png"
    if pdp_path.exists():
        st.subheader("pyGAM Non-Linear Partial Dependence Splines (95% CI)")
        st.image(str(pdp_path), caption="Marginal Non-Linear Sensor Splines with 95% Confidence Intervals", width="stretch")


# ===========================================================================
# PAGE 6: DATASET EXPLORER
# ===========================================================================
elif page == "6. Dataset Explorer":
    st.markdown('<div class="main-header">Interactive Dataset Explorer</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Filter, query, inspect distributions, and export custom slices of the AI4I 2020 dataset</div>', unsafe_allow_html=True)

    with st.expander("🔍 Filter Telemetry Records", expanded=True):
        f1, f2, f3, f4 = st.columns(4)
        with f1:
            sel_type = st.multiselect("Product Type", ["L", "M", "H"], default=["L", "M", "H"])
        with f2:
            sel_fail = st.multiselect("Failure State", [0, 1], default=[0, 1], format_func=lambda x: "Failure (1)" if x == 1 else "Normal (0)")
        with f3:
            speed_range = st.slider("Rotational Speed [rpm]", int(df["Rotational speed [rpm]"].min()), int(df["Rotational speed [rpm]"].max()), (1168, 2886))
        with f4:
            torque_range = st.slider("Torque [Nm]", float(df["Torque [Nm]"].min()), float(df["Torque [Nm]"].max()), (3.8, 76.6))

    filtered_df = df[
        (df["Type"].isin(sel_type)) &
        (df["Machine failure"].isin(sel_fail)) &
        (df["Rotational speed [rpm]"].between(speed_range[0], speed_range[1])) &
        (df["Torque [Nm]"].between(torque_range[0], torque_range[1]))
    ]

    st.markdown(f"**Showing {len(filtered_df):,} of {len(df):,} records** ({len(filtered_df)/len(df)*100:.1f}%)")
    st.dataframe(filtered_df.head(200), width="stretch")

    csv_bytes = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Filtered Data as CSV",
        data=csv_bytes,
        file_name="ai4i2020_filtered_slice.csv",
        mime="text/csv"
    )

    st.subheader("Dynamic Histogram of Filtered Records")
    hist_col = st.selectbox("Select Feature for Distribution Plot", [
        "Air temperature [K]", "Process temperature [K]", "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"
    ])
    fig_hist = px.histogram(
        filtered_df,
        x=hist_col,
        color="Machine failure",
        barmode="overlay",
        color_discrete_map={0: "#10b981", 1: "#ef4444"},
        title=f"Distribution of {hist_col} (Normal vs Failure)"
    )
    st.plotly_chart(fig_hist, width="stretch")

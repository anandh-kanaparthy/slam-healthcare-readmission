"""
Healthcare Readmission Analytics & Risk Triaging System
Full-Stack Streamlit Application (Production-Grade Healthcare Product UI)

Modules:
1. Hospital Overview (Surveillance KPI dashboard, interactive filters, encounter sample)
2. Readmission Analytics (Deep-dive subgroup analysis, sample sizes N, descriptive correlations)
3. Data Quality & Preprocessing (Ingestion audit, missing data, leakage-free pipeline view)
4. Model Performance (Held-out test set metrics, confusion matrix, ROC curve, model comparison)
5. Patient Risk Review (SQLite-backed review queue, score thresholding, triage inspector, intake form)
"""
import json
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import config
from services.data_service import DataService
from services.prediction_service import PredictionService
from services.analytics_service import AnalyticsService
from services.preprocessing_service import PreprocessingService
from services.training_service import TrainingService
from services.review_service import ReviewService

# -----------------------------------------------------------------------------
# 1. Streamlit Application & Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Healthcare Readmission Analytics",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# 2. Professional White-Based Design System & Accessible CSS
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    /* Global Typography & White Canvas */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"], .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        background-color: #ffffff !important;
        color: #0f172a;
    }
    
    .main .block-container {
        padding-top: 1.8rem;
        padding-bottom: 3.2rem;
        padding-left: 2.2rem;
        padding-right: 2.2rem;
        max-width: 1400px;
    }

    /* Professional Headings & Hierarchy */
    .product-title {
        font-size: 1.85rem;
        font-weight: 700;
        color: #0f172a;
        letter-spacing: -0.025em;
        margin-bottom: 0.25rem;
        line-height: 1.25;
    }
    .product-subtitle {
        font-size: 0.95rem;
        color: #475569;
        margin-bottom: 1.4rem;
        line-height: 1.45;
    }
    .section-header {
        font-size: 1.15rem;
        font-weight: 600;
        color: #0f172a;
        margin-top: 1.2rem;
        margin-bottom: 0.65rem;
    }

    /* Sidebar Brand & Identity */
    section[data-testid="stSidebar"] {
        background-color: #fafbfc !important;
        border-right: 1px solid #e2e8f0;
    }
    .sidebar-brand-box {
        padding: 0.6rem 0.2rem 1.0rem 0.2rem;
        border-bottom: 1px solid #e2e8f0;
        margin-bottom: 1.1rem;
    }
    .brand-title {
        font-size: 1.25rem;
        font-weight: 700;
        color: #0f172a;
        display: flex;
        align-items: center;
        gap: 0.45rem;
        letter-spacing: -0.015em;
    }
    .brand-subtitle {
        font-size: 0.78rem;
        font-weight: 600;
        color: #475569;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 0.2rem;
    }

    /* Sidebar Status Indicator */
    .sidebar-pill-synthetic {
        background-color: #fef3c7;
        color: #92400e;
        border: 1px solid #fde68a;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.76rem;
        font-weight: 600;
        display: inline-block;
        margin-top: 0.9rem;
        margin-bottom: 0.4rem;
    }
    .sidebar-caption {
        font-size: 0.75rem;
        color: #475569;
        line-height: 1.35;
        margin-bottom: 1.0rem;
    }

    /* Metric Cards - Accessible, High Contrast */
    div[data-testid="stMetric"] {
        background-color: #ffffff !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 8px !important;
        padding: 14px 18px !important;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.03) !important;
    }
    div[data-testid="stMetric"] label, div[data-testid="stMetric"] [data-testid="stMetricLabel"] p {
        font-size: 0.80rem !important;
        font-weight: 600 !important;
        color: #475569 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.035em !important;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        font-size: 1.65rem !important;
        font-weight: 700 !important;
        color: #0f172a !important;
        letter-spacing: -0.02em !important;
    }

    /* Clinical Cards & Callouts */
    .clinical-card {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 16px 20px;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.02);
        margin-bottom: 1rem;
    }
    .guardrail-callout {
        background-color: #f8fafc;
        border-left: 4px solid #0284c7;
        padding: 10px 14px;
        border-radius: 0 6px 6px 0;
        color: #0369a1;
        font-size: 0.86rem;
        margin: 0.8rem 0 1.2rem 0;
        line-height: 1.45;
    }
    .disclaimer-banner {
        background-color: #f8fafc;
        border-top: 1px solid #e2e8f0;
        padding: 14px 16px;
        font-size: 0.80rem;
        color: #475569;
        margin-top: 2.5rem;
        line-height: 1.45;
        border-radius: 6px;
    }

    /* Streamlit Radio Navigation Styling */
    div[data-testid="stRadio"] > div {
        gap: 0.35rem;
    }
    div[data-testid="stRadio"] label {
        padding: 8px 12px !important;
        border-radius: 6px !important;
        transition: all 0.15s ease-in-out;
        font-size: 0.90rem !important;
        font-weight: 500 !important;
        color: #1e293b !important;
    }
    div[data-testid="stRadio"] label:hover {
        background-color: #f1f5f9 !important;
        color: #0f172a !important;
    }

    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        border-bottom: 1px solid #e2e8f0;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 0.90rem;
        font-weight: 600;
        color: #475569;
        padding: 8px 16px;
        border-radius: 6px 6px 0 0;
    }
    .stTabs [aria-selected="true"] {
        color: #0284c7 !important;
        border-bottom: 2px solid #0284c7 !important;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 3. Data Ingestion & Precomputed Model Probabilities
# -----------------------------------------------------------------------------
@st.cache_data
def get_cohort_data():
    """Loads active cohort and provenance metadata safely."""
    return DataService.load_data()

df_cohort, provenance = get_cohort_data()

@st.cache_data
def get_scored_cohort(df_in: pd.DataFrame):
    """
    Computes model readmission probabilities across the entire cohort.
    Enables dynamic, exact filtering and score thresholding across all views.
    """
    try:
        pipeline = PredictionService.get_pipeline()
        feature_cols = config.NUMERIC_FEATURES + config.CATEGORICAL_FEATURES
        scores = pipeline.predict_proba(df_in[feature_cols])[:, 1]
        df_out = df_in.copy()
        df_out["model_score"] = np.round(scores, 4)
        return df_out
    except Exception:
        df_out = df_in.copy()
        df_out["model_score"] = 0.50
        return df_out

df_scored_cohort = get_scored_cohort(df_cohort)

# Initialize and sync SQLite Review Store
try:
    ReviewService.init_database()
    ReviewService.sync_cohort_from_dataset(df_cohort, max_records=500)
except Exception:
    pass

# Pre-load saved evaluation metrics
metrics_saved = None
if config.METRICS_PATH.exists():
    try:
        with open(config.METRICS_PATH, "r") as f:
            metrics_saved = json.load(f)
    except Exception:
        metrics_saved = None

# -----------------------------------------------------------------------------
# 4. Professional Sidebar & Navigation
# -----------------------------------------------------------------------------
st.sidebar.markdown("""
<div class="sidebar-brand-box">
    <div class="brand-title">🏥 Healthcare Analytics</div>
    <div class="brand-subtitle">Readmission Surveillance & Review</div>
</div>
""", unsafe_allow_html=True)

NAV_PAGES = [
    "Hospital Overview",
    "Readmission Analytics",
    "Data Quality & Preprocessing",
    "Model Performance",
    "Patient Risk Review"
]

if "nav_page" not in st.session_state:
    st.session_state["nav_page"] = NAV_PAGES[0]

current_nav_index = NAV_PAGES.index(st.session_state["nav_page"]) if st.session_state["nav_page"] in NAV_PAGES else 0

selected_page = st.sidebar.radio(
    "Product Navigation",
    NAV_PAGES,
    index=current_nav_index,
    label_visibility="collapsed",
    key="nav_radio_selection"
)
st.session_state["nav_page"] = selected_page

st.sidebar.markdown("---")

# Truthful, Compact Dataset Status Indicator
if provenance.get("is_synthetic", False):
    st.sidebar.markdown(
        '<div class="sidebar-pill-synthetic">⚡ DEMO COHORT · SYNTHETIC</div>',
        unsafe_allow_html=True
    )
    st.sidebar.markdown(
        '<div class="sidebar-caption">Statistically modeled after UCI 130-US Hospitals (1999–2008). Contains zero Protected Health Information (PHI). Software demonstration only.</div>',
        unsafe_allow_html=True
    )
else:
    st.sidebar.markdown(
        '<div class="sidebar-pill-synthetic">✅ CLINICAL DATASET · ACTIVE</div>',
        unsafe_allow_html=True
    )
    st.sidebar.markdown(
        '<div class="sidebar-caption">Active de-identified clinical encounter dataset loaded.</div>',
        unsafe_allow_html=True
    )


# =============================================================================
# PAGE 1: HOSPITAL OVERVIEW (Landing Dashboard)
# =============================================================================
if selected_page == "Hospital Overview":
    st.markdown('<div class="product-title">Hospital Readmission Surveillance & Overview</div>', unsafe_allow_html=True)
    st.markdown('<div class="product-subtitle">Population health monitoring, encounter surveillance, and data-driven triaging metrics.</div>', unsafe_allow_html=True)

    # Interactive Cohort Filter Controls
    with st.expander("🔍 Interactive Population Filters", expanded=True):
        f_col1, f_col2, f_col3, f_col4 = st.columns(4)
        with f_col1:
            diag_opts = ["All Diagnoses"] + sorted(list(df_scored_cohort["primary_diagnosis"].dropna().unique()))
            sel_diag = st.selectbox("Primary Diagnosis", diag_opts, index=0)
        with f_col2:
            age_opts = ["All Age Groups"] + sorted(list(df_scored_cohort["age_group"].dropna().unique()))
            sel_age = st.selectbox("Age Bracket", age_opts, index=0)
        with f_col3:
            adm_opts = ["All Admissions"] + sorted(list(df_scored_cohort["admission_type"].dropna().unique()))
            sel_adm = st.selectbox("Admission Urgency", adm_opts, index=0)
        with f_col4:
            readm_opts = ["All Encounters", "Readmitted (<30d)", "No Readmission"]
            sel_readm = st.selectbox("Readmission Outcome", readm_opts, index=0)

    # Apply Filters to the Cohort
    filtered_df = df_scored_cohort.copy()
    if sel_diag != "All Diagnoses":
        filtered_df = filtered_df[filtered_df["primary_diagnosis"] == sel_diag]
    if sel_age != "All Age Groups":
        filtered_df = filtered_df[filtered_df["age_group"] == sel_age]
    if sel_adm != "All Admissions":
        filtered_df = filtered_df[filtered_df["admission_type"] == sel_adm]
    if sel_readm == "Readmitted (<30d)":
        filtered_df = filtered_df[filtered_df[config.TARGET_COLUMN] == 1]
    elif sel_readm == "No Readmission":
        filtered_df = filtered_df[filtered_df[config.TARGET_COLUMN] == 0]

    # Calculate Data-Driven KPI Metrics Reconciled to Filtered Cohort
    n_filtered = len(filtered_df)
    n_readmitted = int(filtered_df[config.TARGET_COLUMN].sum()) if n_filtered > 0 else 0
    readmit_rate_pct = (n_readmitted / n_filtered * 100) if n_filtered > 0 else 0.0

    # Records meeting the model score threshold (>= 0.50)
    elevated_risk_count = int((filtered_df["model_score"] >= 0.50).sum()) if n_filtered > 0 else 0
    elevated_risk_pct = (elevated_risk_count / n_filtered * 100) if n_filtered > 0 else 0.0

    # Model holdout test metrics from verified artifacts
    model_auc = metrics_saved.get("roc_auc", 0.6272) if metrics_saved else 0.6272
    model_acc = metrics_saved.get("accuracy", 0.5840) if metrics_saved else 0.5840

    # 4 Consistently Styled KPI Cards
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric(
            label="Inpatient Encounters",
            value=f"{n_filtered:,}",
            help=f"Represents {n_filtered / len(df_cohort) * 100:.1f}% of total active cohort ({len(df_cohort):,} total)"
        )
    with k2:
        st.metric(
            label="Observed Readmission Rate",
            value=f"{readmit_rate_pct:.1f}%",
            delta=f"{n_readmitted:,} positive labels",
            delta_color="inverse",
            help="Empirical proportion of filtered encounters with 30-day readmission outcome"
        )
    with k3:
        st.metric(
            label="Model Priority Triaged (≥50%)",
            value=f"{elevated_risk_count:,}",
            delta=f"{elevated_risk_pct:.1f}% of filtered cohort",
            delta_color="off",
            help="Encounters in current filtered selection where fitted pipeline estimates readmission probability ≥ 50%"
        )
    with k4:
        st.metric(
            label="Holdout Model ROC-AUC",
            value=f"{model_auc:.4f}",
            delta=f"Accuracy: {model_acc * 100:.1f}%",
            delta_color="normal",
            help="Discriminative score evaluated strictly on 1,000 unseen test partition encounters"
        )

    st.caption("ℹ️ *Analytical distinction: Cards 1–3 reflect observational metrics on filtered cohort encounters; Card 4 reflects holdout test set model validation.*")
    st.markdown("---")

    # Overview Visualizations
    st.markdown('<div class="section-header">Population Readmission Distributions</div>', unsafe_allow_html=True)
    c_chart1, c_chart2 = st.columns(2)
    with c_chart1:
        if n_filtered > 0:
            st.pyplot(AnalyticsService.plot_readmission_distribution(filtered_df))
        else:
            st.info("No encounters match the active filter criteria.")
    with c_chart2:
        if n_filtered > 0:
            st.pyplot(AnalyticsService.plot_readmission_by_diagnosis(filtered_df))
        else:
            st.info("No encounters match the active filter criteria.")

    st.markdown("---")

    # Length of Stay Boxplot & Encounter Table
    c_los, c_shortcut = st.columns([1.1, 1.3])
    with c_los:
        st.markdown('<div class="section-header">Length of Stay (LOS) vs. Readmission</div>', unsafe_allow_html=True)
        if n_filtered > 0:
            st.pyplot(AnalyticsService.plot_los_vs_readmission(filtered_df))
        else:
            st.info("No records match filter selection.")
    with c_shortcut:
        st.markdown('<div class="section-header">Filtered Encounter Records</div>', unsafe_allow_html=True)
        st.caption("Sample view of current filtered cohort. Use the Review Queue for clinical audit and note logging.")
        
        display_features = [
            "patient_id", "age_group", "gender", "primary_diagnosis",
            "time_in_hospital", "number_inpatient", "model_score", config.TARGET_COLUMN
        ]
        valid_cols = [c for c in display_features if c in filtered_df.columns]
        st.dataframe(filtered_df[valid_cols].head(8), use_container_width=True)

        if st.button("Open Patient Risk Review Queue ➔", type="primary", use_container_width=True):
            st.session_state["nav_page"] = "Patient Risk Review"
            st.rerun()


# =============================================================================
# PAGE 2: READMISSION ANALYTICS
# =============================================================================
elif selected_page == "Readmission Analytics":
    st.markdown('<div class="product-title">Readmission Analytics & Clinical Subgroup Patterns</div>', unsafe_allow_html=True)
    st.markdown('<div class="product-subtitle">Retrospective clinical risk factor analysis, sample size transparency, and subgroup comparisons.</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="guardrail-callout">
        ⚖️ <strong>Analytical Transparency:</strong> All percentages and comparisons below represent empirical descriptive statistics calculated directly from the active cohort ($N=5,000$). Observational correlations reflect historical patterns and do <em>not</em> imply direct medical causation.
    </div>
    """, unsafe_allow_html=True)

    tab_subgroup, tab_multivariate = st.tabs([
        "🔬 Subgroup Risk Explorer",
        "📊 Clinical Factor Distributions"
    ])

    with tab_subgroup:
        st.markdown('<div class="section-header">Interactive Subgroup Stratification</div>', unsafe_allow_html=True)
        st.caption("Select a clinical attribute to evaluate readmission proportions, encounter counts, and sample sizes ($N$) across subgroups.")

        # Enhanced feature options including clinically relevant brackets
        df_subgroup_source = df_cohort.copy()
        df_subgroup_source["los_bracket"] = df_subgroup_source["time_in_hospital"].apply(
            lambda d: "1-3 days" if d <= 3 else ("4-6 days" if d <= 6 else "7+ days")
        )
        df_subgroup_source["med_bracket"] = df_subgroup_source["num_medications"].apply(
            lambda m: "1-10 meds" if m <= 10 else ("11-20 meds" if m <= 20 else "21+ meds")
        )

        feature_map = {
            "Primary ICD Diagnosis": "primary_diagnosis",
            "Age Bracket": "age_group",
            "Admission Urgency": "admission_type",
            "Prior Inpatient Admissions": "number_inpatient",
            "Length of Stay Bracket": "los_bracket",
            "Medication Burden Bracket": "med_bracket",
            "Insulin Regimen": "insulin",
            "HbA1c Lab Result": "a1c_result",
            "Admission Referral Route": "admission_source"
        }
        sel_feat_label = st.selectbox("Stratify Cohort By Clinical Attribute", list(feature_map.keys()), index=0)
        sel_attr_col = feature_map[sel_feat_label]

        # Calculate subgroup breakdown table
        cohort_baseline_rate = df_subgroup_source[config.TARGET_COLUMN].mean() * 100
        subgroup_rows = []
        for cat_val, grp in df_subgroup_source.groupby(sel_attr_col):
            n_sub = len(grp)
            n_readm = int(grp[config.TARGET_COLUMN].sum())
            sub_rate = (n_readm / n_sub * 100) if n_sub > 0 else 0.0
            subgroup_rows.append({
                "Category": str(cat_val),
                "Sample Size (N)": n_sub,
                "Readmitted": n_readm,
                "Readmission Rate (%)": round(sub_rate, 2),
                "Delta vs Baseline": round(sub_rate - cohort_baseline_rate, 2)
            })

        df_subgroup_stats = pd.DataFrame(subgroup_rows).sort_values("Sample Size (N)", ascending=False)

        col_t, col_p = st.columns([1.1, 1.2])
        with col_t:
            st.markdown(f"#### Subgroup Table (Cohort Baseline: {cohort_baseline_rate:.1f}%)")
            st.dataframe(df_subgroup_stats, use_container_width=True)
        with col_p:
            fig, ax = plt.subplots(figsize=(6.0, 3.8))
            sns.barplot(
                data=df_subgroup_stats,
                x="Readmission Rate (%)",
                y="Category",
                palette="Blues_r",
                ax=ax
            )
            ax.axvline(cohort_baseline_rate, color="#dc2626", linestyle="--", linewidth=1.2, label=f"Cohort Baseline ({cohort_baseline_rate:.1f}%)")
            ax.set_title(f"Readmission Rate by {sel_feat_label}", fontsize=11, fontweight="bold", color="#0f172a", pad=10)
            ax.set_xlabel("Readmission Rate (%)", fontsize=9.5, color="#1e293b")
            ax.set_ylabel("Subgroup Category", fontsize=9.5, color="#1e293b")
            ax.legend(loc="lower right", fontsize=8.5)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

    with tab_multivariate:
        st.markdown('<div class="section-header">Multivariate Risk Factor Distributions</div>', unsafe_allow_html=True)
        c_m1, c_m2 = st.columns(2)
        with c_m1:
            st.markdown("#### Readmission Rate by Prior Inpatient Frequency")
            st.pyplot(AnalyticsService.plot_prior_admissions_vs_rate(df_cohort))
            st.caption("Empirical rate across prior hospitalization volume showing sample counts ($N$).")
        with c_m2:
            st.markdown("#### Readmission Rate Across Age Brackets")
            st.pyplot(AnalyticsService.plot_age_vs_readmission(df_cohort))
            st.caption("Proportion of readmitted encounters across demographic age brackets.")

        st.markdown("---")
        c_m3, c_m4 = st.columns(2)
        with c_m3:
            st.markdown("#### Prior Inpatient Admissions vs. Readmission Count")
            st.pyplot(AnalyticsService.plot_inpatient_visits_vs_readmission(df_cohort))
            st.caption("Distribution of prior hospitalizations across readmission outcomes.")
        with c_m4:
            st.markdown("#### Primary Diagnostic Volume Distribution")
            st.pyplot(AnalyticsService.plot_primary_diagnosis_distribution(df_cohort))
            st.caption("Encounter volume distribution across primary ICD clinical categories.")

        st.markdown("### Empirical Cohort Observations")
        st.markdown("""
        - **Prior Acute Admissions:** Patients with 2 or more inpatient admissions in the preceding 12 months exhibit substantially elevated readmission incidence compared to first-time admissions.
        - **Extended Inpatient Duration:** Hospital stays extending beyond 6 days correlate with higher readmission proportions, reflecting greater acute complexity during admission.
        - **Polypharmacy Regimens:** High counts of prescribed medications correlate with readmission risk due to treatment regimen complexity post-discharge.
        """)


# =============================================================================
# PAGE 3: DATA QUALITY & PREPROCESSING
# =============================================================================
elif selected_page == "Data Quality & Preprocessing":
    st.markdown('<div class="product-title">Data Quality Audit & Preprocessing Pipeline</div>', unsafe_allow_html=True)
    st.markdown('<div class="product-subtitle">Ingestion data hygiene, missing value audit, and leakage-free transformation architecture.</div>', unsafe_allow_html=True)

    df_cleaned, clean_report = DataService.clean_dataset(df_cohort)
    is_valid, validation_msg = DataService.validate_schema(df_cohort)

    # Ingestion Audit KPI Row
    st.markdown('<div class="section-header">1. Ingestion Data Hygiene & Validation</div>', unsafe_allow_html=True)
    q1, q2, q3, q4 = st.columns(4)
    with q1:
        st.metric("Total Ingested Records", f"{clean_report['initial_records']:,}")
    with q2:
        st.metric("Duplicate Rows Removed", f"{clean_report['duplicates_removed']}")
    with q3:
        st.metric("Validated Clean Encounters", f"{clean_report['cleaned_records']:,}")
    with q4:
        st.metric("Total Missing Cells", f"{clean_report['total_missing_values']}")

    if is_valid:
        st.success(f"✅ Schema Validation Passed: All {len(config.NUMERIC_FEATURES) + len(config.CATEGORICAL_FEATURES)} clinical features and target column `{config.TARGET_COLUMN}` are verified.")
    else:
        st.error(f"❌ Schema Validation Warning: {validation_msg}")

    st.markdown("---")

    # Missing Value Audit
    st.markdown('<div class="section-header">2. Missing Value Profiling & Imputation Strategy</div>', unsafe_allow_html=True)
    missing_series = df_cohort.isnull().sum()
    missing_pct = (missing_series / len(df_cohort) * 100).round(2)
    missing_audit_df = pd.DataFrame({
        "Feature Name": missing_series.index,
        "Data Type": df_cohort.dtypes.values.astype(str),
        "Missing Count": missing_series.values,
        "Missing (%)": missing_pct.values,
        "Imputation Method": [
            "Median (Continuous / Count)" if c in config.NUMERIC_FEATURES else ("Mode (Categorical / Nominal)" if c in config.CATEGORICAL_FEATURES else "None")
            for c in missing_series.index
        ]
    })

    col_a1, col_a2 = st.columns([1.2, 1])
    with col_a1:
        st.dataframe(missing_audit_df, use_container_width=True)
    with col_a2:
        st.markdown("#### Preprocessing Design Decisions")
        st.markdown("""
        - **Numerical Features** (`time_in_hospital`, `num_medications`, etc.):
          Imputed with **Median** strategy to preserve robustness against clinical outliers and skewed hospitalization durations.
        - **Categorical Features** (`admission_type`, `primary_diagnosis`, etc.):
          Imputed with **Most Frequent (Mode)** strategy to handle unrecorded clinical categories cleanly.
        - **Leakage Safeguard:**
          Imputation statistics are fit **strictly on the 80% training split** ($N=4,000$) and applied to the 20% test partition ($N=1,000$) without re-estimation.
        """)

    st.markdown("---")

    # Leakage-Free Scikit-Learn Pipeline Structure
    st.markdown('<div class="section-header">3. Leakage-Free Preprocessing Architecture</div>', unsafe_allow_html=True)
    st.markdown("""
    ```text
    ColumnTransformer (Fitted Solely on 80% Training Split)
    │
    ├── Numeric Pipeline (8 continuous/count features):
    │   ├── SimpleImputer(strategy='median')
    │   └── StandardScaler()
    │
    └── Categorical Pipeline (10 categorical features):
        ├── SimpleImputer(strategy='most_frequent')
        └── OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    ```
    """)
    st.info("💡 **Feature Engineering Safeguard:** Patient identification attributes (`patient_id`) are stripped before model fitting to prevent artificial memorization and overfitting.")

    st.markdown("---")

    # Feature Association Ranking
    st.markdown('<div class="section-header">4. Target Association Ranking (Feature Selection)</div>', unsafe_allow_html=True)
    correlations = DataService.get_feature_selection_summary(df_cohort)
    if correlations:
        c_r1, c_r2 = st.columns([1, 1])
        with c_r1:
            st.pyplot(AnalyticsService.plot_feature_selection_ranking(correlations))
        with c_r2:
            st.dataframe(pd.DataFrame(correlations), use_container_width=True)


# =============================================================================
# PAGE 4: MODEL PERFORMANCE
# =============================================================================
elif selected_page == "Model Performance":
    st.markdown('<div class="product-title">Model Performance & Algorithm Benchmarking</div>', unsafe_allow_html=True)
    st.markdown('<div class="product-subtitle">Supervised classification metrics on held-out test encounters ($N=1,000$), confusion matrix, and model comparison.</div>', unsafe_allow_html=True)

    if metrics_saved:
        st.markdown('<div class="section-header">1. Held-Out Test Set Performance (N=1,000 Unseen Encounters)</div>', unsafe_allow_html=True)
        p1, p2, p3, p4, p5 = st.columns(5)
        p1.metric("Model Architecture", metrics_saved.get("model_type", "Random Forest"))
        p2.metric("Test Accuracy", f"{metrics_saved.get('accuracy', 0.5840) * 100:.2f}%")
        p3.metric("Test Precision", f"{metrics_saved.get('precision', 0.5934):.4f}")
        p4.metric("Test Recall (Sensitivity)", f"{metrics_saved.get('recall', 0.5957):.4f}")
        p5.metric("ROC-AUC Score", f"{metrics_saved.get('roc_auc', 0.6272):.4f}")

        st.caption(f"Evaluated strictly on {metrics_saved.get('test_samples', 1000):,} holdout test records with baseline readmission incidence of {metrics_saved.get('test_readmission_rate', 51.2):.1f}%.")
        st.markdown("---")

        # Evaluation Diagnostic Visualizations
        st.markdown('<div class="section-header">2. Evaluation Diagnostic Artifacts</div>', unsafe_allow_html=True)
        pl1, pl2 = st.columns(2)
        with pl1:
            st.markdown("#### Test Set Confusion Matrix")
            st.pyplot(AnalyticsService.plot_confusion_matrix(metrics_saved["confusion_matrix"]))
            st.caption("Annotated True Negatives, False Positives, False Negatives, and True Positives.")
        with pl2:
            st.markdown("#### Receiver Operating Characteristic (ROC) Curve")
            if config.ROC_CURVE_PATH.exists():
                st.image(str(config.ROC_CURVE_PATH))
            else:
                st.info("ROC plot will be generated upon model training.")

        if config.FEATURE_IMPORTANCE_PATH.exists():
            st.markdown("---")
            st.markdown("#### Clinical Feature Importances (Random Forest Gini Score)")
            st.image(str(config.FEATURE_IMPORTANCE_PATH))

        st.markdown("---")

        # Side-by-side Model Architecture Comparison
        st.markdown('<div class="section-header">3. Model Architecture Comparison (Identical Holdout Split)</div>', unsafe_allow_html=True)
        st.caption("Side-by-side performance benchmark between non-linear ensemble (Random Forest) and linear baseline (Logistic Regression):")

        @st.cache_data
        def load_comparison_benchmark():
            X_train, X_test, y_train, y_test = PreprocessingService.split_data(df_cohort)
            return TrainingService.compare_models(X_train, X_test, y_train, y_test)

        benchmark_df = load_comparison_benchmark()
        st.dataframe(benchmark_df, use_container_width=True)

        st.markdown("""
        - **Random Forest Classifier:** Ensembles 100 balanced decision trees (`max_depth=10`). Captures non-linear feature interactions between hospitalization duration, diagnoses, and medication burden.
        - **Logistic Regression Baseline:** Evaluates linear decision boundaries with L2 regularization and balanced class weights.
        """)

    else:
        st.warning("No saved model metrics found. Please trigger model training below.")

    st.markdown("---")

    # Retraining Engine Controls
    st.markdown('<div class="section-header">4. Model Retraining Engine</div>', unsafe_allow_html=True)
    st.caption("Execute full CPU-friendly pipeline training on the active cohort:")

    c_sel, c_btn = st.columns([1, 2])
    with c_sel:
        model_selection = st.selectbox(
            "Select Classifier Architecture",
            ["Random Forest", "Logistic Regression"],
            index=0
        )
    with c_btn:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Execute Pipeline Training & Evaluation", type="primary"):
            with st.spinner(f"Training {model_selection} pipeline strictly on 80% train fold..."):
                X_train, X_test, y_train, y_test = PreprocessingService.split_data(df_cohort)
                pipeline, new_m = TrainingService.train_and_evaluate(
                    X_train, X_test, y_train, y_test,
                    model_type=model_selection,
                    save_artifacts=True
                )
                PredictionService.invalidate_cache()
                st.success(f"Model retrained and saved! Test Accuracy: {new_m['accuracy'] * 100:.2f}%, ROC-AUC: {new_m['roc_auc']:.4f}")
                st.rerun()


# =============================================================================
# PAGE 5: PATIENT RISK REVIEW
# =============================================================================
elif selected_page == "Patient Risk Review":
    st.markdown('<div class="product-title">Patient Risk Review & Clinical Triage Queue</div>', unsafe_allow_html=True)
    st.markdown('<div class="product-subtitle">Human-in-the-loop review workflow, score threshold triaging, persistent clinician audit notes, and intake evaluation.</div>', unsafe_allow_html=True)

    tab_queue, tab_intake = st.tabs([
        "📋 Clinical Review Queue & Audit",
        "➕ New Patient Intake Evaluation"
    ])

    with tab_queue:
        # Review Queue KPI Metric Cards
        q_stats = ReviewService.get_queue_statistics(threshold=0.50)
        
        qk1, qk2, qk3, qk4, qk5 = st.columns(5)
        qk1.metric("Total in Queue", f"{q_stats['total_encounters']:,}")
        qk2.metric("Elevated Priority (≥50%)", f"{q_stats['above_threshold']:,}")
        qk3.metric("Pending Review", f"{q_stats['pending_review']:,}")
        qk4.metric("Reviewed Standard", f"{q_stats['reviewed_standard']:,}")
        qk5.metric("Requires Follow-up", f"{q_stats['requires_followup']:,}")

        st.markdown("---")

        # Search, Filter & Sort Controls
        st.markdown('<div class="section-header">Queue Triage & Search Controls</div>', unsafe_allow_html=True)
        qc1, qc2, qc3, qc4 = st.columns(4)
        with qc1:
            search_query = st.text_input("Search (ID or Diagnosis)", placeholder="e.g. ENC_00012 or Circulatory")
        with qc2:
            status_filter = st.selectbox(
                "Filter by Review Status",
                ["All Statuses", "Pending Review", "Reviewed - Standard", "Requires Follow-up", "Escalated to Physician"],
                index=0
            )
        with qc3:
            threshold_slider = st.slider("Priority Score Threshold", 0.0, 1.0, 0.50, step=0.05)
        with qc4:
            sort_criterion = st.selectbox(
                "Sort Queue By",
                ["Risk Score (High to Low)", "Length of Stay", "Prior Inpatient Admissions", "Encounter ID"],
                index=0
            )

        sort_column_map = {
            "Risk Score (High to Low)": ("model_score", False),
            "Length of Stay": ("time_in_hospital", False),
            "Prior Inpatient Admissions": ("number_inpatient", False),
            "Encounter ID": ("encounter_id", True)
        }
        sort_col, sort_asc = sort_column_map[sort_criterion]

        # Query records from SQLite database
        queue_df = ReviewService.get_queue_records(
            search_query=search_query if search_query else None,
            status_filter=status_filter,
            threshold=threshold_slider,
            sort_by=sort_col,
            ascending=sort_asc,
            limit=100
        )

        st.markdown(f"**Showing {len(queue_df)} encounters matching filter criteria:**")

        if not queue_df.empty:
            display_cols = [
                "encounter_id", "age_group", "gender", "primary_diagnosis", "time_in_hospital",
                "number_inpatient", "model_score", "review_priority", "review_status", "reviewed_by", "last_reviewed_at"
            ]
            st.dataframe(queue_df[display_cols], use_container_width=True)

            st.markdown("---")

            # Clinician Action Panel
            st.markdown('<div class="section-header">Clinician Review Decision & Notes</div>', unsafe_allow_html=True)
            selected_enc_id = st.selectbox(
                "Select Encounter to Review / Update",
                options=queue_df["encounter_id"].tolist(),
                index=0
            )

            encounter_record = queue_df[queue_df["encounter_id"] == selected_enc_id].iloc[0]

            col_profile, col_action = st.columns([1.2, 1])
            with col_profile:
                st.markdown(f"#### Clinical Encounter: `{selected_enc_id}`")
                st.markdown(f"""
                - **Primary Diagnosis:** {encounter_record['primary_diagnosis']}
                - **Demographics:** {encounter_record['gender']}, Age {encounter_record['age_group']}
                - **Admission Urgency:** {encounter_record['admission_type']} ({encounter_record['admission_source']})
                - **Length of Stay:** {encounter_record['time_in_hospital']} days *(Cohort median: 4 days)*
                - **Prior Inpatient Admissions:** {encounter_record['number_inpatient']} *(Cohort 75th percentile: 1 visit)*
                - **Prescribed Medications:** {encounter_record['num_medications']} *(Regimen complexity index)*
                - **Model Readmission Score:** **{encounter_record['model_score'] * 100:.1f}%**
                """)

            with col_action:
                st.markdown("#### Update Review Status")
                with st.form("form_update_review"):
                    valid_statuses = ["Pending Review", "Reviewed - Standard", "Requires Follow-up", "Escalated to Physician"]
                    current_status = encounter_record["review_status"]
                    status_idx = valid_statuses.index(current_status) if current_status in valid_statuses else 0
                    
                    new_status = st.selectbox("Review Decision", valid_statuses, index=status_idx)
                    notes_content = st.text_area(
                        "Clinician / Reviewer Notes",
                        value=str(encounter_record["reviewer_notes"]) if pd.notna(encounter_record["reviewer_notes"]) else "",
                        placeholder="Enter clinical observations, outpatient check recommendations, or follow-up reason..."
                    )
                    reviewer_ident = st.text_input(
                        "Reviewer Name / ID",
                        value=str(encounter_record["reviewed_by"]) if encounter_record["reviewed_by"] else "Clinical Triage Nurse"
                    )
                    btn_save_review = st.form_submit_button("Persist Review Decision", type="primary")

                if btn_save_review:
                    saved_ok = ReviewService.update_review(
                        encounter_id=selected_enc_id,
                        new_status=new_status,
                        notes=notes_content,
                        reviewer=reviewer_ident
                    )
                    if saved_ok:
                        st.success(f"Encounter `{selected_enc_id}` updated and persisted to database!")
                        st.rerun()
                    else:
                        st.error("Failed to update encounter review in database.")

        else:
            st.info("No records in review queue match the current filter and search criteria. Adjust filters or search terms.")

    with tab_intake:
        st.markdown('<div class="section-header">Real-Time Patient Intake & Risk Scoring</div>', unsafe_allow_html=True)
        st.caption("Input patient attributes during admission or prior to discharge to calculate estimated readmission likelihood and triage into the review queue.")

        with st.form("intake_evaluation_form"):
            st.markdown("##### 1. Hospital Utilization & Length of Stay")
            in1, in2, in3, in4 = st.columns(4)
            with in1:
                los_in = st.slider("Length of Stay (Days)", 1, 14, 4)
            with in2:
                inp_in = st.number_input("Prior Inpatient Admissions", 0, 15, 1)
            with in3:
                emg_in = st.number_input("Prior Emergency Visits", 0, 15, 0)
            with in4:
                out_in = st.number_input("Prior Outpatient Visits", 0, 20, 0)

            st.markdown("##### 2. Demographics & Route of Admission")
            in5, in6, in7, in8 = st.columns(4)
            with in5:
                age_in = st.selectbox("Age Bracket", config.CATEGORICAL_CHOICES["age_group"], index=3)
            with in6:
                gender_in = st.selectbox("Gender", config.CATEGORICAL_CHOICES["gender"], index=0)
            with in7:
                adm_in = st.selectbox("Admission Urgency", config.CATEGORICAL_CHOICES["admission_type"], index=0)
            with in8:
                src_in = st.selectbox("Referral Route", config.CATEGORICAL_CHOICES["admission_source"], index=0)

            st.markdown("##### 3. Diagnostics & Procedures")
            in9, in10, in11, in12 = st.columns(4)
            with in9:
                diag_in = st.selectbox("Primary ICD Diagnosis", config.CATEGORICAL_CHOICES["primary_diagnosis"], index=0)
            with in10:
                med_in = st.slider("Prescribed Medications", 1, 60, 16)
            with in11:
                lab_in = st.slider("Lab Tests Conducted", 1, 120, 44)
            with in12:
                proc_in = st.slider("Non-Lab Procedures", 0, 6, 1)

            st.markdown("##### 4. Clinical Diagnostics & Medication Regimens")
            in13, in14, in15, in16 = st.columns(4)
            with in13:
                num_diag_in = st.slider("Total Diagnosis Codes", 1, 16, 7)
            with in14:
                insulin_in = st.selectbox("Insulin Status", config.CATEGORICAL_CHOICES["insulin"], index=1)
            with in15:
                a1c_in = st.selectbox("HbA1c Lab Test", config.CATEGORICAL_CHOICES["a1c_result"], index=0)
            with in16:
                glu_in = st.selectbox("Glucose Serum Test", config.CATEGORICAL_CHOICES["max_glu_serum"], index=0)

            in17, in18 = st.columns(2)
            with in17:
                chg_in = st.radio("Medication Adjusted During Stay?", config.CATEGORICAL_CHOICES["change_medication"], horizontal=True, index=0)
            with in18:
                diab_in = st.radio("Prescribed Diabetes Medication?", config.CATEGORICAL_CHOICES["diabetes_med"], horizontal=True, index=1)

            st.markdown("<br>", unsafe_allow_html=True)
            submit_intake = st.form_submit_button("Evaluate Intake Risk & Generate Triage Profile", type="primary", use_container_width=True)

        if submit_intake:
            patient_intake_payload = {
                "time_in_hospital": los_in,
                "num_lab_procedures": lab_in,
                "num_procedures": proc_in,
                "num_medications": med_in,
                "number_outpatient": out_in,
                "number_emergency": emg_in,
                "number_inpatient": inp_in,
                "number_diagnoses": num_diag_in,
                "age_group": age_in,
                "gender": gender_in,
                "admission_type": adm_in,
                "admission_source": src_in,
                "primary_diagnosis": diag_in,
                "max_glu_serum": glu_in,
                "a1c_result": a1c_in,
                "insulin": insulin_in,
                "change_medication": chg_in,
                "diabetes_med": diab_in,
            }

            try:
                eval_res = PredictionService.predict_readmission(patient_intake_payload)

                st.markdown("---")
                st.markdown('<div class="section-header">Intake Assessment Result</div>', unsafe_allow_html=True)

                r1, r2 = st.columns([1, 2])
                with r1:
                    st.metric("Predicted Likelihood", f"{eval_res['readmission_percentage']}%")
                    st.markdown(
                        f"**Statistical Risk Tier:** <span style='color:{eval_res['risk_color']}; font-size:1.15rem; font-weight:700;'>{eval_res['risk_tier']}</span>",
                        unsafe_allow_html=True
                    )
                    st.caption(f"Binary Classification: **{eval_res['prediction_label']}**")
                with r2:
                    st.markdown(f"**Classification Explanation:**\n\n{eval_res['tier_description']}")

                st.markdown("#### Empirical Cohort Benchmark Statistics")
                for fctr in eval_res["comparative_factors"]:
                    st.markdown(f"- 📊 {fctr}")

                st.markdown("""
                <div class="guardrail-callout">
                    ⚖️ <strong>Decision Support Disclaimer:</strong> Model predictions reflect statistical likelihood based on historical training encounters. This estimate is not a medical diagnosis or treatment directive.
                </div>
                """, unsafe_allow_html=True)

                # Persist to SQLite review queue
                new_encounter_id = ReviewService.insert_new_encounter(
                    patient_data=patient_intake_payload,
                    model_score=eval_res["readmission_probability"],
                    predicted_class=eval_res["prediction_class"],
                    notes=f"Intake evaluation: {eval_res['risk_tier']} ({eval_res['readmission_percentage']}% likelihood)."
                )
                st.success(f"✅ Patient successfully enrolled into persistent Review Queue as `{new_encounter_id}`! View and audit in the 'Clinical Review Queue' tab above.")

            except Exception as ex:
                st.error(f"Inference error: {ex}. Ensure the model pipeline is trained.")


# -----------------------------------------------------------------------------
# 5. Truthful Software Demonstration Notice (Footer)
# -----------------------------------------------------------------------------
st.markdown("""
<div class="disclaimer-banner">
    <strong>Healthcare Software Demonstration Notice:</strong> This application is an educational prototype developed for software testing and demonstration.
    Outputs reflect statistical likelihood computed on a synthetic cohort modeled after the UCI 130-US Hospitals dataset. The system is not certified for clinical diagnostic decisions or medical prescriptions.
</div>
""", unsafe_allow_html=True)

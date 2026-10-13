"""
Smart Healthcare Analytics and Patient Readmission Prediction System
Full-Stack Streamlit Application (IARE SLAM Project Review-2)

Students:
    - A. Akshith (Roll No: 2125951A05)
    - K. Anandh  (Roll No: 2125951A0527)
Guide: Ms. Y Sujana, Assistant Professor, CSE
Department of Computer Science and Engineering
Institute of Aeronautical Engineering (IARE)
SDGs Mapped: SDG 3 (Good Health & Well-being), SDG 9 (Industry, Innovation & Infrastructure)
"""
import json
from pathlib import Path
import streamlit as st
import pandas as pd
import matplotlib
matplotlib.use("Agg")

import config
from services.data_service import DataService
from services.prediction_service import PredictionService
from services.analytics_service import AnalyticsService
from services.preprocessing_service import PreprocessingService
from services.training_service import TrainingService

# Page Setup
st.set_page_config(
    page_title="Smart Healthcare Readmission Predictor | IARE",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional CSS Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.15rem;
        font-weight: 700;
        color: #1e3a8a;
        margin-bottom: 0.2rem;
        line-height: 1.2;
    }
    .subtitle {
        font-size: 1.05rem;
        color: #475569;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #f8fafc;
        border-radius: 8px;
        padding: 16px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .disclaimer-box {
        background-color: #fef3c7;
        border-left: 5px solid #d97706;
        padding: 14px 18px;
        border-radius: 4px;
        font-size: 0.90rem;
        color: #92400e;
        margin-top: 2rem;
    }
    .badge-real {
        background-color: #dcfce7;
        color: #166534;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-synthetic {
        background-color: #fef3c7;
        color: #92400e;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar Branding & Navigation
st.sidebar.markdown("### 🏥 SLAM Healthcare")
st.sidebar.markdown("**Patient Readmission System**")
st.sidebar.caption("IARE • Computer Science & Engineering")

menu_choice = st.sidebar.radio(
    "Navigation Menu",
    [
        "1. Executive Overview",
        "2. Patient Risk Predictor",
        "3. Exploratory Analytics",
        "4. Model Performance",
        "5. Methodology & SDGs"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Review-2 Project Details")
st.sidebar.markdown(f"**Guide:** {config.PROJECT_META['guide']}")
st.sidebar.markdown("**Students:**")
for s in config.PROJECT_META["students"]:
    st.sidebar.markdown(f"- {s['name']} (`{s['roll_no']}`)")
st.sidebar.markdown(f"**Mapped SDGs:**\n- {config.PROJECT_META['sdgs'][0]}\n- {config.PROJECT_META['sdgs'][1]}")

# Cached Data Ingestion
@st.cache_data
def get_cached_dataset():
    return DataService.load_data()

# -------------------------------------------------------------
# PAGE 1: EXECUTIVE OVERVIEW
# -------------------------------------------------------------
if menu_choice == "1. Executive Overview":
    st.markdown('<div class="main-title">Smart Healthcare Analytics and Patient Readmission Prediction System</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">A machine-learning-based system to analyze historical patient records and predict 30-day hospital readmission risk.</div>', unsafe_allow_html=True)

    df, provenance = get_cached_dataset()
    summary = DataService.get_dataset_summary(df)

    # Provenance Badge
    badge_html = f'<span class="badge-synthetic">⚠️ {provenance["label"]}</span>' if provenance["is_synthetic"] else f'<span class="badge-real">✅ {provenance["label"]}</span>'
    st.markdown(f"**Active Dataset Provenance:** {badge_html}", unsafe_allow_html=True)
    st.caption(f"Loaded from `{provenance['source']}`")

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Cohort Records", f"{summary['total_records']:,}")
    with col2:
        st.metric("Readmission Rate (<30d)", f"{summary['readmitted_pct']:.1f}%")
    with col3:
        st.metric("Clinical Features", f"{summary['total_features']}")
    with col4:
        st.metric("Missing Values", f"{summary['missing_values_count']}")

    st.markdown("---")
    st.subheader("System Architecture & Workflow Pipeline")
    st.info("🔄 **End-to-End Pipeline:** Historical Dataset ➔ Preprocessing (Imputation & Encoding) ➔ Feature Engineering ➔ ML Pipeline ➔ Real-time Prediction ➔ Interactive Dashboard")

    st.markdown("### Clinical Motivation & Problem Background")
    c_left, c_right = st.columns(2)
    with c_left:
        st.markdown("""
        - **Hospital Burden**: Unplanned 30-day readmissions strain hospital bed capacity, elevate healthcare costs, and often indicate post-discharge care gaps.
        - **Manual Triaging Limitations**: Sifting through complex electronic records manually to evaluate discharge risk is labor-intensive and inconsistent.
        """)
    with c_right:
        st.markdown("""
        - **Proactive Interventions**: Identifying patients at elevated risk enables targeted post-discharge calls, home healthcare, and timely medication reconciliation.
        - **Actionable Insights**: Translates multivariate clinical predictors into clear, interpretable risk tiers (Low, Moderate, High).
        """)

# -------------------------------------------------------------
# PAGE 2: PATIENT RISK PREDICTOR
# -------------------------------------------------------------
elif menu_choice == "2. Patient Risk Predictor":
    st.markdown('<div class="main-title">Patient Readmission Risk Assessment</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Enter patient clinical metrics to evaluate 30-day readmission risk probability using the fitted machine learning pipeline.</div>', unsafe_allow_html=True)

    with st.form("patient_assessment_form"):
        st.subheader("1. Encounter Details & Prior Hospital Utilization")
        u1, u2, u3, u4 = st.columns(4)
        with u1:
            time_in_hospital = st.slider("Length of Stay (Days)", 1, 14, 4, help="Number of days spent in the acute care unit.")
        with u2:
            number_inpatient = st.number_input("Prior Inpatient Admissions (Past Year)", 0, 15, 1, help="Number of hospitalizations during the preceding 12 months.")
        with u3:
            number_emergency = st.number_input("Prior Emergency Encounters (Past Year)", 0, 15, 0, help="Emergency department visits during the preceding 12 months.")
        with u4:
            number_outpatient = st.number_input("Prior Outpatient Visits (Past Year)", 0, 20, 0, help="Outpatient clinic consultations.")

        st.subheader("2. Demographics & Route of Admission")
        d1, d2, d3, d4 = st.columns(4)
        with d1:
            age_group = st.selectbox("Patient Age Bracket", config.CATEGORICAL_CHOICES["age_group"], index=3)
        with d2:
            gender = st.selectbox("Biological Sex", config.CATEGORICAL_CHOICES["gender"], index=0)
        with d3:
            admission_type = st.selectbox("Admission Urgency", config.CATEGORICAL_CHOICES["admission_type"], index=0)
        with d4:
            admission_source = st.selectbox("Admission Referral Source", config.CATEGORICAL_CHOICES["admission_source"], index=0)

        st.subheader("3. Clinical Diagnostics & Medication Regimens")
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            primary_diagnosis = st.selectbox("Primary ICD Diagnosis Category", config.CATEGORICAL_CHOICES["primary_diagnosis"], index=0)
        with m2:
            num_medications = st.slider("Total Prescribed Medications", 1, 60, 16, help="Count of distinct medications active on discharge.")
        with m3:
            num_lab_procedures = st.slider("Lab Tests Administered", 1, 120, 44, help="Number of lab procedures performed.")
        with m4:
            num_procedures = st.slider("Non-Lab Medical Procedures", 0, 6, 1, help="Count of bedside or surgical procedures.")

        m5, m6, m7, m8 = st.columns(4)
        with m5:
            number_diagnoses = st.slider("Total Diagnoses Entered", 1, 16, 8, help="Comorbidity index count.")
        with m6:
            insulin = st.selectbox("Insulin Status", config.CATEGORICAL_CHOICES["insulin"], index=1)
        with m7:
            a1c_result = st.selectbox("HbA1c Lab Test", config.CATEGORICAL_CHOICES["a1c_result"], index=0)
        with m8:
            max_glu_serum = st.selectbox("Glucose Serum Test", config.CATEGORICAL_CHOICES["max_glu_serum"], index=0)

        c_opt1, c_opt2 = st.columns(2)
        with c_opt1:
            change_medication = st.radio("Medication Dosage Adjusted During Stay?", config.CATEGORICAL_CHOICES["change_medication"], horizontal=True, index=0)
        with c_opt2:
            diabetes_med = st.radio("Prescribed Diabetes Medication?", config.CATEGORICAL_CHOICES["diabetes_med"], horizontal=True, index=1)

        st.markdown("<br>", unsafe_allow_html=True)
        submit_btn = st.form_submit_button("Assess Patient Readmission Risk", use_container_width=True, type="primary")

    if submit_btn:
        patient_data = {
            "time_in_hospital": time_in_hospital,
            "num_lab_procedures": num_lab_procedures,
            "num_procedures": num_procedures,
            "num_medications": num_medications,
            "number_outpatient": number_outpatient,
            "number_emergency": number_emergency,
            "number_inpatient": number_inpatient,
            "number_diagnoses": number_diagnoses,
            "age_group": age_group,
            "gender": gender,
            "admission_type": admission_type,
            "admission_source": admission_source,
            "primary_diagnosis": primary_diagnosis,
            "max_glu_serum": max_glu_serum,
            "a1c_result": a1c_result,
            "insulin": insulin,
            "change_medication": change_medication,
            "diabetes_med": diabetes_med,
        }

        try:
            prediction_result = PredictionService.predict_readmission(patient_data)

            st.markdown("---")
            st.subheader("Predictive Assessment Results")

            res_col1, res_col2 = st.columns([1, 2])
            with res_col1:
                st.metric("30-Day Readmission Probability", f"{prediction_result['readmission_percentage']}%")
                st.markdown(
                    f"**Stratified Risk Tier:** <span style='color:{prediction_result['risk_color']}; font-size:1.3rem; font-weight:700;'>{prediction_result['risk_tier']}</span>",
                    unsafe_allow_html=True
                )
                st.caption(f"Classification Label: **{prediction_result['prediction_label']}**")

            with res_col2:
                st.markdown(f"**Recommended Action Protocol:**\n\n{prediction_result['recommendation']}")

            st.markdown("#### Primary Elevated Risk Indicators for this Patient")
            for factor in prediction_result["contributing_factors"]:
                st.markdown(f"- ⚠️ {factor}")

        except Exception as err:
            st.error(f"Inference Error: {err}")
            st.info("Tip: Ensure the model has been trained via 'python train_model.py' or through the Model Performance tab.")

# -------------------------------------------------------------
# PAGE 3: EXPLORATORY ANALYTICS
# -------------------------------------------------------------
elif menu_choice == "3. Exploratory Analytics":
    st.markdown('<div class="main-title">Exploratory Healthcare Data Analytics</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Cohort patterns, encounter distributions, and readmission risk associations.</div>', unsafe_allow_html=True)

    df, _ = get_cached_dataset()

    tab1, tab2 = st.tabs(["Encounter Distributions", "Clinical Associations"])

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            st.pyplot(AnalyticsService.plot_readmission_distribution(df))
        with c2:
            st.pyplot(AnalyticsService.plot_los_vs_readmission(df))

    with tab2:
        c3, c4 = st.columns(2)
        with c3:
            st.pyplot(AnalyticsService.plot_inpatient_visits_vs_readmission(df))
        with c4:
            st.pyplot(AnalyticsService.plot_age_vs_readmission(df))

    st.markdown("---")
    st.subheader("Diagnostic Breakdown")
    st.pyplot(AnalyticsService.plot_primary_diagnosis_distribution(df))

    st.markdown("### Clinical Insight Summary")
    st.markdown("""
    - **Prior Hospitalizations**: Patients with 2 or more prior inpatient admissions exhibit sharply elevated 30-day readmission rates.
    - **Length of Stay**: Hospitalizations exceeding 7 days demonstrate increased likelihood of return due to higher illness severity.
    - **Polypharmacy Impact**: Patients prescribed > 15 distinct medications require rigorous post-discharge reconciliation to avoid drug-interaction readmissions.
    """)

# -------------------------------------------------------------
# PAGE 4: MODEL PERFORMANCE
# -------------------------------------------------------------
elif menu_choice == "4. Model Performance":
    st.markdown('<div class="main-title">Model Evaluation & Performance Metrics</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Evaluation metrics calculated on the 20% holdout test partition without data leakage.</div>', unsafe_allow_html=True)

    if config.METRICS_PATH.exists():
        with open(config.METRICS_PATH, "r") as f:
            metrics = json.load(f)

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Algorithm", metrics["model_type"])
        m2.metric("Test Accuracy", f"{metrics['accuracy'] * 100:.2f}%")
        m3.metric("Test Precision", f"{metrics['precision']:.4f}")
        m4.metric("Test Recall", f"{metrics['recall']:.4f}")
        m5.metric("ROC-AUC Score", f"{metrics['roc_auc']:.4f}")

        st.markdown("---")
        c_cm, c_fi = st.columns(2)
        with c_cm:
            st.pyplot(AnalyticsService.plot_confusion_matrix(metrics["confusion_matrix"]))
        with c_fi:
            if config.FEATURE_IMPORTANCE_PATH.exists():
                st.image(str(config.FEATURE_IMPORTANCE_PATH), caption="Top 10 Clinical Risk Feature Importances")
            else:
                st.info("Feature importance plot will appear after retraining.")

        st.markdown("### Evaluation Interpretation & Methodology")
        st.markdown(f"""
        - **Validation Strategy**: 80% Training / 20% Testing stratified split ({metrics.get('train_samples', 4000):,} train, {metrics.get('test_samples', 1000):,} test).
        - **Leakage Prevention**: All imputation (median/mode) and categorical encoders were fitted strictly on training data.
        - **Recall ({metrics['recall']:.2%})**: Measures sensitivity — the proportion of actual readmissions correctly identified for clinical intervention.
        """)

    else:
        st.warning("No saved model evaluation metrics found. Please train the model below.")

    st.markdown("---")
    st.subheader("Retrain Model Pipeline")
    selected_model = st.selectbox("Select Classification Algorithm", ["Random Forest", "Logistic Regression"], index=0)
    if st.button("Retrain Model Now", type="secondary"):
        with st.spinner(f"Fitting Scikit-learn Pipeline ({selected_model}) on CPU..."):
            df, _ = get_cached_dataset()
            X_train, X_test, y_train, y_test = PreprocessingService.split_data(df)
            pipeline, new_metrics = TrainingService.train_and_evaluate(
                X_train, X_test, y_train, y_test, model_type=selected_model
            )
            PredictionService.invalidate_cache()
            st.success(f"Successfully retrained {selected_model}! Accuracy: {new_metrics['accuracy']:.4f}, ROC-AUC: {new_metrics['roc_auc']:.4f}")
            st.rerun()

# -------------------------------------------------------------
# PAGE 5: METHODOLOGY & SDGS
# -------------------------------------------------------------
elif menu_choice == "5. Methodology & SDGs":
    st.markdown('<div class="main-title">Methodology, SDGs & Academic Review</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Alignment with IARE SLAM Project Review-2 Presentation Requirements & UN Sustainable Development Goals.</div>', unsafe_allow_html=True)

    sdg1, sdg2 = st.columns(2)
    with sdg1:
        st.markdown("### 🩺 SDG 3: Good Health and Well-Being")
        st.markdown("""
        - **Target 3.8**: Achieve universal health coverage and access to quality healthcare services.
        - **Contribution**: Anticipating patient readmissions facilitates timely transitional care, reduces avoidable complications, and improves clinical outcomes post-discharge.
        """)
    with sdg2:
        st.markdown("### 💡 SDG 9: Industry, Innovation and Infrastructure")
        st.markdown("""
        - **Target 9.5**: Enhance scientific research and upgrade technological capabilities in industrial sectors.
        - **Contribution**: Demonstrates practical deployment of modern machine-learning pipelines to modernize clinical informatics infrastructure on standard computing hardware.
        """)

    st.markdown("---")
    st.subheader("Key Literature Foundations (Slide 6–8)")
    lit_data = [
        {"S.No": 1, "Title": "Predictive models for hospital readmission risk: A systematic review", "Source": "CMPB, 2018", "Focus": "Reviewed 77 studies on hospital readmission prediction and examined statistical and ML methods."},
        {"S.No": 2, "Title": "Application of machine learning in predicting hospital readmissions", "Source": "BMC Med. Res. Methodol., 2021", "Focus": "Identified tree-based models, neural networks, logistic regression and SVM as standard approaches."},
        {"S.No": 3, "Title": "Use of electronic medical records in risk prediction models of readmission", "Source": "BMJ, 2020", "Focus": "Reviewed prediction models using electronic medical records for 28- and 30-day hospital readmission."},
        {"S.No": 4, "Title": "Predicting hospital readmission in multimorbid patients with AI", "Source": "Eur. J. Public Health, 2024", "Focus": "Examined AI models forecasting 30-day unplanned readmissions among multimorbid cohorts."},
        {"S.No": 5, "Title": "Prediction of 30-day readmission in diabetes management using ML", "Source": "Comput. Biol. Med., 2025", "Focus": "Developed ML approaches for forecasting 30-day readmission among patients with diabetes."},
        {"S.No": 6, "Title": "Identifying Diabetic Patients with High Risk of Readmission", "Source": "Research Conference, 2016", "Focus": "Applied machine learning to identify diabetic patients at higher risk of readmission."}
    ]
    st.table(pd.DataFrame(lit_data))

# Academic Disclaimer (Visible across all pages)
st.markdown(f"""
<div class="disclaimer-box">
    <strong>Academic Prototype Disclaimer:</strong> {config.PROJECT_META['disclaimer']}
</div>
""", unsafe_allow_html=True)

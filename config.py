"""
Global Configuration for Smart Healthcare Analytics & Readmission Prediction System.
Targeted for CPU execution on Windows 11 (Dell Latitude E6420).
Aligned with IARE SLAM Project Review-2 specifications.
"""
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
DOCS_DIR = BASE_DIR / "docs"

# Create directories if they do not exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
DOCS_DIR.mkdir(parents=True, exist_ok=True)

# Dataset Paths
DATASET_PATH = DATA_DIR / "diabetic_data.csv"
SYNTHETIC_DATASET_PATH = DATA_DIR / "synthetic_demo_cohort.csv"

# Model & Evaluation Artifact Paths
MODEL_PATH = MODELS_DIR / "readmission_pipeline.joblib"
METRICS_PATH = MODELS_DIR / "evaluation_metrics.json"
CONFUSION_MATRIX_PATH = ARTIFACTS_DIR / "confusion_matrix.png"
FEATURE_IMPORTANCE_PATH = ARTIFACTS_DIR / "feature_importance.png"

# Reproducibility and Validation Split
RANDOM_STATE = 42
TEST_SIZE = 0.20  # 80/20 train/test split per Review-2 presentation Slide 12

# Target Variable
TARGET_COLUMN = "readmitted"

# Clinical Numeric Features
NUMERIC_FEATURES = [
    "time_in_hospital",
    "num_lab_procedures",
    "num_procedures",
    "num_medications",
    "number_outpatient",
    "number_emergency",
    "number_inpatient",
    "number_diagnoses",
]

# Clinical Categorical Features
CATEGORICAL_FEATURES = [
    "age_group",
    "gender",
    "admission_type",
    "admission_source",
    "primary_diagnosis",
    "max_glu_serum",
    "a1c_result",
    "insulin",
    "change_medication",
    "diabetes_med",
]

# Feature Documentation & Help Text for UI Form
FEATURE_METADATA = {
    "time_in_hospital": {"label": "Length of Stay (Days)", "min": 1, "max": 14, "default": 4, "help": "Number of days spent in the hospital (1-14)."},
    "num_lab_procedures": {"label": "Lab Tests Conducted", "min": 1, "max": 120, "default": 43, "help": "Total number of lab tests performed during this encounter."},
    "num_procedures": {"label": "Non-Lab Procedures", "min": 0, "max": 6, "default": 1, "help": "Number of medical/surgical procedures performed."},
    "num_medications": {"label": "Medications Prescribed", "min": 1, "max": 60, "default": 15, "help": "Total count of distinct medications prescribed."},
    "number_outpatient": {"label": "Prior Outpatient Visits", "min": 0, "max": 20, "default": 0, "help": "Outpatient clinic encounters in the preceding year."},
    "number_emergency": {"label": "Prior Emergency Visits", "min": 0, "max": 15, "default": 0, "help": "Emergency room encounters in the preceding year."},
    "number_inpatient": {"label": "Prior Inpatient Admissions", "min": 0, "max": 15, "default": 1, "help": "Number of hospital admissions in the preceding year."},
    "number_diagnoses": {"label": "Number of Diagnoses", "min": 1, "max": 16, "default": 7, "help": "Total active clinical diagnosis codes entered."},
}

# Categorical Feature Choice Options
CATEGORICAL_CHOICES = {
    "age_group": ["[0-20)", "[20-40)", "[40-60)", "[60-80)", "[80-100)"],
    "gender": ["Female", "Male"],
    "admission_type": ["Emergency", "Urgent", "Elective"],
    "admission_source": ["Emergency Room", "Physician Referral", "Clinic Referral", "Transfer"],
    "primary_diagnosis": ["Circulatory", "Diabetes", "Respiratory", "Digestive", "Injury", "Musculoskeletal", "Other"],
    "max_glu_serum": ["None", "Normal", ">200", ">300"],
    "a1c_result": ["None", "Normal", ">7", ">8"],
    "insulin": ["No", "Steady", "Up", "Down"],
    "change_medication": ["No", "Ch"],
    "diabetes_med": ["No", "Yes"],
}

# Academic Project Metadata
PROJECT_META = {
    "title": "Smart Healthcare Analytics and Patient Readmission Prediction System",
    "institution": "Institute of Aeronautical Engineering (IARE)",
    "department": "Computer Science and Engineering",
    "review": "SLAM Project Review-2",
    "guide": "Ms. Y Sujana",
    "students": [
        {"name": "A. Akshith", "roll_no": "2125951A05"},
        {"name": "K. Anandh", "roll_no": "2125951A0527"}
    ],
    "sdgs": ["SDG 3 (Good Health and Well-being)", "SDG 9 (Industry, Innovation and Infrastructure)"],
    "disclaimer": "Academic Prototype Disclaimer: This software is an educational prototype developed for SLAM Project Review-2 at IARE. It is not certified for clinical diagnostic use or real-world patient care decisions."
}

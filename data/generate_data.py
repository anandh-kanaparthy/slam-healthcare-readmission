"""
Data Preparation & Synthetic Cohort Generator for SLAM Healthcare System.
Generates a statistically representative clinical cohort mirroring the
UCI Diabetes 130-US Hospitals readmission dataset distributions.

PROMINENT NOTICE:
The output dataset is SYNTHETIC — FOR SOFTWARE TESTING AND ACADEMIC DEMONSTRATION ONLY.
It does not contain real patient records and must not be used for actual clinical decisions.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parents[1]))
import config

def generate_synthetic_cohort(n_records: int = 5000, output_path: Path = None) -> pd.DataFrame:
    """
    Generate a high-fidelity synthetic healthcare cohort for offline pipeline validation.
    """
    np.random.seed(config.RANDOM_STATE)

    # 1. Demographics & Encounters
    age_groups = ["[0-20)", "[20-40)", "[40-60)", "[60-80)", "[80-100)"]
    age_probs = [0.03, 0.12, 0.35, 0.40, 0.10]

    genders = ["Female", "Male"]
    gender_probs = [0.53, 0.47]

    admission_types = ["Emergency", "Urgent", "Elective"]
    admission_type_probs = [0.60, 0.25, 0.15]

    admission_sources = ["Emergency Room", "Physician Referral", "Clinic Referral", "Transfer"]
    admission_source_probs = [0.45, 0.35, 0.12, 0.08]

    diagnoses = ["Circulatory", "Diabetes", "Respiratory", "Digestive", "Injury", "Musculoskeletal", "Other"]
    diagnoses_probs = [0.30, 0.18, 0.15, 0.12, 0.08, 0.05, 0.12]

    glu_levels = ["None", "Normal", ">200", ">300"]
    glu_probs = [0.85, 0.06, 0.05, 0.04]

    a1c_levels = ["None", "Normal", ">7", ">8"]
    a1c_probs = [0.75, 0.10, 0.07, 0.08]

    insulin_states = ["No", "Steady", "Up", "Down"]
    insulin_probs = [0.45, 0.30, 0.15, 0.10]

    df = pd.DataFrame({
        "patient_id": [f"DEMO_PID_{100000 + i}" for i in range(n_records)],
        "age_group": np.random.choice(age_groups, size=n_records, p=age_probs),
        "gender": np.random.choice(genders, size=n_records, p=gender_probs),
        "admission_type": np.random.choice(admission_types, size=n_records, p=admission_type_probs),
        "admission_source": np.random.choice(admission_sources, size=n_records, p=admission_source_probs),
        "primary_diagnosis": np.random.choice(diagnoses, size=n_records, p=diagnoses_probs),
        "time_in_hospital": np.random.poisson(lam=4.2, size=n_records).clip(1, 14),
        "num_lab_procedures": np.random.normal(loc=43, scale=18, size=n_records).astype(int).clip(1, 120),
        "num_procedures": np.random.poisson(lam=1.3, size=n_records).clip(0, 6),
        "num_medications": np.random.poisson(lam=15.5, size=n_records).clip(1, 60),
        "number_outpatient": np.random.negative_binomial(n=0.5, p=0.7, size=n_records).clip(0, 15),
        "number_emergency": np.random.negative_binomial(n=0.3, p=0.8, size=n_records).clip(0, 10),
        "number_inpatient": np.random.negative_binomial(n=0.4, p=0.6, size=n_records).clip(0, 12),
        "number_diagnoses": np.random.poisson(lam=7.2, size=n_records).clip(1, 16),
        "max_glu_serum": np.random.choice(glu_levels, size=n_records, p=glu_probs),
        "a1c_result": np.random.choice(a1c_levels, size=n_records, p=a1c_probs),
        "insulin": np.random.choice(insulin_states, size=n_records, p=insulin_probs),
        "change_medication": np.random.choice(["No", "Ch"], size=n_records, p=[0.54, 0.46]),
        "diabetes_med": np.random.choice(["No", "Yes"], size=n_records, p=[0.23, 0.77]),
    })

    # Realistic Readmission Risk Scoring based on empirical literature
    log_odds = (
        -2.6
        + 0.14 * df["time_in_hospital"]
        + 0.52 * df["number_inpatient"]
        + 0.31 * df["number_emergency"]
        + 0.04 * df["num_medications"]
        + 0.09 * df["number_diagnoses"]
        + (df["admission_type"] == "Emergency").astype(int) * 0.35
        + (df["primary_diagnosis"] == "Circulatory").astype(int) * 0.40
        + (df["primary_diagnosis"] == "Diabetes").astype(int) * 0.35
        + (df["insulin"].isin(["Up", "Down"])).astype(int) * 0.32
        + (df["a1c_result"].isin([">7", ">8"])).astype(int) * 0.28
        + (df["age_group"].isin(["[60-80)", "[80-100)"])).astype(int) * 0.22
    )

    probabilities = 1.0 / (1.0 + np.exp(-log_odds))
    # Binary readmission label: 1 if readmitted within 30 days, 0 otherwise
    df["readmitted"] = (np.random.rand(n_records) < probabilities).astype(int)

    dest = output_path or config.SYNTHETIC_DATASET_PATH
    df.to_csv(dest, index=False)
    print(f"[DATA GENERATOR] Generated {len(df):,} synthetic patient records.")
    print(f"[DATA GENERATOR] Output location: {dest}")
    print(f"[DATA GENERATOR] Readmission Rate: {df['readmitted'].mean() * 100:.2f}%")
    return df

if __name__ == "__main__":
    generate_synthetic_cohort()

# Healthcare Readmission Dataset Documentation

## 1. Overview and Problem Context
This project investigates hospital readmission prediction, a critical healthcare informatics challenge. 30-day unplanned readmissions place significant clinical and financial strain on healthcare systems and indicate potential gaps in transition-of-care or post-discharge management.

## 2. Dataset Options Supported

### A. Primary Real Dataset: UCI Diabetes 130-US Hospitals (1999–2008)
- **Source**: UCI Machine Learning Repository / Strack et al. (2014)
- **Cohort**: 101,766 hospital encounters from 130 participating US hospitals representing diabetic inpatients.
- **Features**: Patient demographics, admission details, length of stay, laboratory test counts, medication regimens, glucose/HbA1c test results, and prior utilization (inpatient, outpatient, emergency).
- **Target Definition**: Readmission within 30 days (`<30`), after 30 days (`>30`), or none (`NO`). Mapped to binary classification:
  - `1` = Readmitted within 30 days (`<30`)
  - `0` = Not readmitted within 30 days (`>30` or `NO`)

### B. High-Fidelity Demonstration Cohort
- **File**: `data/synthetic_demo_cohort.csv`
- **Label**: **SYNTHETIC — FOR SOFTWARE TESTING ONLY**
- **Purpose**: Enables immediate, reliable, offline execution on CPU-constrained laptops without requiring external network downloads or exposing protected patient health information.
- **Design**: Statistically models the marginal and conditional distributions of the UCI Diabetes readmission dataset, preserving clinical relationships (e.g. prior inpatient encounters, extended length of stay, and medication adjustments elevating readmission risk).

## 3. Data Dictionary

| Feature | Type | Range / Categories | Description |
| :--- | :--- | :--- | :--- |
| `time_in_hospital` | Numeric | 1 – 14 | Total days spent in the hospital |
| `num_lab_procedures` | Numeric | 1 – 120 | Count of laboratory procedures administered |
| `num_procedures` | Numeric | 0 – 6 | Count of non-laboratory clinical procedures |
| `num_medications` | Numeric | 1 – 60 | Total count of unique medications prescribed |
| `number_outpatient` | Numeric | 0 – 20 | Outpatient visits in the preceding 12 months |
| `number_emergency` | Numeric | 0 – 15 | Emergency encounters in the preceding 12 months |
| `number_inpatient` | Numeric | 0 – 15 | Inpatient admissions in the preceding 12 months |
| `number_diagnoses` | Numeric | 1 – 16 | Number of active clinical diagnosis codes |
| `age_group` | Categorical | `[0-20)`, `[20-40)`, `[40-60)`, `[60-80)`, `[80-100)` | Patient age decade bracket |
| `gender` | Categorical | `Female`, `Male` | Patient biological sex |
| `admission_type` | Categorical | `Emergency`, `Urgent`, `Elective` | Admission urgency classification |
| `admission_source` | Categorical | `Emergency Room`, `Physician Referral`, `Clinic Referral`, `Transfer` | Route of patient referral |
| `primary_diagnosis` | Categorical | `Circulatory`, `Diabetes`, `Respiratory`, `Digestive`, `Injury`, `Other` | Primary ICD category |
| `max_glu_serum` | Categorical | `None`, `Normal`, `>200`, `>300` | Peak glucose serum lab test |
| `a1c_result` | Categorical | `None`, `Normal`, `>7`, `>8` | Glycated hemoglobin test result |
| `insulin` | Categorical | `No`, `Steady`, `Up`, `Down` | Insulin therapy state and titration |
| `change_medication` | Categorical | `No`, `Ch` | Medication adjusted during stay |
| `diabetes_med` | Categorical | `No`, `Yes` | Diabetic medication prescribed |
| **`readmitted`** | **Binary Target** | **0 (No), 1 (Yes)** | **Unplanned readmission within 30 days** |

## 4. Privacy & Ethical Safeguards
- No Personally Identifiable Information (PII) or Protected Health Information (PHI) is included.
- Identifiers (`patient_id`, `encounter_id`) are stripped before feature modeling to prevent arbitrary identification and memory leakage.

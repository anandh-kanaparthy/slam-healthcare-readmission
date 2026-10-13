# Smart Healthcare Analytics and Patient Readmission Prediction System

**SLAM Project Review-2 — Department of Computer Science and Engineering**
**Institution:** Institute of Aeronautical Engineering (IARE), Hyderabad
**Guide:** Ms. Y Sujana (Assistant Professor, CSE)
**Students:**
- **A. Akshith** (Roll No: `2125951A05`)
- **K. Anandh** (Roll No: `2125951A0527`)
**SDGs Mapped:** **SDG 3** (Good Health and Well-Being), **SDG 9** (Industry, Innovation and Infrastructure)

---

## 1. Executive Summary & Problem Statement

Hospital readmission within 30 days of discharge is a major clinical challenge. High readmission rates reflect care gaps, increase healthcare expenditures, and strain hospital bed capacity. Identifying high-risk patients manually from multivariate clinical charts is time-consuming and subjective.

This project delivers an end-to-end Machine Learning and Healthcare Analytics prototype that forecasts 30-day readmission risk from historical patient encounters. The system supports post-discharge planning by categorizing patients into actionable risk tiers (**Low**, **Moderate**, **High**) and highlighting individual clinical risk factors.

---

## 2. Key Features

- **Decoupled Architecture**: Pure Python service layer (`data_service`, `preprocessing_service`, `training_service`, `prediction_service`, `analytics_service`).
- **Interactive Streamlit Dashboard**:
  - **Overview**: Real-time cohort metrics, provenance badges, and workflow diagrams.
  - **Patient Risk Predictor**: Multi-parameter clinical input form with instant probability score, risk tier, and recommended action protocol.
  - **Exploratory Analytics**: Interactive distribution charts (Length of Stay, Prior Encounters, Age Brackets, and Primary Diagnoses).
  - **Model Performance**: Test set metrics, confusion matrix heatmap, feature importance analysis, and live retraining.
  - **Methodology & SDGs**: Research literature review (2018–2025) and UN SDG mappings.
- **Leakage-Free ML Pipeline**: Stratified $80/20$ train/test splitting performed *prior* to fitting preprocessors.
- **100% Free & Open-Source**: Runs offline on dual-core CPU hardware (Dell Latitude E6420 under Windows 11). Zero paid APIs or cloud dependencies.
- **Automated QA Test Suite**: 15 comprehensive unit and integration tests with Pytest.

---

## 3. Technology Stack

- **Language:** Python 3.14 (64-bit)
- **Data Manipulation:** Pandas, NumPy
- **Machine Learning:** Scikit-learn (ColumnTransformer, Pipeline, Random Forest, Logistic Regression)
- **Visualization:** Matplotlib (Agg backend), Seaborn
- **User Interface:** Streamlit
- **Model Persistence:** Joblib
- **Testing:** Pytest

---

## 4. Dataset Provenance & Data Dictionary

### Supported Datasets
1. **Real Dataset**: UCI Diabetes 130-US Hospitals (1999–2008), 101,766 inpatient encounters across 130 US hospitals.
2. **Synthetic Demonstration Cohort**: `data/synthetic_demo_cohort.csv`
   > ⚠️ **NOTICE: SYNTHETIC — FOR SOFTWARE TESTING ONLY.** Generated strictly for offline software validation without exposing protected health information (PHI).

### Feature Dictionary

| Feature | Type | Range / Options | Description |
| :--- | :--- | :--- | :--- |
| `time_in_hospital` | Numeric | 1 – 14 | Total days admitted in acute care |
| `num_lab_procedures` | Numeric | 1 – 120 | Count of laboratory procedures administered |
| `num_procedures` | Numeric | 0 – 6 | Count of non-laboratory clinical procedures |
| `num_medications` | Numeric | 1 – 60 | Total count of distinct medications prescribed |
| `number_outpatient` | Numeric | 0 – 20 | Outpatient visits in preceding 12 months |
| `number_emergency` | Numeric | 0 – 15 | Emergency encounters in preceding 12 months |
| `number_inpatient` | Numeric | 0 – 15 | Inpatient admissions in preceding 12 months |
| `number_diagnoses` | Numeric | 1 – 16 | Number of active clinical diagnosis codes |
| `age_group` | Categorical | `[0-20)`, `[20-40)`, `[40-60)`, `[60-80)`, `[80-100)` | Patient age decade bracket |
| `gender` | Categorical | `Female`, `Male` | Patient biological sex |
| `admission_type` | Categorical | `Emergency`, `Urgent`, `Elective` | Urgency classification of admission |
| `admission_source` | Categorical | `Emergency Room`, `Physician Referral`, `Clinic Referral`, `Transfer` | Route of admission referral |
| `primary_diagnosis` | Categorical | `Circulatory`, `Diabetes`, `Respiratory`, `Digestive`, `Injury`, `Other` | Primary ICD diagnosis category |
| `max_glu_serum` | Categorical | `None`, `Normal`, `>200`, `>300` | Peak glucose serum test result |
| `a1c_result` | Categorical | `None`, `Normal`, `>7`, `>8` | Glycated hemoglobin test result |
| `insulin` | Categorical | `No`, `Steady`, `Up`, `Down` | Insulin titration during stay |
| `change_medication` | Categorical | `No`, `Ch` | Medication dosage change during stay |
| `diabetes_med` | Categorical | `No`, `Yes` | Diabetic medication prescribed |
| **`readmitted`** | **Target** | **0 (No), 1 (Yes)** | **Unplanned 30-day hospital readmission** |

---

## 5. Machine Learning Methodology & Actual Results

### Preprocessing & Validation
- **Partitioning**: $80\%$ Training ($4,000$ encounters) / $20\%$ Test ($1,000$ encounters), stratified by target class.
- **Transformations**: Numeric features imputed with median and scaled using `StandardScaler`. Categorical features imputed with mode and encoded using `OneHotEncoder(handle_unknown="ignore")`.

### Actual Computed Metrics (Random Forest Classifier)

| Metric | Measured Score | Evaluation Meaning |
| :--- | :--- | :--- |
| **Accuracy** | **58.40%** | Overall proportion of correct classifications |
| **Precision** | **59.34%** | Positive predictive value (predicted readmissions that were true) |
| **Recall (Sensitivity)** | **59.57%** | Sensitivity (true readmitted patients correctly identified) |
| **F1-Score** | **59.45%** | Harmonic mean balancing precision and recall |
| **ROC-AUC** | **0.6272** | Discrimination capacity across probability thresholds |

**Holdout Confusion Matrix (1,000 Test Encounters):**
- True Negatives (No Readmission): **279**
- False Positives: **209**
- False Negatives: **207**
- True Positives (Readmitted): **305**

---

## 6. Quick Setup & Execution Guide (PowerShell)

### Step 1: Open Terminal in Project Folder
```powershell
cd "C:\Users\DELL\SLAM-Readmission-Prototype"
```

### Step 2: Activate the Virtual Environment
```powershell
.\venv\Scripts\Activate.ps1
```

### Step 3: Run the Automated Test Suite (15 Tests)
```powershell
pytest -v
```

### Step 4: Retrain Model (Optional CLI)
```powershell
python train_model.py --model "Random Forest"
```

### Step 5: Launch the Streamlit Dashboard
```powershell
streamlit run app.py
```
*The interactive dashboard will open automatically in your browser at `http://localhost:8501`.*

---

## 7. Project Directory Structure

```text
SLAM-Readmission-Prototype/
├── .gitignore                      # Git ignore rules for cache & venvs
├── requirements.txt                # Lightweight dependencies
├── config.py                       # Global configuration, paths, features, metadata
├── train_model.py                  # Standalone CLI training script
├── app.py                          # 5-Page Streamlit Healthcare Dashboard
├── data/
│   ├── README.md                   # Data dictionary and provenance documentation
│   ├── generate_data.py            # High-fidelity synthetic cohort generator
│   └── synthetic_demo_cohort.csv   # 5,000 patient records (clearly labeled)
├── services/
│   ├── __init__.py
│   ├── data_service.py             # Data loading, validation, summary stats
│   ├── preprocessing_service.py    # Leakage-free ColumnTransformer & stratified split
│   ├── training_service.py         # Pipeline fitting, evaluation, and artifact saving
│   ├── prediction_service.py       # Real-time inference, validation, and risk tiers
│   └── analytics_service.py        # Clean clinical charts (headless Agg backend)
├── models/
│   ├── readmission_pipeline.joblib # Persisted Scikit-learn pipeline
│   └── evaluation_metrics.json     # Saved actual evaluation metrics
├── artifacts/
│   ├── confusion_matrix.png        # Saved confusion matrix plot
│   └── feature_importance.png      # Top 10 clinical risk drivers plot
├── tests/
│   ├── __init__.py
│   ├── test_data_service.py        # Tests data loading and schema validation
│   ├── test_preprocessing.py       # Tests leakage-free transformations and splits
│   ├── test_training.py            # Tests model training and metrics calculations
│   ├── test_prediction.py          # Tests inference and boundary input handling
│   └── test_model_pipeline.py      # End-to-end integration test
└── docs/
    ├── project_architecture.md     # System architecture and technical design
    └── review_presentation_guide.md# Guide and live demo script for SLAM Review-2
```

---

## 8. Alignment with UN Sustainable Development Goals

- **SDG 3: Good Health and Well-Being (Target 3.8)**: Promotes proactive transitional care for discharged patients, reducing avoidable complications and emergency department crowding.
- **SDG 9: Industry, Innovation, and Infrastructure (Target 9.5)**: Advances intelligent data-driven healthcare analytics that can operate on standard hospital computing hardware.

---

## 9. Academic & Ethical Disclaimer

> **ACADEMIC PROTOTYPE NOTICE:** This project is an educational software prototype created for SLAM Project Review-2 at the Institute of Aeronautical Engineering (IARE). It is **not** certified as medical software and must **never** be used to make actual clinical decisions or substitute for licensed medical judgment.

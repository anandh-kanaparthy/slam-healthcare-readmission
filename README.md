# Smart Healthcare Analytics and Patient Readmission Prediction System

**SLAM Project Review-2 — Department of Computer Science and Engineering**  
**Institution:** Institute of Aeronautical Engineering (IARE), Hyderabad  
**Guide:** Ms. Y Sujana (Assistant Professor, Department of CSE)  
**Students:**  
- **A. Akshith** (Roll No: `25951A0521`)  
- **K. Anandh** (Roll No: `25951A0527`)  
**Project Presentation:** `slam project(2).pptx`  
**GitHub Repository:** [https://github.com/anandh-kanaparthy/slam-healthcare-readmission](https://github.com/anandh-kanaparthy/slam-healthcare-readmission)  
**UN SDGs Mapped:** **SDG 3** (Good Health and Well-Being — Target 3.8), **SDG 9** (Industry, Innovation and Infrastructure — Target 9.5)  

---

## 1. Executive Summary & Clinical Context

Unplanned hospital readmissions within 30 days of discharge represent a major burden on healthcare systems globally. Patients returning unexpectedly to acute care consume critical inpatient resources, elevate hospital operational costs, and indicate opportunities to optimize transitional care planning.

This system provides a full-stack, data-driven machine learning application designed for hospital readmission surveillance, retrospective analytics, and human-in-the-loop patient review triaging.

---

## 2. Product Workflow & Architecture

The application is organized into 5 core operational modules:

```text
[ Patient Dataset ] ➔ [ Data Preprocessing ] ➔ [ Feature Selection ] ➔ [ ML Training ] ➔ [ Prediction Engine ] ➔ [ Triage Queue & Dashboard ]
```

### The 5 Operational Product Pages:
1. **Hospital Overview**: High-level hospital operational metrics, population readmission distributions, interactive cohort filters (by diagnosis, age, admission urgency), and encounter sample viewer.
2. **Readmission Analytics**: Deep-dive clinical subgroup risk patterns, sample size ($N$) reporting, empirical relative risk calculations, and multivariate factor comparisons.
3. **Data Quality & Preprocessing**: Ingestion hygiene audit, duplicate detection, missing value handling (median for numeric, mode for categorical), leakage-free `ColumnTransformer` visualization, and target association ranking.
4. **Model Performance**: Held-out test set metrics ($N=1,000$ unseen encounters), confusion matrix, ROC curve, feature importances, side-by-side Random Forest vs. Logistic Regression benchmark, and deliberate retraining controls.
5. **Patient Risk Review**: SQLite-backed clinical review queue (`data/review_store.db`), priority score threshold slider, patient profile inspector, persistent clinician review notes and decisions, and real-time intake scoring form.

---

## 3. Dataset Provenance & Data Hygiene

### Active Dataset
- **Synthetic Demonstration Cohort**: `data/synthetic_demo_cohort.csv` (5,000 encounters, 18 features).  
  > ⚠️ **NOTICE: SYNTHETIC — FOR SOFTWARE TESTING AND ACADEMIC DEMONSTRATION ONLY.** Statistically modeled after the public UCI Diabetes 130-US Hospitals (1999–2008) dataset. It contains **zero Protected Health Information (PHI)** and is used for demonstration purposes.

### Clinical Features:
- **Numerical (8)**: `time_in_hospital`, `num_lab_procedures`, `num_procedures`, `num_medications`, `number_outpatient`, `number_emergency`, `number_inpatient`, `number_diagnoses`.
- **Categorical (10)**: `age_group`, `gender`, `admission_type`, `admission_source`, `primary_diagnosis`, `max_glu_serum`, `a1c_result`, `insulin`, `change_medication`, `diabetes_med`.
- **Target Label**: `readmitted` (Binary: 0 = No Readmission, 1 = Readmitted within 30 days).

---

## 4. Leakage-Free Preprocessing Architecture

- **Stratified Partitioning**: 80% Training ($N=4,000$) / 20% Testing ($N=1,000$).
- **Leakage Prevention**: All transformations (`SimpleImputer`, `StandardScaler`, `OneHotEncoder`) are fit **strictly on the 80% training partition**. The holdout test set is evaluated without re-fitting.
- **Identifier Exclusion**: Patient identification numbers (`patient_id`) are stripped before model fitting to prevent artificial memorization.

---

## 5. Actual Model Evaluation Results

Evaluated on the 20% holdout test partition ($N=1,000$ unseen encounters):

| Metric | Measured Score | Clinical Context |
| :--- | :--- | :--- |
| **Model Architecture** | **Random Forest Classifier** | 100 balanced trees, max depth 10 |
| **Test Accuracy** | **58.40%** | Proportion of overall correct classifications on unseen test encounters |
| **Test Precision** | **59.34%** | Accuracy of high-risk classifications |
| **Test Recall (Sensitivity)** | **59.57%** | True readmissions correctly identified by the pipeline |
| **Test F1-Score** | **59.45%** | Harmonic balance between precision and sensitivity |
| **Test ROC-AUC** | **0.6272** | Discrimination capacity across probability thresholds |

### Test Set Confusion Matrix (1,000 Test Records)
- **True Negatives**: 279
- **False Positives**: 209
- **False Negatives**: 207
- **True Positives**: 305

---

## 6. Setup & Execution Guide (Windows PowerShell)

### Step 1: Open Terminal in Project Folder
```powershell
cd "C:\Users\DELL\SLAM-Readmission-Prototype"
```

### Step 2: Activate the Virtual Environment
```powershell
.\venv\Scripts\Activate.ps1
```

### Step 3: Run the Automated Test Suite (25 Tests)
```powershell
pytest -v
```
*All 25 automated tests verify data ingestion, schema validation, data cleaning, preprocessing stratification, prediction boundary validation, SQLite review persistence, and training benchmarking.*

### Step 4: Run CLI Model Training (Optional)
```powershell
python train_model.py --model "Random Forest"
```

### Step 5: Launch the Streamlit Dashboard
```powershell
streamlit run app.py
```
*The interactive dashboard will open automatically in your browser at `http://localhost:8501`.*

---

## 7. Git Workflow (Review & Push to Existing GitHub Repository)

Review your local changes and push them to your existing repository on branch `main`:

```powershell
# 1. Review status and modified files
git status

# 2. View differences against origin/main
git diff

# 3. Stage the verified files
git add app.py config.py train_model.py requirements.txt README.md
git add services/ data/ tests/ docs/ artifacts/ models/

# 4. Commit with descriptive message
git commit -m "Upgrade SLAM Readmission Prototype to 5-page hospital analytics and triage review product"

# 5. Push to GitHub main branch
git push origin main
```

---

## 8. Academic & Ethical Disclaimer

> **ACADEMIC PROTOTYPE NOTICE:**  
> The Smart Healthcare Analytics and Patient Readmission Prediction System is developed solely as an educational software prototype for SLAM Project Review-2 at IARE.  
> - It is **not** certified, validated, or intended to replace clinical medical diagnosis or physician judgment.  
> - Statistical associations in the dataset reflect retrospective observational patterns and do **not** imply direct clinical causation.  
> - The application does **not** generate patient-specific medical orders or treatment prescriptions.

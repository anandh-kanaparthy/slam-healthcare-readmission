# System Architecture & Technical Specifications

**Project Title:** Smart Healthcare Analytics and Patient Readmission Prediction System  
**Institution:** Institute of Aeronautical Engineering (IARE), Hyderabad  
**Review Stage:** SLAM Project Review-2  
**Mapped SDGs:** SDG 3 (Good Health and Well-Being), SDG 9 (Industry, Innovation and Infrastructure)  

---

## 1. Architectural Overview

The application is structured as a decoupled, service-oriented modular architecture designed for lightweight, offline CPU execution on standard hardware (e.g., Dell Latitude E6420 under Windows 11).

```text
+-----------------------------------------------------------------------------------+
|                                  USER INTERFACE                                   |
|                        Streamlit Web Application (app.py)                         |
|  Page 1: Hospital Overview        |  Page 2: Readmission Analytics                |
|  Page 3: Data Quality & Preproc   |  Page 4: Model Performance                    |
|  Page 5: Patient Risk Review (Triage Queue & Intake Evaluator)                    |
+-----------------------------------------------------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                                  SERVICE LAYER                                    |
|                                                                                   |
|  +--------------------+   +-----------------------+   +------------------------+  |
|  |    DataService     |   | PreprocessingService  |   |    TrainingService     |  |
|  | - Schema Validation|   | - Median/Mode Impute  |   | - Pipeline Fitting     |  |
|  | - Cohort Ingestion |   | - StandardScaler      |   | - Stratified Split     |  |
|  | - Provenance Stats |   | - OneHotEncoder       |   | - Metrics & Benchmark  |  |
|  | - Clean & Dedup    |   | - Leakage Prevention  |   | - Confusion & ROC Plots|  |
|  +--------------------+   +-----------------------+   +------------------------+  |
|                                                                                   |
|  +--------------------+   +-----------------------+   +------------------------+  |
|  | PredictionService  |   |   AnalyticsService    |   |     ReviewService      |  |
|  | - Range Validation |   | - Subgroup Analysis   |   | - SQLite Store & Sync  |  |
|  | - Inference Engine |   | - Class Distributions |   | - Queue Filtering/Sort |  |
|  | - Ethical Tiers    |   | - LOS & Inpatient Box |   | - Clinician Note Audit |  |
|  | - Percentile Bench |   | - Headless Agg Backend|   | - Intake Persistence   |  |
|  +--------------------+   +-----------------------+   +------------------------+  |
+-----------------------------------------------------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                               PERSISTENCE LAYER                                   |
|  - data/review_store.db (SQLite database: encounter_reviews table & audit trail)  |
|  - models/readmission_pipeline.joblib (Fitted Scikit-learn Pipeline)              |
|  - models/evaluation_metrics.json (Accuracy, Precision, Recall, F1, ROC-AUC)      |
|  - artifacts/confusion_matrix.png, feature_importance.png, roc_curve.png          |
|  - data/synthetic_demo_cohort.csv / diabetic_data.csv                             |
+-----------------------------------------------------------------------------------+
```

---

## 2. Core Service Modules

### 2.1 `services/data_service.py`
- **Responsibility**: Ingestion, schema verification, data quality auditing, duplicate detection, and provenance tracking.
- **Data Integrity**: Enforces strict verification of all 8 continuous/integer features, 10 categorical features, and the binary target column. Automatically detects and flags whether the active cohort is a synthetic demonstration cohort or a real de-identified cohort.

### 2.2 `services/preprocessing_service.py`
- **Responsibility**: Leakage-free feature transformation and stratified train/test partitioning.
- **Pipeline Structure**:
  - `num`: `SimpleImputer(strategy="median")` $\rightarrow$ `StandardScaler()`
  - `cat`: `SimpleImputer(strategy="most_frequent")` $\rightarrow$ `OneHotEncoder(handle_unknown="ignore", sparse_output=False)`
- **Data Leakage Safeguard**: Stratified partitioning occurs *strictly prior* to any transformer fitting. The `ColumnTransformer` is fitted exclusively on the 80% training partition ($N=4,000$) and applied to the 20% test partition ($N=1,000$). Non-predictive identification codes (`patient_id`) are stripped before fitting to prevent artificial memorization.

### 2.3 `services/training_service.py`
- **Responsibility**: Supervised model training, evaluation metrics calculation, multi-model benchmarking, and artifact serialization.
- **Classifiers Supported**:
  - `RandomForestClassifier`: 100 estimators, `max_depth=10`, `class_weight="balanced"`. Handles non-linear feature interactions without GPU requirements.
  - `LogisticRegression`: High-interpretability linear baseline with balanced class weights.
- **Benchmarking Engine**: `compare_models()` trains and evaluates both architectures on the exact same holdout split to display side-by-side performance tables.

### 2.4 `services/prediction_service.py`
- **Responsibility**: Real-time patient inference, clinical input boundary validation, risk tiering, and empirical cohort comparative factors.
- **Ethical Risk Tiering**:
  - **Lower Relative Risk (<35%)**: Probability below 35%. Consistent with baseline routine discharge follow-up.
  - **Moderate Relative Risk (35% – 65%)**: Intermediate risk band. Suggests heightened vigilance.
  - **Elevated Relative Risk (>65%)**: Top percentile risk band. Flagged for multi-factor clinical review.
- **Empirical Cohort Benchmarking**: Computes factual descriptive comparisons between patient attributes and the active cohort percentiles (e.g., patient LOS compared against cohort median of 4 days, inpatient visits against cohort 75th percentile).

### 2.5 `services/analytics_service.py`
- **Responsibility**: Headless visual analytics and statistical chart generation.
- **Backend Architecture**: Enforces `matplotlib.use("Agg")` across all modules before importing `pyplot`, preventing GUI display conflicts in headless and web-server environments.

### 2.6 `services/review_service.py` *(New)*
- **Responsibility**: SQLite-backed human-in-the-loop review queue and clinical audit trail (`data/review_store.db`).
- **Capabilities**:
  - Cohort synchronization via `INSERT OR IGNORE` to preserve existing review statuses and clinician notes.
  - Dynamic parameterized search, status filtering, priority score thresholding, and sorting.
  - Review decision auditing: Updates review status (`Pending Review`, `Reviewed - Standard`, `Requires Follow-up`, `Escalated to Physician`), records reviewer identity and UTC timestamp.
  - Interactive intake enrollment: Persists newly scored intake patient profiles directly into the review queue.

---

## 3. Technology Stack Justification

| Technology | Role | Justification |
| :--- | :--- | :--- |
| **Python 3.14** | Core Language | Standard, robust programming language for modern data science. |
| **Pandas & NumPy** | Data Manipulation | High-performance vectorized operations on structured clinical tables. |
| **Scikit-learn** | Machine Learning | CPU-efficient, stable, industry-standard ML pipeline with zero GPU requirements. |
| **SQLite 3** | Local Persistence | Zero-configuration, file-based relational storage for clinical review triage queue. |
| **Streamlit** | Web Application | Interactive, multi-page product interface rendered directly in Python. |
| **Matplotlib & Seaborn** | Visualization | Lightweight clinical charts generated in headless mode (`Agg` backend). |
| **Joblib** | Serialization | Efficient serialization of Scikit-learn pipelines with embedded transformers. |
| **Pytest** | Automated Testing | Comprehensive QA test suite (25 automated unit and integration tests). |

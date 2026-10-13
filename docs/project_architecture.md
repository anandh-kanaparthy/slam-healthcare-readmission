# System Architecture & Technical Specifications

**Project Title:** Smart Healthcare Analytics and Patient Readmission Prediction System
**Institution:** Institute of Aeronautical Engineering (IARE)
**Review Stage:** SLAM Project Review-2
**Mapped SDGs:** SDG 3 (Good Health and Well-Being), SDG 9 (Industry, Innovation and Infrastructure)

---

## 1. Architectural Overview

The system is structured as a decoupled, modular service-oriented architecture designed for lightweight CPU execution on standard hardware (e.g., Dell Latitude E6420 under Windows 11).

```
+-----------------------------------------------------------------------------------+
|                                  USER INTERFACE                                   |
|                        Streamlit Web Application (app.py)                         |
|  - Executive Overview   - Patient Predictor   - Analytics   - Model Performance   |
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
|  | - Provenance Stats |   | - OneHotEncoder       |   | - Metrics Calculation  |  |
|  +--------------------+   +-----------------------+   +------------------------+  |
|                                                                                   |
|  +--------------------+   +-----------------------+                               |
|  | PredictionService  |   |   AnalyticsService    |                               |
|  | - Range Validation |   | - Class Distributions |                               |
|  | - Inference Engine |   | - Box/Bar Analytics   |                               |
|  | - Risk Tiers (L/M/H|   | - Confusion Matrices  |                               |
|  +--------------------+   +-----------------------+                               |
+-----------------------------------------------------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                               PERSISTENCE ARTIFACTS                               |
|  - models/readmission_pipeline.joblib (Fitted Scikit-learn Pipeline)              |
|  - models/evaluation_metrics.json (Accuracy, Precision, Recall, F1, ROC-AUC)      |
|  - artifacts/confusion_matrix.png & feature_importance.png                        |
|  - data/synthetic_demo_cohort.csv / diabetic_data.csv                             |
+-----------------------------------------------------------------------------------+
```

---

## 2. Core Service Modules

### 2.1 `services/data_service.py`
- **Responsibility**: Ingestion, dataset schema validation, missingness profiling, and provenance tracking.
- **Data Integrity**: Enforces strict verification of all 8 numerical features, 10 categorical features, and the binary target column. Automatically distinguishes real clinical datasets from clearly labeled synthetic demo cohorts.

### 2.2 `services/preprocessing_service.py`
- **Responsibility**: Leakage-free feature transformation and stratified train/test partitioning.
- **Pipeline Structure**:
  - `num`: `SimpleImputer(strategy="median")` $\rightarrow$ `StandardScaler()`
  - `cat`: `SimpleImputer(strategy="most_frequent")` $\rightarrow$ `OneHotEncoder(handle_unknown="ignore", sparse_output=False)`
- **Data Leakage Safeguard**: Data splitting occurs *prior* to any transformation fitting. The preprocessor is fit solely on the training partition ($80\%$) and applied to the test partition ($20\%$).

### 2.3 `services/training_service.py`
- **Responsibility**: Model fitting, hyperparameter specification, performance evaluation, and artifact serialization.
- **Classifiers Supported**:
  - `RandomForestClassifier`: Ensembles 100 decision trees (`max_depth=10`, `class_weight="balanced"`) to handle non-linear clinical interactions.
  - `LogisticRegression`: High-interpretability baseline with balanced class weighting.
- **Evaluation Outputs**: Accuracy, Precision, Recall, F1-Score, ROC-AUC, and Confusion Matrix.

### 2.4 `services/prediction_service.py`
- **Responsibility**: Real-time patient inference, input boundary validation, risk stratification, and factor attribution.
- **Risk Stratification Engine**:
  - **Low Risk ($< 35\%$)**: Standard discharge plan and routine follow-up.
  - **Moderate Risk ($35\% - 65\%$)**: Enhanced follow-up, telephone call within 72 hours, medication reconciliation.
  - **High Risk ($> 65\%$)**: Multidisciplinary transitional intervention and dedicated case management.
- **Contextual Risk Attributions**: Identifies key patient-specific flags such as prior inpatient hospitalizations ($\ge 2$), extended length of stay ($\ge 7$ days), polypharmacy ($\ge 20$ medications), and emergency admission route.

### 2.5 `services/analytics_service.py`
- **Responsibility**: Visual analytics and exploratory cohort distributions.
- **Backend**: Headless `matplotlib` (`Agg`) and `seaborn` to ensure reliable rendering in Streamlit and testing environments without GUI server dependencies.

---

## 3. Technology Stack Justification

| Technology | Role | Justification |
| :--- | :--- | :--- |
| **Python 3.14** | Core Language | Robust, cross-platform ecosystem for data science. |
| **Pandas & NumPy** | Data Manipulation | High-performance vectorized operations on structured clinical tables. |
| **Scikit-learn** | Machine Learning | CPU-efficient, stable, industry-standard pipelines with zero GPU requirements. |
| **Streamlit** | Web Interface | Rapid interactive dashboarding directly in Python without separate web servers. |
| **Matplotlib & Seaborn** | Visualization | Lightweight, customizable clinical charts rendered in headless mode. |
| **Joblib** | Serialization | Efficient serialization of Scikit-learn pipelines with embedded transformers. |
| **Pytest** | QA & Automated Testing | Standard automated unit and integration test runner. |

# SLAM Review-2 Academic Presentation & Live Demonstration Guide

**Project Title:** Smart Healthcare Analytics and Patient Readmission Prediction System
**College:** Institute of Aeronautical Engineering (IARE)
**Students:** A. Akshith (Roll No: `2125951A05`), K. Anandh (Roll No: `2125951A0527`)
**Project Guide:** Ms. Y Sujana (Assistant Professor, CSE)
**Sustainable Development Goals:** SDG 3 (Good Health and Well-Being), SDG 9 (Industry, Innovation and Infrastructure)

---

## 1. 3-Minute Presentation Walkthrough Script

### Slide 1–3: Title, Abstract & Introduction
> *"Good morning respected Guide and Evaluators. We are Akshith and Anandh presenting our SLAM Review-2 project: 'Smart Healthcare Analytics and Patient Readmission Prediction System'.*
> *Hospital readmissions within 30 days of discharge place severe financial and bed-capacity strain on healthcare institutions. Our project provides an intelligent, data-driven prototype that predicts whether an admitted patient has an elevated risk of readmission based on historical encounter metrics, clinical procedures, and medication records."*

### Slide 4–5: Problem Identification & Literature Review
> *"Manual risk triaging is time-consuming and often misses complex patterns across multivariate clinical records. Based on literature reviews published between 2018 and 2025 in journals such as CMPB, BMJ, and Computers in Biology and Medicine, machine-learning classifiers like Random Forest and Logistic Regression are proven, effective tools for risk stratification in hospital settings."*

### Slide 9–12: Proposed System & Architecture Workflow
> *"Our architecture follows a 6-stage pipeline:*
> **[Dataset Ingestion]** ➔ **[Data Preprocessing]** ➔ **[Feature Selection]** ➔ **[ML Pipeline Training]** ➔ **[Prediction Engine]** ➔ **[Streamlit UI]**
> *To prevent data leakage, we perform an 80/20 stratified split before fitting our preprocessor, ensuring that median imputation and one-hot encoding are fit solely on training data."*

### Slide 13–15: Live Demonstration & Expected Outcomes
> *"Now let us demonstrate the working software prototype running locally on our laptop..."*

---

## 2. Live Prototype Demonstration Steps (During Review)

1. **Launch the Application**:
   ```powershell
   streamlit run app.py
   ```
2. **Tab 1 — Executive Overview**:
   - Point out the total cohort records ($5,000$), the baseline readmission rate ($51.2\%$), and the $18$ clinical attributes.
   - Explain the prominent data provenance indicator (**SYNTHETIC — FOR SOFTWARE TESTING ONLY** or real dataset).
3. **Tab 2 — Patient Risk Predictor (Interactive Demo)**:
   - Walk through the clinical input form:
     - Set *Prior Inpatient Admissions* to `3`.
     - Set *Length of Stay* to `8` days.
     - Set *Medications Prescribed* to `22`.
     - Set *Admission Urgency* to `Emergency`.
   - Click **"Assess Patient Readmission Risk"**.
   - Show the evaluator the resulting **Risk Stratification** (`High Risk`), the **Probability Metric**, the **Recommended Post-Discharge Action Protocol**, and the **Primary Contributing Risk Factors**.
4. **Tab 3 — Exploratory Analytics**:
   - Show the evaluator the **Length of Stay vs. Readmission** boxplot and the **Prior Inpatient Admissions** distribution.
5. **Tab 4 — Model Performance**:
   - Display the actual computed metrics:
     - **Accuracy**: $59.90\%$
     - **ROC-AUC**: $0.6274$
     - **Recall**: $0.5918$
   - Show the **Confusion Matrix** heatmap and **Top 10 Feature Importances**.
   - Demonstrate the **Retrain Model** functionality.
6. **Tab 5 — Methodology & SDGs**:
   - Emphasize alignment with **SDG 3** (Target 3.8: quality universal healthcare) and **SDG 9** (Target 9.5: scientific innovation).
   - Point to the academic disclaimer at the bottom of every page.

# SLAM Review-2 Academic Presentation & Live Demonstration Guide

**Project Title:** Smart Healthcare Analytics and Patient Readmission Prediction System  
**College:** Institute of Aeronautical Engineering (IARE), Hyderabad  
**Department:** Computer Science and Engineering  
**Students:**  
- A. Akshith (Roll No: `2125951A05`)  
- K. Anandh (Roll No: `2125951A0527`)  
**Project Guide:** Ms. Y Sujana (Assistant Professor, CSE)  
**Sustainable Development Goals:** SDG 3 (Good Health and Well-Being), SDG 9 (Industry, Innovation and Infrastructure)  
**Source Presentation:** `slam project(2).pptx`  

---

## 1. 3-Minute Presentation Walkthrough Script

### Slide 1–3: Title, Abstract & Introduction
> *"Good morning respected Guide and Evaluators. We are Akshith and Anandh presenting our SLAM Review-2 project: 'Smart Healthcare Analytics and Patient Readmission Prediction System'.*  
> *Hospital readmissions within 30 days of discharge place severe financial and bed-capacity strain on healthcare institutions. Our project delivers a data-driven system that analyzes historical encounter records, preprocesses multivariate clinical features without data leakage, trains machine-learning models, and provides an interactive clinical triage queue with persistent audit logging."*

### Slide 4–5: Problem Identification & Literature Review
> *"Manual chart review is labor-intensive and prone to missing complex multivariate risk indicators. Our system addresses data quality challenges like missing values and categorical encoding while providing hospital administrators with real-time risk triaging and surveillance dashboards."*

### Slide 9–12: Proposed System & Architecture Workflow
> *"Our architecture implements the 6-stage pipeline from Slide 12:*  
> **[Patient Dataset]** ➔ **[Data Preprocessing]** ➔ **[Feature Selection]** ➔ **[ML Training]** ➔ **[Prediction Engine]** ➔ **[Triage Queue & Dashboard]**  
> *To eliminate data leakage, we perform an 80/20 stratified split before fitting our preprocessor, ensuring that median imputation and one-hot encoding are fit solely on training encounters."*

### Slide 13–15: Live Demonstration & Expected Outcomes
> *"Now let us demonstrate the working software prototype running locally on our laptop..."*

---

## 2. Live Prototype Demonstration Steps (During Review)

1. **Launch the Application**:
   ```powershell
   streamlit run app.py
   ```
2. **Page 1 — Hospital Overview**:
   - Point out the 4 KPI cards (Filtered Encounters, Readmission Rate, Median LOS, Prior Inpatient Visits).
   - Demonstrate the interactive cohort filters (by diagnosis, age, admission urgency, readmission label) dynamically recalculating the KPI metrics and plots.
   - Point out the active dataset provenance badge (**SYNTHETIC — FOR SOFTWARE TESTING ONLY**).
   - Click **"Open Patient Risk Review Queue ➔"** to transition seamlessly to Page 5.

3. **Page 2 — Readmission Analytics**:
   - Showcase the **Subgroup Risk Explorer**: Select *Primary ICD Diagnosis* or *Age Bracket* to display subgroup sample sizes ($N$), readmission counts, and rates compared to the cohort average ($51.2\%$).
   - Show the multivariate factor plots (Prior Inpatient Visits, Length of Stay, Demographics).
   - Highlight the analytical disclaimer: *Statistical associations reflect retrospective correlation and do not establish clinical causation*.

4. **Page 3 — Data Quality & Preprocessing**:
   - Show the ingestion audit: 5,000 records, 0 duplicates, clean schema validation.
   - Walk through the missing value analysis and the leakage-free `ColumnTransformer` structure (median for numeric, mode for categorical, non-predictive `patient_id` removed).
   - Point out the **Feature Association Summary** ranking table and plot.

5. **Page 4 — Model Performance**:
   - Display the actual computed metrics on the 20% holdout test partition ($1,000$ unseen encounters):
     - **Accuracy**: $58.40\%$
     - **Precision**: $59.34\%$
     - **Recall (Sensitivity)**: $59.57\%$
     - **F1-Score**: $59.45\%$
     - **ROC-AUC**: $0.6272$
   - Show the holdout **Confusion Matrix** heatmap, the **ROC Curve**, and the **Top 10 Feature Importances**.
   - Show the **Side-by-Side Model Comparison Table** evaluating Random Forest vs. Logistic Regression on the exact same holdout split.
   - Demonstrate the **Model Retraining Engine**.

6. **Page 5 — Patient Risk Review**:
   - Demonstrate the **Clinical Review Queue**: Search by encounter ID or diagnosis, filter by review status, and adjust the **Priority Score Threshold** slider.
   - Select an encounter from the table: Show how its attributes compare to cohort percentiles (e.g., LOS vs cohort median of 4 days).
   - Enter clinician notes, select status (*Requires Follow-up* or *Reviewed - Standard*), and click **"Persist Review Decision"** to demonstrate persistent SQLite database storage.
   - Switch to the **New Patient Intake Evaluation** tab: Input a profile (e.g. 8 days LOS, 3 inpatient visits), click evaluate, view the statistical risk tier (`Elevated Relative Risk`), and show the instant enrollment into the review queue database!

---

## 3. Anticipated Viva & Evaluator Questions

**Q1: How does your system prevent data leakage?**  
*Answer:* We partition the cohort into 80% training and 20% testing splits *prior* to fitting any transformations. The Scikit-learn `ColumnTransformer` (median numeric imputation, standard scaling, mode categorical imputation, and one-hot encoding) is fitted exclusively on the 80% training partition ($N=4,000$). The test partition ($N=1,000$) is evaluated strictly using the fitted pipeline without re-estimating parameters. Non-predictive identification codes (`patient_id`) are stripped before fitting.

**Q2: Why are both Random Forest and Logistic Regression provided?**  
*Answer:* Random Forest ensembles 100 decision trees to capture non-linear feature interactions between hospitalization length, diagnoses, and medication counts without requiring GPU acceleration. Logistic Regression serves as a linear baseline. Page 4 provides a side-by-side benchmark comparing accuracy, precision, recall, and ROC-AUC on the exact same holdout split.

**Q3: How are patient review statuses and notes persisted?**  
*Answer:* We implemented an ACID-compliant local SQLite database (`data/review_store.db`) via `ReviewService`. Streamlit session state is ephemeral across browser reloads, whereas our SQLite database reliably persists clinician triage notes, review statuses, and newly scored intake records across server restarts.

**Q4: How does this project map to the Sustainable Development Goals?**  
*Answer:* It maps to **SDG 3** (Good Health and Well-Being, Target 3.8) by supporting proactive care coordination and post-discharge planning, and to **SDG 9** (Industry, Innovation and Infrastructure, Target 9.5) by demonstrating an efficient, open-source, CPU-friendly data science architecture that runs on standard institutional laptop hardware.

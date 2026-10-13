"""
Review Service: SQLite-backed Encounter Review Queue and Audit Trail.
Manages persistent review statuses, reviewer notes, queue prioritization,
and newly scored patient encounters across Streamlit reruns and restarts.
"""
import sqlite3
from pathlib import Path
from datetime import datetime, timezone
from contextlib import contextmanager
import pandas as pd

import config

class ReviewService:
    """Manages the persistent human-in-the-loop clinical review queue via SQLite."""

    DB_PATH: Path = config.DATA_DIR / "review_store.db"

    @classmethod
    @contextmanager
    def get_connection(cls):
        """Context manager that yields a connection and guarantees closing it on exit."""
        cls.DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(cls.DB_PATH))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    @classmethod
    def init_database(cls):
        """Initializes the database schema if not already present."""
        with cls.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS encounter_reviews (
                    encounter_id TEXT PRIMARY KEY,
                    patient_id TEXT,
                    age_group TEXT,
                    gender TEXT,
                    admission_type TEXT,
                    admission_source TEXT,
                    primary_diagnosis TEXT,
                    time_in_hospital INTEGER,
                    num_lab_procedures INTEGER,
                    num_procedures INTEGER,
                    num_medications INTEGER,
                    number_outpatient INTEGER,
                    number_emergency INTEGER,
                    number_inpatient INTEGER,
                    number_diagnoses INTEGER,
                    max_glu_serum TEXT,
                    a1c_result TEXT,
                    insulin TEXT,
                    change_medication TEXT,
                    diabetes_med TEXT,
                    actual_readmitted INTEGER,
                    predicted_class INTEGER,
                    model_score REAL,
                    review_status TEXT NOT NULL DEFAULT 'Pending Review',
                    reviewer_notes TEXT DEFAULT '',
                    reviewed_by TEXT DEFAULT '',
                    last_reviewed_at TEXT,
                    is_custom_entry INTEGER DEFAULT 0,
                    created_at TEXT NOT NULL
                )
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_review_status 
                ON encounter_reviews (review_status)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_model_score 
                ON encounter_reviews (model_score)
            """)
            conn.commit()

    @classmethod
    def sync_cohort_from_dataset(cls, df: pd.DataFrame, pipeline=None, max_records: int = 500):
        """
        Populates the review database from the cohort dataset if the queue is empty or has new records.
        Uses INSERT OR IGNORE so existing review statuses and notes are NEVER overwritten.
        """
        cls.init_database()
        if df.empty:
            return 0

        # Load pipeline if not provided
        if pipeline is None:
            from services.prediction_service import PredictionService
            try:
                pipeline = PredictionService.get_pipeline()
            except Exception:
                return 0

        sample_df = df.head(max_records).copy()
        feature_cols = config.NUMERIC_FEATURES + config.CATEGORICAL_FEATURES

        # Generate predictions using the fitted pipeline
        try:
            pred_classes = pipeline.predict(sample_df[feature_cols])
            model_scores = pipeline.predict_proba(sample_df[feature_cols])[:, 1]
        except Exception:
            return 0

        now_str = datetime.now(timezone.utc).isoformat()
        records_to_insert = []

        for idx, row in sample_df.iterrows():
            enc_id = str(row.get("patient_id", f"ENC_{idx:05d}"))
            actual_y = int(row.get(config.TARGET_COLUMN, 0)) if config.TARGET_COLUMN in row else None
            pred_c = int(pred_classes[idx])
            m_score = round(float(model_scores[idx]), 4)

            records_to_insert.append((
                enc_id,
                enc_id,
                str(row.get("age_group", "[60-80)")),
                str(row.get("gender", "Female")),
                str(row.get("admission_type", "Emergency")),
                str(row.get("admission_source", "Emergency Room")),
                str(row.get("primary_diagnosis", "Circulatory")),
                int(row.get("time_in_hospital", 4)),
                int(row.get("num_lab_procedures", 40)),
                int(row.get("num_procedures", 1)),
                int(row.get("num_medications", 15)),
                int(row.get("number_outpatient", 0)),
                int(row.get("number_emergency", 0)),
                int(row.get("number_inpatient", 0)),
                int(row.get("number_diagnoses", 7)),
                str(row.get("max_glu_serum", "None")),
                str(row.get("a1c_result", "None")),
                str(row.get("insulin", "No")),
                str(row.get("change_medication", "No")),
                str(row.get("diabetes_med", "Yes")),
                actual_y,
                pred_c,
                m_score,
                "Pending Review",
                "",
                "",
                None,
                0,
                now_str
            ))

        with cls.get_connection() as conn:
            cursor = conn.cursor()
            cursor.executemany("""
                INSERT OR IGNORE INTO encounter_reviews (
                    encounter_id, patient_id, age_group, gender, admission_type, admission_source,
                    primary_diagnosis, time_in_hospital, num_lab_procedures, num_procedures,
                    num_medications, number_outpatient, number_emergency, number_inpatient,
                    number_diagnoses, max_glu_serum, a1c_result, insulin, change_medication,
                    diabetes_med, actual_readmitted, predicted_class, model_score,
                    review_status, reviewer_notes, reviewed_by, last_reviewed_at,
                    is_custom_entry, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, records_to_insert)
            conn.commit()

        return len(records_to_insert)

    @classmethod
    def get_queue_records(
        cls,
        search_query: Optional[str] = None,
        status_filter: Optional[str] = None,
        threshold: float = 0.50,
        priority_filter: Optional[str] = None,
        sort_by: str = "model_score",
        ascending: bool = False,
        limit: int = 150
    ) -> pd.DataFrame:
        """Queries review queue records with dynamic search, filtering, and sorting."""
        cls.init_database()
        query = "SELECT * FROM encounter_reviews WHERE 1=1"
        params = []

        if search_query:
            query += " AND (encounter_id LIKE ? OR primary_diagnosis LIKE ? OR age_group LIKE ?)"
            term = f"%{search_query.strip()}%"
            params.extend([term, term, term])

        if status_filter and status_filter != "All Statuses":
            query += " AND review_status = ?"
            params.append(status_filter)

        if priority_filter == "High Priority (>= Threshold)":
            query += " AND model_score >= ?"
            params.append(threshold)
        elif priority_filter == "Standard Priority (< Threshold)":
            query += " AND model_score < ?"
            params.append(threshold)

        # Sorting validation
        allowed_sorts = {
            "model_score": "model_score",
            "time_in_hospital": "time_in_hospital",
            "number_inpatient": "number_inpatient",
            "num_medications": "num_medications",
            "encounter_id": "encounter_id",
            "created_at": "created_at",
            "last_reviewed_at": "last_reviewed_at"
        }
        sort_col = allowed_sorts.get(sort_by, "model_score")
        direction = "ASC" if ascending else "DESC"
        query += f" ORDER BY {sort_col} {direction} LIMIT ?"
        params.append(limit)

        with cls.get_connection() as conn:
            df = pd.read_sql_query(query, conn, params=params)

        if not df.empty:
            df["review_priority"] = df["model_score"].apply(
                lambda s: "High Priority" if s >= threshold else "Standard"
            )
        return df

    @classmethod
    def update_review(
        cls,
        encounter_id: str,
        new_status: str,
        notes: str = "",
        reviewer: str = "Clinical Reviewer"
    ) -> bool:
        """Updates review status and audit notes for an encounter."""
        cls.init_database()
        now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

        with cls.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE encounter_reviews
                SET review_status = ?,
                    reviewer_notes = ?,
                    reviewed_by = ?,
                    last_reviewed_at = ?
                WHERE encounter_id = ?
            """, (new_status, notes.strip(), reviewer.strip(), now_str, encounter_id))
            conn.commit()
            return cursor.rowcount > 0

    @classmethod
    def get_queue_statistics(cls, threshold: float = 0.50) -> Dict[str, Any]:
        """Calculates queue summary counts for dashboard KPI cards."""
        cls.init_database()
        with cls.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM encounter_reviews")
            total = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM encounter_reviews WHERE model_score >= ?", (threshold,))
            above_threshold = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM encounter_reviews WHERE review_status = 'Pending Review'")
            pending = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM encounter_reviews WHERE review_status = 'Reviewed - Standard'")
            reviewed = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM encounter_reviews WHERE review_status = 'Requires Follow-up'")
            followup = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM encounter_reviews WHERE review_status = 'Escalated to Physician'")
            escalated = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM encounter_reviews WHERE is_custom_entry = 1")
            custom_count = cursor.fetchone()[0]

        return {
            "total_encounters": total,
            "above_threshold": above_threshold,
            "pending_review": pending,
            "reviewed_standard": reviewed,
            "requires_followup": followup,
            "escalated": escalated,
            "custom_entries": custom_count,
            "threshold_used": threshold
        }

    @classmethod
    def insert_new_encounter(
        cls,
        patient_data: Dict[str, Any],
        model_score: float,
        predicted_class: int,
        review_status: str = "Pending Review",
        notes: str = "Newly scored encounter via interactive assessment."
    ) -> str:
        """Persists an interactively scored patient encounter into the review database."""
        cls.init_database()
        now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        
        with cls.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM encounter_reviews WHERE is_custom_entry = 1")
            seq = cursor.fetchone()[0] + 1
            enc_id = f"CLIN_REC_{seq:04d}"

            cursor.execute("""
                INSERT INTO encounter_reviews (
                    encounter_id, patient_id, age_group, gender, admission_type, admission_source,
                    primary_diagnosis, time_in_hospital, num_lab_procedures, num_procedures,
                    num_medications, number_outpatient, number_emergency, number_inpatient,
                    number_diagnoses, max_glu_serum, a1c_result, insulin, change_medication,
                    diabetes_med, actual_readmitted, predicted_class, model_score,
                    review_status, reviewer_notes, reviewed_by, last_reviewed_at,
                    is_custom_entry, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?)
            """, (
                enc_id, enc_id,
                str(patient_data.get("age_group", "[60-80)")),
                str(patient_data.get("gender", "Female")),
                str(patient_data.get("admission_type", "Emergency")),
                str(patient_data.get("admission_source", "Emergency Room")),
                str(patient_data.get("primary_diagnosis", "Circulatory")),
                int(patient_data.get("time_in_hospital", 4)),
                int(patient_data.get("num_lab_procedures", 40)),
                int(patient_data.get("num_procedures", 1)),
                int(patient_data.get("num_medications", 15)),
                int(patient_data.get("number_outpatient", 0)),
                int(patient_data.get("number_emergency", 0)),
                int(patient_data.get("number_inpatient", 0)),
                int(patient_data.get("number_diagnoses", 7)),
                str(patient_data.get("max_glu_serum", "None")),
                str(patient_data.get("a1c_result", "None")),
                str(patient_data.get("insulin", "No")),
                str(patient_data.get("change_medication", "No")),
                str(patient_data.get("diabetes_med", "Yes")),
                None,
                int(predicted_class),
                round(float(model_score), 4),
                review_status,
                notes,
                "Intake Clinician",
                now_str,
                now_str
            ))
            conn.commit()
            return enc_id

"""
Unit tests for ReviewService: SQLite persistence, cohort synchronization,
status updates, queue filtering, and audit logging.
"""
import pytest
import sqlite3
import pandas as pd
from pathlib import Path

import config
from services.data_service import DataService
from services.review_service import ReviewService

def test_init_database():
    ReviewService.init_database()
    assert ReviewService.DB_PATH.exists()

    with ReviewService.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='encounter_reviews'")
        table = cursor.fetchone()
        assert table is not None

def test_sync_cohort_and_query():
    df, _ = DataService.load_data()
    # Sync small subset
    count = ReviewService.sync_cohort_from_dataset(df.head(20), max_records=20)
    assert count >= 0

    records_df = ReviewService.get_queue_records(limit=20)
    assert isinstance(records_df, pd.DataFrame)
    assert not records_df.empty
    assert "encounter_id" in records_df.columns
    assert "review_status" in records_df.columns
    assert "model_score" in records_df.columns

def test_update_review():
    records_df = ReviewService.get_queue_records(limit=1)
    assert not records_df.empty
    enc_id = records_df.iloc[0]["encounter_id"]

    success = ReviewService.update_review(
        encounter_id=enc_id,
        new_status="Requires Follow-up",
        notes="High risk profile flagged for outpatient follow-up call.",
        reviewer="Test Reviewer"
    )
    assert success is True

    # Verify update persisted
    updated_df = ReviewService.get_queue_records(search_query=enc_id)
    assert not updated_df.empty
    match = updated_df[updated_df["encounter_id"] == enc_id].iloc[0]
    assert match["review_status"] == "Requires Follow-up"
    assert "outpatient follow-up call" in match["reviewer_notes"]
    assert match["reviewed_by"] == "Test Reviewer"

def test_queue_filtering_and_sorting():
    # Test status filter
    pending_df = ReviewService.get_queue_records(status_filter="Pending Review")
    assert isinstance(pending_df, pd.DataFrame)

    # Test search
    searched_df = ReviewService.get_queue_records(search_query="Circulatory")
    assert isinstance(searched_df, pd.DataFrame)

    # Test sorting
    sorted_df = ReviewService.get_queue_records(sort_by="model_score", ascending=False)
    if len(sorted_df) > 1:
        assert sorted_df.iloc[0]["model_score"] >= sorted_df.iloc[-1]["model_score"]

def test_queue_statistics():
    stats = ReviewService.get_queue_statistics(threshold=0.50)
    assert isinstance(stats, dict)
    assert "total_encounters" in stats
    assert "pending_review" in stats
    assert "above_threshold" in stats
    assert stats["total_encounters"] >= 0

def test_insert_new_encounter():
    new_patient = {
        "age_group": "[70-80)",
        "gender": "Male",
        "admission_type": "Emergency",
        "admission_source": "Emergency Room",
        "primary_diagnosis": "Circulatory",
        "time_in_hospital": 6,
        "num_lab_procedures": 55,
        "num_procedures": 2,
        "num_medications": 22,
        "number_outpatient": 1,
        "number_emergency": 1,
        "number_inpatient": 2,
        "number_diagnoses": 9,
        "max_glu_serum": "None",
        "a1c_result": ">8",
        "insulin": "Up",
        "change_medication": "Ch",
        "diabetes_med": "Yes",
    }
    enc_id = ReviewService.insert_new_encounter(
        patient_data=new_patient,
        model_score=0.742,
        predicted_class=1,
        notes="High risk admission profile submitted during test"
    )
    assert enc_id.startswith("CLIN_REC_")

    check_df = ReviewService.get_queue_records(search_query=enc_id)
    assert not check_df.empty
    record = check_df[check_df["encounter_id"] == enc_id].iloc[0]
    assert record["model_score"] == 0.742
    assert record["is_custom_entry"] == 1

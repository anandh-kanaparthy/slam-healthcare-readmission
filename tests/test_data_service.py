"""
Unit tests for DataService: Ingestion, schema validation, and summary stats.
"""
import pytest
import pandas as pd
import config
from services.data_service import DataService

def test_load_data_returns_dataframe_and_provenance():
    df, provenance = DataService.load_data()
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    assert "source" in provenance
    assert "label" in provenance
    assert "is_synthetic" in provenance

def test_validate_schema_valid_data():
    df, _ = DataService.load_data()
    is_valid, msg = DataService.validate_schema(df)
    assert is_valid is True
    assert "passed" in msg.lower()

def test_validate_schema_missing_column():
    df, _ = DataService.load_data()
    df_missing = df.drop(columns=[config.NUMERIC_FEATURES[0]])
    is_valid, msg = DataService.validate_schema(df_missing)
    assert is_valid is False
    assert "missing" in msg.lower()

def test_validate_schema_missing_target():
    df, _ = DataService.load_data()
    df_no_target = df.drop(columns=[config.TARGET_COLUMN])
    is_valid, msg = DataService.validate_schema(df_no_target, require_target=True)
    assert is_valid is False
    assert config.TARGET_COLUMN in msg

def test_validate_schema_empty_dataframe():
    empty_df = pd.DataFrame()
    is_valid, msg = DataService.validate_schema(empty_df)
    assert is_valid is False
    assert "empty" in msg.lower()

def test_dataset_summary_metrics():
    df, _ = DataService.load_data()
    summary = DataService.get_dataset_summary(df)
    assert summary["total_records"] == len(df)
    assert summary["total_features"] == len(config.NUMERIC_FEATURES) + len(config.CATEGORICAL_FEATURES)
    assert 0.0 <= summary["readmitted_pct"] <= 100.0
    assert summary["missing_values_count"] >= 0

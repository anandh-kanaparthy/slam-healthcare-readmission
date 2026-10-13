"""
Unit tests for TrainingService: Model fitting, metrics generation, and artifact saving.
"""
import pytest
import json
from pathlib import Path
from sklearn.pipeline import Pipeline

import config
from services.data_service import DataService
from services.preprocessing_service import PreprocessingService
from services.training_service import TrainingService

def test_model_training_and_artifacts():
    df, _ = DataService.load_data()
    # Use small subset for rapid test execution
    df_sample = df.sample(n=200, random_state=42)
    X_train, X_test, y_train, y_test = PreprocessingService.split_data(df_sample, test_size=0.25)

    pipeline, metrics = TrainingService.train_and_evaluate(
        X_train, X_test, y_train, y_test, model_type="Logistic Regression"
    )

    assert isinstance(pipeline, Pipeline)
    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1_score" in metrics
    assert "roc_auc" in metrics
    assert "confusion_matrix" in metrics

    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["precision"] <= 1.0
    assert 0.0 <= metrics["recall"] <= 1.0
    assert 0.0 <= metrics["f1_score"] <= 1.0
    assert 0.0 <= metrics["roc_auc"] <= 1.0

    assert config.MODEL_PATH.exists()
    assert config.METRICS_PATH.exists()

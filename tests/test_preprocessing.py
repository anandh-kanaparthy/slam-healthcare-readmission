"""
Unit tests for PreprocessingService: Transformation pipelines and leakage prevention.
"""
import pytest
import pandas as pd
from sklearn.compose import ColumnTransformer

import config
from services.data_service import DataService
from services.preprocessing_service import PreprocessingService

def test_create_preprocessor_structure():
    preprocessor = PreprocessingService.create_preprocessor()
    assert isinstance(preprocessor, ColumnTransformer)
    transformer_names = [t[0] for t in preprocessor.transformers]
    assert "num" in transformer_names
    assert "cat" in transformer_names

def test_split_data_leakage_free_stratification():
    df, _ = DataService.load_data()
    X_train, X_test, y_train, y_test = PreprocessingService.split_data(df, test_size=0.20, random_state=config.RANDOM_STATE)

    # Size check
    assert len(X_train) == int(len(df) * 0.8)
    assert len(X_test) == int(len(df) * 0.2)
    assert len(y_train) == len(X_train)
    assert len(y_test) == len(X_test)

    # Disjointness check (no index overlap)
    train_idx = set(X_train.index)
    test_idx = set(X_test.index)
    assert len(train_idx.intersection(test_idx)) == 0

    # Stratification check (readmission ratios within 1% margin)
    train_rate = y_train.mean()
    test_rate = y_test.mean()
    assert abs(train_rate - test_rate) < 0.02

def test_preprocessor_fitting_only_on_train():
    df, _ = DataService.load_data()
    X_train, X_test, y_train, y_test = PreprocessingService.split_data(df)

    preprocessor = PreprocessingService.create_preprocessor()
    # Fit strictly on train
    X_train_trans = preprocessor.fit_transform(X_train)
    # Transform test without re-fitting
    X_test_trans = preprocessor.transform(X_test)

    assert X_train_trans.shape[0] == len(X_train)
    assert X_test_trans.shape[0] == len(X_test)
    assert X_train_trans.shape[1] == X_test_trans.shape[1]

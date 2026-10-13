"""
Integration test: Complete end-to-end data ingestion, pipeline training, reloading, and inference flow.
"""
import pytest
import config
from services.data_service import DataService
from services.preprocessing_service import PreprocessingService
from services.training_service import TrainingService
from services.prediction_service import PredictionService

def test_full_pipeline_end_to_end():
    # 1. Ingestion
    df, provenance = DataService.load_data()
    assert len(df) > 0
    is_valid, msg = DataService.validate_schema(df)
    assert is_valid is True

    # 2. Partitioning
    X_train, X_test, y_train, y_test = PreprocessingService.split_data(df, test_size=0.20)
    assert len(X_train) + len(X_test) == len(df)

    # 3. Training & Evaluation
    pipeline, metrics = TrainingService.train_and_evaluate(
        X_train, X_test, y_train, y_test, model_type="Random Forest"
    )
    assert metrics["accuracy"] > 0.50

    # 4. Persistence Check
    assert config.MODEL_PATH.exists()

    # 5. Pipeline Reload and Inference
    PredictionService.invalidate_cache()
    test_patient = {
        "time_in_hospital": 5,
        "num_lab_procedures": 50,
        "num_procedures": 2,
        "num_medications": 18,
        "number_outpatient": 0,
        "number_emergency": 1,
        "number_inpatient": 1,
        "number_diagnoses": 6,
        "age_group": "[60-80)",
        "gender": "Male",
        "admission_type": "Emergency",
        "admission_source": "Emergency Room",
        "primary_diagnosis": "Diabetes",
        "max_glu_serum": ">200",
        "a1c_result": ">7",
        "insulin": "Steady",
        "change_medication": "Ch",
        "diabetes_med": "Yes",
    }
    pred_result = PredictionService.predict_readmission(test_patient)
    assert pred_result["prediction"] in [0, 1]
    assert 0.0 <= pred_result["readmission_probability"] <= 1.0

"""
Unit tests for PredictionService: Validation, inference, and error handling.
"""
import pytest
import config
from services.prediction_service import PredictionService

@pytest.fixture
def valid_patient():
    return {
        "time_in_hospital": 4,
        "num_lab_procedures": 45,
        "num_procedures": 1,
        "num_medications": 16,
        "number_outpatient": 0,
        "number_emergency": 0,
        "number_inpatient": 2,
        "number_diagnoses": 8,
        "age_group": "[60-80)",
        "gender": "Female",
        "admission_type": "Emergency",
        "admission_source": "Emergency Room",
        "primary_diagnosis": "Circulatory",
        "max_glu_serum": "None",
        "a1c_result": ">8",
        "insulin": "Up",
        "change_medication": "Ch",
        "diabetes_med": "Yes",
    }

def test_prediction_output_structure(valid_patient):
    PredictionService.invalidate_cache()
    result = PredictionService.predict_readmission(valid_patient)

    assert "prediction" in result
    assert result["prediction"] in [0, 1]
    assert "readmission_probability" in result
    assert 0.0 <= result["readmission_probability"] <= 1.0
    assert "risk_tier" in result
    assert result["risk_tier"] in ["Low Risk", "Moderate Risk", "High Risk"]
    assert "risk_color" in result
    assert "recommendation" in result
    assert isinstance(result["contributing_factors"], list)
    assert len(result["contributing_factors"]) > 0

def test_missing_feature_validation():
    invalid_patient = {"time_in_hospital": 3}
    errors = PredictionService.validate_patient_input(invalid_patient)
    assert len(errors) > 0
    with pytest.raises(ValueError):
        PredictionService.predict_readmission(invalid_patient)

def test_negative_numeric_feature_validation(valid_patient):
    corrupt_patient = valid_patient.copy()
    corrupt_patient["time_in_hospital"] = -5
    errors = PredictionService.validate_patient_input(corrupt_patient)
    assert any("negative" in e.lower() for e in errors)
    with pytest.raises(ValueError):
        PredictionService.predict_readmission(corrupt_patient)

def test_invalid_category_validation(valid_patient):
    corrupt_patient = valid_patient.copy()
    corrupt_patient["admission_type"] = "INVALID_ADMISSION"
    errors = PredictionService.validate_patient_input(corrupt_patient)
    assert any("invalid option" in e.lower() for e in errors)
    with pytest.raises(ValueError):
        PredictionService.predict_readmission(corrupt_patient)

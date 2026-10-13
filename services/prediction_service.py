"""
Prediction Service: Real-time inference, input validation, risk stratification, and factor attribution.
"""
from pathlib import Path
from typing import Dict, Any, List
import joblib
import pandas as pd

import config

class PredictionService:
    """Handles patient risk inference using the fitted machine learning pipeline."""

    _cached_pipeline = None

    @classmethod
    def get_pipeline(cls):
        """Loads and caches the fitted Scikit-learn pipeline."""
        if cls._cached_pipeline is None:
            if not config.MODEL_PATH.exists():
                raise FileNotFoundError(
                    f"Fitted model artifact not found at {config.MODEL_PATH}. "
                    f"Please run 'python train_model.py' or train a model via the dashboard first."
                )
            cls._cached_pipeline = joblib.load(config.MODEL_PATH)
        return cls._cached_pipeline

    @classmethod
    def invalidate_cache(cls):
        """Clears the cached pipeline when a new model is retrained."""
        cls._cached_pipeline = None

    @classmethod
    def validate_patient_input(cls, patient_data: Dict[str, Any]) -> List[str]:
        """Validates incoming patient feature values and ranges."""
        errors = []
        if not isinstance(patient_data, dict):
            return ["Patient data must be provided as a dictionary."]

        # Check for missing numeric features
        for num_col in config.NUMERIC_FEATURES:
            if num_col not in patient_data:
                errors.append(f"Missing required numeric feature: '{num_col}'.")
            else:
                val = patient_data[num_col]
                try:
                    num_val = float(val)
                    if num_val < 0:
                        errors.append(f"Feature '{num_col}' cannot be negative (got {val}).")
                except (ValueError, TypeError):
                    errors.append(f"Feature '{num_col}' must be a valid number (got {val}).")

        # Check for missing categorical features
        for cat_col in config.CATEGORICAL_FEATURES:
            if cat_col not in patient_data:
                errors.append(f"Missing required categorical feature: '{cat_col}'.")
            else:
                val = str(patient_data[cat_col])
                allowed = config.CATEGORICAL_CHOICES.get(cat_col, [])
                if allowed and val not in allowed:
                    errors.append(f"Invalid option '{val}' for '{cat_col}'. Allowed: {allowed}")

        return errors

    @classmethod
    def predict_readmission(cls, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes inference for a single patient record.
        Returns prediction, probability, risk tier, and clinical driver factors.
        """
        # Validate inputs
        validation_errors = cls.validate_patient_input(patient_data)
        if validation_errors:
            raise ValueError(f"Input validation failed: {'; '.join(validation_errors)}")

        pipeline = cls.get_pipeline()
        df_input = pd.DataFrame([patient_data])

        # Generate predictions
        pred_class = int(pipeline.predict(df_input)[0])
        probabilities = pipeline.predict_proba(df_input)[0]
        prob_readmitted = float(probabilities[1])

        # Stratified Risk Tier Classification
        if prob_readmitted < 0.35:
            risk_tier = "Low Risk"
            risk_color = "#28a745"
            recommendation = (
                "Standard post-discharge pathway. Schedule routine primary-care follow-up within 14 days. "
                "Provide written discharge instructions and standard medication guide."
            )
        elif prob_readmitted < 0.65:
            risk_tier = "Moderate Risk"
            risk_color = "#e67e22"
            recommendation = (
                "Enhanced transition-of-care protocol recommended. Conduct clinical telephone follow-up "
                "within 48 to 72 hours, perform formal medication reconciliation, and assess home support."
            )
        else:
            risk_tier = "High Risk"
            risk_color = "#d9534f"
            recommendation = (
                "Urgent multidisciplinary discharge intervention advised. Assign dedicated nurse case manager, "
                "schedule priority outpatient appointment within 7 days, and evaluate eligibility for transitional care monitoring."
            )

        # Contextual Risk Drivers based on patient attributes
        drivers = []
        if float(patient_data.get("number_inpatient", 0)) >= 2:
            drivers.append("History of multiple prior inpatient hospitalizations (>= 2 visits in preceding year)")
        if float(patient_data.get("time_in_hospital", 0)) >= 7:
            drivers.append("Extended length of acute stay (>= 7 days indicates elevated clinical complexity)")
        if float(patient_data.get("num_medications", 0)) >= 20:
            drivers.append("High medication regimen burden / Polypharmacy (>= 20 active medications)")
        if patient_data.get("admission_type") == "Emergency":
            drivers.append("Emergency admission pathway indicates acute unscheduled presentation")
        if patient_data.get("insulin") in ["Up", "Down"]:
            drivers.append("Active clinical titration of insulin therapy during hospitalization")
        if patient_data.get("a1c_result") in [">7", ">8"]:
            drivers.append("Elevated glycated hemoglobin (HbA1c > 7) indicating unmanaged glycemic control")
        if float(patient_data.get("number_diagnoses", 0)) >= 9:
            drivers.append("High comorbidity burden (>= 9 recorded clinical diagnoses)")

        return {
            "prediction": pred_class,
            "prediction_label": "Likely Readmission (<30 Days)" if pred_class == 1 else "No Readmission Expected",
            "readmission_probability": round(prob_readmitted, 4),
            "readmission_percentage": round(prob_readmitted * 100, 1),
            "no_readmission_probability": round(float(probabilities[0]), 4),
            "risk_tier": risk_tier,
            "risk_color": risk_color,
            "recommendation": recommendation,
            "contributing_factors": drivers if drivers else ["No elevated acute clinical flags identified for this profile."],
        }

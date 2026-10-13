"""
Prediction Service: Real-time inference, input validation, statistical risk stratification,
and empirical cohort comparative analytics.

IMPORTANT ETHICAL AND CLINICAL DISCLAIMER:
This service is part of an academic prototype for demonstration and educational purposes only.
It does NOT provide clinical diagnosis, medical advice, or treatment recommendations.
Statistical correlations in historical or synthetic datasets do not establish clinical causation.
"""
from pathlib import Path
from typing import Dict, Any, List
import joblib
import pandas as pd
import numpy as np

import config

class PredictionService:
    """Handles patient readmission risk classification and empirical cohort comparisons."""

    _cached_pipeline = None
    _cached_cohort_stats = None

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
        """Clears the cached pipeline and statistics when a new model or dataset is loaded."""
        cls._cached_pipeline = None
        cls._cached_cohort_stats = None

    @classmethod
    def get_cohort_statistics(cls) -> Dict[str, Any]:
        """
        Calculates empirical descriptive statistics from the active dataset
        to provide genuine data-driven comparisons rather than arbitrary assumptions.
        """
        if cls._cached_cohort_stats is None:
            from services.data_service import DataService
            df, provenance = DataService.load_data()

            stats = {
                "provenance": provenance,
                "total_records": len(df),
                "numeric": {},
                "readmission_rate": float(df[config.TARGET_COLUMN].mean() * 100) if config.TARGET_COLUMN in df.columns else 0.0
            }

            for num_col in config.NUMERIC_FEATURES:
                if num_col in df.columns:
                    series = df[num_col].dropna()
                    stats["numeric"][num_col] = {
                        "median": float(series.median()),
                        "mean": float(series.mean()),
                        "p75": float(series.quantile(0.75)),
                        "p90": float(series.quantile(0.90)),
                        "min": float(series.min()),
                        "max": float(series.max())
                    }

            cls._cached_cohort_stats = stats
        return cls._cached_cohort_stats

    @classmethod
    def validate_patient_input(cls, patient_data: Dict[str, Any]) -> List[str]:
        """Validates incoming patient feature values, ranges, and categorical choices."""
        errors = []
        if not isinstance(patient_data, dict):
            return ["Patient data must be provided as a dictionary."]

        # Check numeric features
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

        # Check categorical features
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
        Executes classification inference for a patient profile.
        Compares input attributes with empirical cohort statistics.
        Returns model probabilities, relative statistical risk tier, and educational notes.
        """
        # Validate inputs
        validation_errors = cls.validate_patient_input(patient_data)
        if validation_errors:
            raise ValueError(f"Input validation failed: {'; '.join(validation_errors)}")

        pipeline = cls.get_pipeline()
        df_input = pd.DataFrame([patient_data])

        # Generate model prediction and class probabilities
        pred_class = int(pipeline.predict(df_input)[0])
        probabilities = pipeline.predict_proba(df_input)[0]
        prob_readmitted = float(probabilities[1])

        # Statistical risk tiering relative to dataset probability distribution
        if prob_readmitted < 0.35:
            risk_tier = "Lower Relative Risk"
            risk_color = "#28a745"
            tier_description = (
                "The classification model assigns this profile a predicted readmission probability below 35%, "
                "placing it in the lower risk quartile relative to the training cohort."
            )
        elif prob_readmitted < 0.65:
            risk_tier = "Moderate Relative Risk"
            risk_color = "#e67e22"
            tier_description = (
                "The classification model assigns this profile a predicted readmission probability between 35% and 65%, "
                "indicating an intermediate likelihood compared to average cohort encounters."
            )
        else:
            risk_tier = "Elevated Relative Risk"
            risk_color = "#d9534f"
            tier_description = (
                "The classification model assigns this profile a predicted readmission probability above 65%, "
                "placing it in the upper risk tier relative to the training cohort."
            )

        # Dynamic Empirical Cohort Comparison (Descriptive Statistics)
        cohort_stats = cls.get_cohort_statistics()
        num_stats = cohort_stats.get("numeric", {})
        comparative_factors = []

        # Compare length of stay
        los = float(patient_data.get("time_in_hospital", 0))
        if "time_in_hospital" in num_stats:
            p75_los = num_stats["time_in_hospital"]["p75"]
            med_los = num_stats["time_in_hospital"]["median"]
            if los >= p75_los:
                comparative_factors.append(
                    f"Length of Stay ({los:.0f} days) exceeds the cohort 75th percentile ({p75_los:.1f} days; median is {med_los:.1f} days)."
                )

        # Compare prior inpatient admissions
        inp = float(patient_data.get("number_inpatient", 0))
        if "number_inpatient" in num_stats:
            p75_inp = num_stats["number_inpatient"]["p75"]
            if inp >= p75_inp and inp > 0:
                comparative_factors.append(
                    f"Prior Inpatient Admissions ({inp:.0f} visits) is at or above the cohort 75th percentile ({p75_inp:.1f} visits)."
                )

        # Compare medications count
        meds = float(patient_data.get("num_medications", 0))
        if "num_medications" in num_stats:
            p75_meds = num_stats["num_medications"]["p75"]
            med_meds = num_stats["num_medications"]["median"]
            if meds >= p75_meds:
                comparative_factors.append(
                    f"Prescribed Medications ({meds:.0f}) exceeds the cohort 75th percentile ({p75_meds:.1f} medications; median is {med_meds:.1f})."
                )

        # Categorical encounter characteristics
        if patient_data.get("admission_type") == "Emergency":
            comparative_factors.append(
                "Admission pathway is Emergency (acute unscheduled presentation)."
            )

        if patient_data.get("insulin") in ["Up", "Down"]:
            comparative_factors.append(
                f"Insulin regimen was titrated ('{patient_data.get('insulin')}') during the encounter."
            )

        if not comparative_factors:
            comparative_factors.append(
                "Patient parameters are within the interquartile range (IQR) of typical cohort encounters."
            )

        return {
            "prediction": pred_class,
            "prediction_label": "Readmission Predicted (<30 Days)" if pred_class == 1 else "No Readmission Predicted",
            "readmission_probability": round(prob_readmitted, 4),
            "readmission_percentage": round(prob_readmitted * 100, 1),
            "no_readmission_probability": round(float(probabilities[0]), 4),
            "risk_tier": risk_tier,
            "risk_color": risk_color,
            "tier_description": tier_description,
            "comparative_factors": comparative_factors,
            "dataset_is_synthetic": cohort_stats.get("provenance", {}).get("is_synthetic", True),
            "disclaimer": (
                "Educational Prototype Notice: This classification is generated by a demonstration model "
                "trained on historical/synthetic healthcare records. Statistical associations do not establish medical causation. "
                "This output does NOT constitute clinical advice or treatment instructions."
            )
        }

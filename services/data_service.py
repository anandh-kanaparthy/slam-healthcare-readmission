"""
Data Service: Ingestion, schema validation, profiling, and provenance tracking.
"""
from pathlib import Path
from typing import Dict, Any, Tuple
import pandas as pd

import config
from data.generate_data import generate_synthetic_cohort

class DataService:
    """Service handling dataset ingestion, validation, and statistical summaries."""

    @staticmethod
    def load_data(filepath: str | Path = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Loads the dataset with provenance tracking.
        Prefers real data if present, otherwise falls back to the labeled synthetic cohort.
        """
        target_path = Path(filepath) if filepath else None

        # 1. Specified file path
        if target_path and target_path.exists():
            df = pd.read_csv(target_path)
            is_synthetic = "synthetic" in target_path.name.lower() or "demo" in target_path.name.lower()
            provenance = {
                "source": str(target_path),
                "is_synthetic": is_synthetic,
                "label": "SYNTHETIC — FOR SOFTWARE TESTING ONLY" if is_synthetic else "REAL DATASET"
            }
            return df, provenance

        # 2. Real dataset check
        if config.DATASET_PATH.exists():
            df = pd.read_csv(config.DATASET_PATH)
            provenance = {
                "source": str(config.DATASET_PATH),
                "is_synthetic": False,
                "label": "REAL HOSPITAL COHORT (UCI Diabetes)"
            }
            return df, provenance

        # 3. Synthetic dataset check
        if config.SYNTHETIC_DATASET_PATH.exists():
            df = pd.read_csv(config.SYNTHETIC_DATASET_PATH)
            provenance = {
                "source": str(config.SYNTHETIC_DATASET_PATH),
                "is_synthetic": True,
                "label": "SYNTHETIC — FOR SOFTWARE TESTING ONLY"
            }
            return df, provenance

        # 4. Generate fallback synthetic dataset if neither exists
        df = generate_synthetic_cohort()
        provenance = {
            "source": str(config.SYNTHETIC_DATASET_PATH),
            "is_synthetic": True,
            "label": "SYNTHETIC — FOR SOFTWARE TESTING ONLY"
        }
        return df, provenance

    @staticmethod
    def validate_schema(df: pd.DataFrame, require_target: bool = True) -> Tuple[bool, str]:
        """Validates that a DataFrame contains all required model features and target."""
        if df is None or df.empty:
            return False, "Dataset is empty."

        missing_numeric = [col for col in config.NUMERIC_FEATURES if col not in df.columns]
        missing_categorical = [col for col in config.CATEGORICAL_FEATURES if col not in df.columns]

        if missing_numeric or missing_categorical:
            missing_cols = missing_numeric + missing_categorical
            return False, f"Missing required feature columns: {', '.join(missing_cols)}"

        if require_target and config.TARGET_COLUMN not in df.columns:
            return False, f"Missing target column '{config.TARGET_COLUMN}'."

        return True, "Schema validation passed."

    @staticmethod
    def get_dataset_summary(df: pd.DataFrame) -> Dict[str, Any]:
        """Computes statistical summaries for the dashboard overview."""
        has_target = config.TARGET_COLUMN in df.columns
        readmitted_count = int(df[config.TARGET_COLUMN].sum()) if has_target else 0
        readmitted_pct = float(df[config.TARGET_COLUMN].mean() * 100) if has_target else 0.0

        return {
            "total_records": len(df),
            "total_features": len(config.NUMERIC_FEATURES) + len(config.CATEGORICAL_FEATURES),
            "readmitted_count": readmitted_count,
            "readmitted_pct": round(readmitted_pct, 2),
            "non_readmitted_count": len(df) - readmitted_count if has_target else 0,
            "missing_values_count": int(df.isna().sum().sum()),
            "numeric_feature_count": len(config.NUMERIC_FEATURES),
            "categorical_feature_count": len(config.CATEGORICAL_FEATURES),
        }

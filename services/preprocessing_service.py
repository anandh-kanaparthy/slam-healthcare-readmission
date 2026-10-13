"""
Preprocessing Service: ColumnTransformer pipelines, imputation, encoding, and leakage-free splitting.
"""
from typing import Tuple, List
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

import config

class PreprocessingService:
    """Handles leakage-free data transformations and stratified train/test partitioning."""

    @staticmethod
    def create_preprocessor() -> ColumnTransformer:
        """
        Creates a Scikit-learn ColumnTransformer.
        Numeric: Median Imputation + Z-score Standardization
        Categorical: Mode Imputation + One-Hot Encoding (unknown category tolerant)
        """
        numeric_pipeline = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])

        categorical_pipeline = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_pipeline, config.NUMERIC_FEATURES),
                ("cat", categorical_pipeline, config.CATEGORICAL_FEATURES),
            ],
            remainder="drop"
        )
        return preprocessor

    @staticmethod
    def split_data(
        df: pd.DataFrame,
        test_size: float = None,
        random_state: int = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Performs stratified train/test split.
        CRITICAL SAFEGUARD: Splits BEFORE any fitting to prevent target and test leakage.
        """
        t_size = test_size if test_size is not None else config.TEST_SIZE
        seed = random_state if random_state is not None else config.RANDOM_STATE

        feature_cols = config.NUMERIC_FEATURES + config.CATEGORICAL_FEATURES
        X = df[feature_cols].copy()
        y = df[config.TARGET_COLUMN].copy()

        # Stratified split preserves minority readmission ratio
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=t_size,
            random_state=seed,
            stratify=y
        )
        return X_train, X_test, y_train, y_test

    @staticmethod
    def get_feature_names(fitted_preprocessor: ColumnTransformer) -> List[str]:
        """Extracts transformed feature names after one-hot encoding."""
        feature_names = []
        try:
            num_cols = config.NUMERIC_FEATURES
            cat_cols = fitted_preprocessor.named_transformers_["cat"].named_steps["encoder"].get_feature_names_out(config.CATEGORICAL_FEATURES)
            feature_names = list(num_cols) + list(cat_cols)
        except Exception:
            feature_names = [f"feat_{i}" for i in range(100)]
        return feature_names

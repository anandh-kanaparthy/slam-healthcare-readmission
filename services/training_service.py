"""
Training Service: Model training, evaluation metrics calculation, artifact persistence, and feature importances.
"""
import json
from pathlib import Path
from typing import Dict, Any, Tuple
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

import config
from services.preprocessing_service import PreprocessingService

class TrainingService:
    """Manages machine learning model training, cross-validation, and metrics evaluation."""

    @staticmethod
    def train_and_evaluate(
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        model_type: str = "Random Forest"
    ) -> Tuple[Pipeline, Dict[str, Any]]:
        """
        Trains model pipeline strictly on X_train/y_train and evaluates on X_test/y_test.
        Persists pipeline to models/readmission_pipeline.joblib and metrics to JSON.
        """
        preprocessor = PreprocessingService.create_preprocessor()

        if model_type == "Logistic Regression":
            classifier = LogisticRegression(
                max_iter=1000,
                random_state=config.RANDOM_STATE,
                class_weight="balanced",
                solver="lbfgs"
            )
        else:
            classifier = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                random_state=config.RANDOM_STATE,
                n_jobs=-1,
                class_weight="balanced"
            )

        pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier)
        ])

        # Strictly fit on training set only (prevent leakage)
        pipeline.fit(X_train, y_train)

        # Test set predictions
        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]

        # Calculate metrics
        cm = confusion_matrix(y_test, y_pred).tolist()
        class_rep = classification_report(y_test, y_pred, output_dict=True)

        metrics = {
            "model_type": model_type,
            "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
            "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
            "recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
            "f1_score": round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
            "roc_auc": round(float(roc_auc_score(y_test, y_prob)), 4),
            "confusion_matrix": cm,
            "classification_report": class_rep,
            "train_samples": len(y_train),
            "test_samples": len(y_test),
            "test_readmission_rate": round(float(y_test.mean() * 100), 2),
        }

        # Save artifacts
        config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        config.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

        joblib.dump(pipeline, config.MODEL_PATH)
        with open(config.METRICS_PATH, "w") as f:
            json.dump(metrics, f, indent=2)

        # Save Evaluation Charts
        TrainingService._save_confusion_matrix_plot(cm, config.CONFUSION_MATRIX_PATH)
        TrainingService._save_feature_importance_plot(pipeline, config.FEATURE_IMPORTANCE_PATH)

        return pipeline, metrics

    @staticmethod
    def _save_confusion_matrix_plot(cm: list, output_path: Path):
        """Generates and persists confusion matrix visualization."""
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=["No Readmission", "Readmitted (<30d)"],
            yticklabels=["No Readmission", "Readmitted (<30d)"],
            ax=ax
        )
        ax.set_title("Test Set Confusion Matrix", fontsize=11, fontweight="bold", pad=10)
        ax.set_xlabel("Predicted Label", fontsize=10)
        ax.set_ylabel("True Label", fontsize=10)
        plt.tight_layout()
        fig.savefig(output_path, dpi=150)
        plt.close(fig)

    @staticmethod
    def _save_feature_importance_plot(pipeline: Pipeline, output_path: Path):
        """Extracts and saves top feature importances."""
        try:
            preprocessor = pipeline.named_steps["preprocessor"]
            classifier = pipeline.named_steps["classifier"]
            feature_names = PreprocessingService.get_feature_names(preprocessor)

            if hasattr(classifier, "feature_importances_"):
                importances = classifier.feature_importances_
            elif hasattr(classifier, "coef_"):
                importances = np.abs(classifier.coef_[0])
            else:
                return

            if len(feature_names) == len(importances):
                feat_df = pd.DataFrame({
                    "feature": feature_names,
                    "importance": importances
                }).sort_values("importance", ascending=False).head(10)

                fig, ax = plt.subplots(figsize=(7, 4.5))
                sns.barplot(x="importance", y="feature", data=feat_df, hue="feature", palette="Blues_r", legend=False, ax=ax)
                ax.set_title("Top 10 Clinical Risk Feature Importances", fontsize=11, fontweight="bold", pad=10)
                ax.set_xlabel("Relative Importance Score", fontsize=10)
                ax.set_ylabel("Clinical Feature", fontsize=10)
                plt.tight_layout()
                fig.savefig(output_path, dpi=150)
                plt.close(fig)
        except Exception as e:
            print(f"[WARNING] Could not generate feature importance plot: {e}")

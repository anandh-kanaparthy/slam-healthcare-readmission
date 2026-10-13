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
        model_type: str = "Random Forest",
        save_artifacts: bool = True
    ) -> Tuple[Pipeline, Dict[str, Any]]:
        """
        Trains model pipeline strictly on X_train/y_train and evaluates on X_test/y_test.
        If save_artifacts is True, persists pipeline to models/readmission_pipeline.joblib,
        metrics to JSON, and evaluation plots to artifacts/.
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

        # Save artifacts only when requested (prevents test runs from clobbering production models)
        if save_artifacts:
            config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
            config.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

            joblib.dump(pipeline, config.MODEL_PATH)
            with open(config.METRICS_PATH, "w") as f:
                json.dump(metrics, f, indent=2)

            # Save Evaluation Charts
            TrainingService._save_confusion_matrix_plot(cm, config.CONFUSION_MATRIX_PATH)
            TrainingService._save_feature_importance_plot(pipeline, config.FEATURE_IMPORTANCE_PATH)
            TrainingService._save_roc_curve_plot(y_test, y_prob, metrics["roc_auc"], config.ROC_CURVE_PATH)

        return pipeline, metrics

    @staticmethod
    def compare_models(
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series
    ) -> pd.DataFrame:
        """
        Trains and compares Random Forest and Logistic Regression on the exact same train/test split.
        Does not overwrite disk artifacts.
        Returns a DataFrame summarizing side-by-side performance metrics.
        """
        results = []
        for model_name in ["Random Forest", "Logistic Regression"]:
            _, m = TrainingService.train_and_evaluate(
                X_train, X_test, y_train, y_test,
                model_type=model_name,
                save_artifacts=False
            )
            results.append({
                "Model Architecture": model_name,
                "Accuracy": f"{m['accuracy'] * 100:.2f}%",
                "Precision": f"{m['precision']:.4f}",
                "Recall (Sensitivity)": f"{m['recall']:.4f}",
                "F1 Score": f"{m['f1_score']:.4f}",
                "ROC-AUC": f"{m['roc_auc']:.4f}"
            })
        return pd.DataFrame(results)

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

    @staticmethod
    def _save_roc_curve_plot(y_test, y_prob, roc_auc_val: float, output_path: Path):
        """Generates and persists ROC curve visualization."""
        try:
            from sklearn.metrics import roc_curve
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            fig, ax = plt.subplots(figsize=(5.2, 4.0))
            ax.plot(fpr, tpr, color="#2b5c8f", lw=2, label=f"Model ROC (AUC = {roc_auc_val:.4f})")
            ax.plot([0, 1], [0, 1], color="#94a3b8", lw=1.5, linestyle="--", label="Random (AUC = 0.50)")
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=9)
            ax.set_ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=9)
            ax.set_title("Receiver Operating Characteristic (ROC)", fontsize=11, fontweight="bold", pad=10)
            ax.legend(loc="lower right", fontsize=8.5)
            plt.tight_layout()
            fig.savefig(output_path, dpi=150)
            plt.close(fig)
        except Exception as e:
            print(f"[WARNING] Could not generate ROC curve plot: {e}")


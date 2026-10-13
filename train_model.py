"""
Model Training CLI Script for Smart Healthcare Analytics & Readmission Prediction.
Usage:
    python train_model.py
    python train_model.py --model "Logistic Regression"
    python train_model.py --model "Random Forest"
"""
import sys
import argparse
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent))

import config
from services.data_service import DataService
from services.preprocessing_service import PreprocessingService
from services.training_service import TrainingService

def run_training_pipeline(model_type: str = "Random Forest"):
    print("\n" + "=" * 70)
    print("  SLAM HEALTHCARE READMISSION PREDICTION — MODEL TRAINING PIPELINE")
    print("=" * 70)

    # 1. Dataset Ingestion
    print("\n[Step 1/4] Ingesting patient cohort dataset...")
    df, provenance = DataService.load_data()
    print(f"  * Dataset Provenance: {provenance['label']}")
    print(f"  * Source Path       : {provenance['source']}")
    print(f"  * Total Encounters  : {len(df):,}")

    is_valid, validation_msg = DataService.validate_schema(df)
    if not is_valid:
        print(f"\n[ERROR] Dataset schema validation failed: {validation_msg}")
        sys.exit(1)
    print("  * Schema Validation : PASSED (all required clinical features present)")

    # 2. Stratified Partitioning (No Preprocessing Leakage)
    print("\n[Step 2/4] Splitting dataset into Stratified 80% Train / 20% Test partitions...")
    X_train, X_test, y_train, y_test = PreprocessingService.split_data(df)
    print(f"  * Training Samples  : {len(X_train):,} ({100 - int(config.TEST_SIZE * 100)}%)")
    print(f"  * Test Samples      : {len(X_test):,} ({int(config.TEST_SIZE * 100)}%)")
    print(f"  * Train Readmit Rate: {y_train.mean() * 100:.2f}%")
    print(f"  * Test Readmit Rate : {y_test.mean() * 100:.2f}%")

    # 3. Model Training & Evaluation
    print(f"\n[Step 3/4] Fitting Scikit-learn Pipeline with {model_type}...")
    pipeline, metrics = TrainingService.train_and_evaluate(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        model_type=model_type
    )

    # 4. Reporting & Artifact Persistence
    print("\n[Step 4/4] Model Evaluation and Artifact Verification:")
    print("-" * 70)
    print(f"  Algorithm Model Architecture : {metrics['model_type']}")
    print(f"  Test Set Accuracy            : {metrics['accuracy']:.4f} ({metrics['accuracy'] * 100:.2f}%)")
    print(f"  Test Set Precision           : {metrics['precision']:.4f}")
    print(f"  Test Set Recall (Sensitivity): {metrics['recall']:.4f}")
    print(f"  Test Set F1-Score            : {metrics['f1_score']:.4f}")
    print(f"  Test Set ROC-AUC Score       : {metrics['roc_auc']:.4f}")
    print(f"  Confusion Matrix (Test Set)  : {metrics['confusion_matrix']}")
    print("-" * 70)
    print(f"  [SAVED] Pipeline Artifact    : {config.MODEL_PATH}")
    print(f"  [SAVED] Metrics JSON         : {config.METRICS_PATH}")
    print(f"  [SAVED] Confusion Matrix Plot: {config.CONFUSION_MATRIX_PATH}")
    print(f"  [SAVED] Feature Importance   : {config.FEATURE_IMPORTANCE_PATH}")
    print(f"  [SAVED] ROC Curve Plot       : {config.ROC_CURVE_PATH}")
    print("=" * 70)
    print("Pipeline training completed successfully!\n")
    return pipeline, metrics

def main():
    parser = argparse.ArgumentParser(description="Train hospital readmission classification pipeline.")
    parser.add_argument(
        "--model",
        type=str,
        default="Random Forest",
        choices=["Random Forest", "Logistic Regression"],
        help="Classifier architecture to train (default: 'Random Forest')"
    )
    args = parser.parse_args()
    run_training_pipeline(model_type=args.model)

if __name__ == "__main__":
    main()

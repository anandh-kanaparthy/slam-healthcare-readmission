"""
Analytics Service: Visual healthcare analytics, cohort distribution plotting,
feature association rankings, and model evaluation charts for SLAM Review-2.
"""
from typing import List, Any
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc

# Set clean styling for professional academic presentation
sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams.update({"font.sans-serif": "Arial", "font.size": 9})

class AnalyticsService:
    """Generates clean, CPU-efficient clinical visual analytics for the Streamlit dashboard."""

    @staticmethod
    def plot_readmission_distribution(df: pd.DataFrame):
        """Displays distribution of the binary target variable."""
        fig, ax = plt.subplots(figsize=(6, 3.8))
        counts = df["readmitted"].value_counts().rename({0: "No Readmission", 1: "Readmitted (<30d)"})
        palette = ["#2b5c8f", "#d9534f"]
        sns.barplot(x=counts.index, y=counts.values, hue=counts.index, palette=palette, legend=False, ax=ax)
        ax.set_title("30-Day Readmission Class Distribution", fontsize=11, fontweight="bold", pad=10)
        ax.set_ylabel("Patient Encounters", fontsize=9)
        ax.set_xlabel("Outcome Category", fontsize=9)

        total = len(df)
        for p in ax.patches:
            height = p.get_height()
            pct = (height / total) * 100
            ax.annotate(f"{int(height):,} ({pct:.1f}%)",
                        (p.get_x() + p.get_width() / 2., height),
                        ha="center", va="bottom", fontsize=8.5,
                        xytext=(0, 4), textcoords="offset points")

        plt.tight_layout()
        return fig

    @staticmethod
    def plot_los_vs_readmission(df: pd.DataFrame):
        """Compares Length of Stay (days in hospital) across readmission outcomes."""
        fig, ax = plt.subplots(figsize=(6.0, 3.8))
        sns.boxplot(
            x="readmitted", y="time_in_hospital", data=df,
            palette=["#2563eb", "#dc2626"], ax=ax, width=0.42
        )
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["No Readmission", "Readmitted (<30d)"])
        ax.set_title("Length of Stay vs. Readmission Status", fontsize=11, fontweight="bold", pad=10)
        ax.set_xlabel("Encounter Outcome", fontsize=9)
        ax.set_ylabel("Hospital Stay Duration (Days)", fontsize=9)
        plt.tight_layout()
        return fig

    @staticmethod
    def plot_readmission_by_diagnosis(df: pd.DataFrame):
        """Displays readmission rate across primary diagnosis categories."""
        fig, ax = plt.subplots(figsize=(6.0, 3.8))
        diag_summary = df.groupby("primary_diagnosis")["readmitted"].mean().reset_index()
        diag_summary["rate_pct"] = diag_summary["readmitted"] * 100
        diag_summary = diag_summary.sort_values("rate_pct", ascending=True)

        sns.barplot(
            y="primary_diagnosis", x="rate_pct", data=diag_summary,
            palette="Blues_r", ax=ax
        )
        overall_mean = df["readmitted"].mean() * 100
        ax.axvline(overall_mean, color="#dc2626", linestyle="--", linewidth=1.2, label=f"Cohort Avg ({overall_mean:.1f}%)")
        ax.set_title("Readmission Rate by Primary Diagnosis", fontsize=11, fontweight="bold", pad=10)
        ax.set_xlabel("Readmission Rate (%)", fontsize=9)
        ax.set_ylabel("Diagnosis Category", fontsize=9)
        ax.legend(loc="lower right", fontsize=8.5)

        for p in ax.patches:
            width = p.get_width()
            ax.annotate(f"{width:.1f}%",
                        (width, p.get_y() + p.get_height() / 2.),
                        ha="left", va="center", fontsize=8,
                        xytext=(3, 0), textcoords="offset points")

        plt.tight_layout()
        return fig

    @staticmethod
    def plot_inpatient_visits_vs_readmission(df: pd.DataFrame):
        """Displays readmission rate stratified by number of prior inpatient admissions."""
        fig, ax = plt.subplots(figsize=(7, 3.8))
        grouped = df.groupby(["number_inpatient", "readmitted"]).size().unstack(fill_value=0)
        grouped = grouped.head(6)  # Focus on encounters with 0 to 5 prior visits
        grouped.columns = ["No Readmission", "Readmitted (<30d)"]
        grouped.plot(kind="bar", stacked=True, color=["#5bc0de", "#d9534f"], ax=ax)
        ax.set_title("Prior Inpatient Admissions vs. Readmission Proportion", fontsize=11, fontweight="bold", pad=10)
        ax.set_xlabel("Prior Inpatient Hospitalizations (Preceding 12 Months)", fontsize=9)
        ax.set_ylabel("Patient Encounter Count", fontsize=9)
        plt.xticks(rotation=0)
        plt.tight_layout()
        return fig

    @staticmethod
    def plot_age_vs_readmission(df: pd.DataFrame):
        """Displays readmission rate across age group brackets."""
        fig, ax = plt.subplots(figsize=(7, 3.8))
        age_summary = df.groupby("age_group")["readmitted"].mean().reset_index()
        age_summary["pct"] = age_summary["readmitted"] * 100
        sns.barplot(x="age_group", y="pct", data=age_summary, hue="age_group", palette="Blues_d", legend=False, ax=ax)
        ax.set_title("30-Day Readmission Rate by Patient Age Bracket", fontsize=11, fontweight="bold", pad=10)
        ax.set_xlabel("Age Group Bracket", fontsize=9)
        ax.set_ylabel("Readmission Percentage (%)", fontsize=9)

        for p in ax.patches:
            height = p.get_height()
            ax.annotate(f"{height:.1f}%",
                        (p.get_x() + p.get_width() / 2., height),
                        ha="center", va="bottom", fontsize=8.5,
                        xytext=(0, 3), textcoords="offset points")

        plt.tight_layout()
        return fig

    @staticmethod
    def plot_primary_diagnosis_distribution(df: pd.DataFrame):
        """Displays distribution of primary diagnoses in the cohort."""
        fig, ax = plt.subplots(figsize=(7.5, 4.2))
        diag_counts = df["primary_diagnosis"].value_counts().reset_index()
        diag_counts.columns = ["primary_diagnosis", "count"]
        sns.barplot(y="primary_diagnosis", x="count", data=diag_counts, hue="primary_diagnosis", palette="crest", legend=False, ax=ax)
        ax.set_title("Primary Diagnostic Categories in Patient Cohort", fontsize=11, fontweight="bold", pad=10)
        ax.set_xlabel("Patient Encounters", fontsize=9)
        ax.set_ylabel("Clinical ICD Category", fontsize=9)

        for p in ax.patches:
            width = p.get_width()
            ax.annotate(f"{int(width):,}",
                        (width, p.get_y() + p.get_height() / 2.),
                        ha="left", va="center", fontsize=8,
                        xytext=(4, 0), textcoords="offset points")

        plt.tight_layout()
        return fig

    @staticmethod
    def plot_feature_selection_ranking(feature_summary: List[dict]):
        """Visualizes feature association scores for the Feature Selection stage (PPT Slide 9 & 12)."""
        df_feat = pd.DataFrame(feature_summary).head(10)
        fig, ax = plt.subplots(figsize=(7.5, 4.2))
        sns.barplot(x="association_score", y="feature", data=df_feat, hue="feature", palette="Blues_r", legend=False, ax=ax)
        ax.set_title("Top 10 Attributes by Statistical Association with Target", fontsize=11, fontweight="bold", pad=10)
        ax.set_xlabel("Association Score (|Correlation| or Group Spread)", fontsize=9)
        ax.set_ylabel("Clinical Feature", fontsize=9)
        plt.tight_layout()
        return fig

    @staticmethod
    def plot_confusion_matrix(cm_data: list):
        """Visualizes the confusion matrix."""
        fig, ax = plt.subplots(figsize=(4.8, 3.8))
        sns.heatmap(
            cm_data, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=["Pred: No", "Pred: Yes"],
            yticklabels=["True: No", "True: Yes"],
            ax=ax
        )
        ax.set_title("Holdout Test Set Confusion Matrix", fontsize=11, fontweight="bold", pad=10)
        ax.set_xlabel("Predicted Class", fontsize=9)
        ax.set_ylabel("Actual Class", fontsize=9)
        plt.tight_layout()
        return fig

    @staticmethod
    def plot_roc_curve(y_test, y_prob, roc_auc_val: float):
        """Plots the Receiver Operating Characteristic (ROC) curve."""
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        fig, ax = plt.subplots(figsize=(5.2, 4.0))
        ax.plot(fpr, tpr, color="#2b5c8f", lw=2, label=f"Model ROC (AUC = {roc_auc_val:.4f})")
        ax.plot([0, 1], [0, 1], color="#94a3b8", lw=1.5, linestyle="--", label="Random Classifier (AUC = 0.50)")
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=9)
        ax.set_ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=9)
        ax.set_title("ROC Curve on Test Partition", fontsize=11, fontweight="bold", pad=10)
        ax.legend(loc="lower right", fontsize=8.5)
        plt.tight_layout()
        return fig

"""
Analytics Service: Visual healthcare analytics, cohort distribution plotting, and metric visualizations.
"""
from typing import List, Any
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

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
        sns.barplot(x=counts.index, y=counts.values, palette=palette, ax=ax)
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
        fig, ax = plt.subplots(figsize=(6.5, 3.8))
        sns.boxplot(
            x="readmitted", y="time_in_hospital", data=df,
            palette=["#337ab7", "#e74c3c"], ax=ax, width=0.45
        )
        ax.set_xticklabels(["No Readmission", "Readmitted (<30d)"])
        ax.set_title("Length of Stay vs. Readmission Status", fontsize=11, fontweight="bold", pad=10)
        ax.set_xlabel("Encounter Outcome", fontsize=9)
        ax.set_ylabel("Hospital Stay Duration (Days)", fontsize=9)
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
        ax.set_title("Prior Inpatient Admissions vs. Readmission Risk", fontsize=11, fontweight="bold", pad=10)
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
        sns.barplot(x="age_group", y="pct", data=age_summary, palette="Blues_d", ax=ax)
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
        """Displays distribution of primary diagnoses and associated readmission rate."""
        fig, ax = plt.subplots(figsize=(7.5, 4.2))
        diag_counts = df["primary_diagnosis"].value_counts().reset_index()
        diag_counts.columns = ["primary_diagnosis", "count"]
        sns.barplot(y="primary_diagnosis", x="count", data=diag_counts, palette="crest", ax=ax)
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
        plt.tight_layout()
        return fig

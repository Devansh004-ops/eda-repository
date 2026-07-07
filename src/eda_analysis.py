"""
eda_analysis.py
=============================================
Exploratory Data Analysis on the Titanic Dataset

Goal: uncover patterns and trends in what determined passenger survival,
using statistical summaries and visualizations, and identify the
strongest influencing factors.

Usage:
    python src/eda_analysis.py

Outputs (written to results/):
    - statistical_summary.txt
    - correlation_heatmap.png
    - correlation_with_survival.png
    - distributions_by_survival.png
    - categorical_breakdown.png
    - pairplot.png
    - key_influencing_factors.csv
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 110

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "titanic_cleaned.csv")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

report_lines = []


def log(line=""):
    print(line)
    report_lines.append(line)


def statistical_summary(df):
    log("=" * 60)
    log("STATISTICAL SUMMARY")
    log("=" * 60)

    log("\n--- Numeric columns: describe() ---")
    log(df[["Age", "Fare", "SibSp", "Parch"]].describe().round(2).to_string())

    log("\n--- Skewness & Kurtosis (shape of distribution) ---")
    for col in ["Age", "Fare"]:
        skew = stats.skew(df[col].dropna())
        kurt = stats.kurtosis(df[col].dropna())
        shape = "right-skewed (long tail of high values)" if skew > 0.5 else \
                "left-skewed" if skew < -0.5 else "roughly symmetric"
        log(f"{col}: skew={skew:.2f} ({shape}), kurtosis={kurt:.2f}")

    log("\n--- Categorical breakdowns ---")
    for col in ["Survived", "Pclass", "Sex", "Embarked"]:
        log(f"\n{col} value counts:")
        counts = df[col].value_counts()
        pcts = (df[col].value_counts(normalize=True) * 100).round(1)
        for val in counts.index:
            log(f"  {val}: {counts[val]} ({pcts[val]}%)")

    log("\n--- Overall survival rate ---")
    log(f"{df['Survived'].mean()*100:.1f}% of passengers survived ({df['Survived'].sum()} of {len(df)})")


def correlation_analysis(df):
    log("\n" + "=" * 60)
    log("CORRELATION ANALYSIS")
    log("=" * 60)

    # Build numeric-encoded copy for correlation
    corr_df = df.copy()
    corr_df["Sex_encoded"] = corr_df["Sex"].map({"male": 0, "female": 1})
    corr_df["Embarked_encoded"] = corr_df["Embarked"].map({"S": 0, "C": 1, "Q": 2})

    numeric_cols = ["Survived", "Pclass", "Sex_encoded", "Age", "SibSp", "Parch",
                     "Fare", "Embarked_encoded", "HasCabin", "FamilySize"]
    numeric_cols = [c for c in numeric_cols if c in corr_df.columns]

    corr_matrix = corr_df[numeric_cols].corr()

    # Full heatmap
    plt.figure(figsize=(9, 7))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                square=True, linewidths=0.5)
    plt.title("Correlation Heatmap — All Features")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "correlation_heatmap.png"), bbox_inches="tight")
    plt.close()

    # Correlation with Survived specifically, ranked
    surv_corr = corr_matrix["Survived"].drop("Survived").sort_values(key=abs, ascending=False)
    log("\n--- Correlation with Survival (ranked by strength) ---")
    for feat, val in surv_corr.items():
        direction = "positively" if val > 0 else "negatively"
        log(f"  {feat}: {val:.3f} ({direction} correlated)")

    plt.figure(figsize=(8, 6))
    colors = ["#16a34a" if v > 0 else "#dc2626" for v in surv_corr.values]
    surv_corr.plot(kind="barh", color=colors, edgecolor="black")
    plt.title("Correlation with Survival — Key Influencing Factors")
    plt.xlabel("Correlation Coefficient")
    plt.axvline(0, color="black", lw=0.8)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "correlation_with_survival.png"), bbox_inches="tight")
    plt.close()

    surv_corr.to_frame("correlation_with_survival").to_csv(
        os.path.join(RESULTS_DIR, "key_influencing_factors.csv")
    )

    return surv_corr


def distribution_plots(df):
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    sns.histplot(data=df, x="Age", hue="Survived", kde=True, ax=axes[0, 0],
                 palette=["#dc2626", "#16a34a"], element="step")
    axes[0, 0].set_title("Age Distribution by Survival")

    sns.histplot(data=df, x="Fare", hue="Survived", kde=True, ax=axes[0, 1],
                 palette=["#dc2626", "#16a34a"], element="step")
    axes[0, 1].set_title("Fare Distribution by Survival")

    sns.boxplot(data=df, x="Pclass", y="Age", hue="Survived", ax=axes[0, 2],
                palette=["#dc2626", "#16a34a"])
    axes[0, 2].set_title("Age by Class and Survival")

    sns.violinplot(data=df, x="Survived", y="Fare", ax=axes[1, 0], palette=["#dc2626", "#16a34a"])
    axes[1, 0].set_title("Fare Spread by Survival")
    axes[1, 0].set_xticklabels(["Died", "Survived"])

    sns.countplot(data=df, x="Pclass", hue="Survived", ax=axes[1, 1], palette=["#dc2626", "#16a34a"])
    axes[1, 1].set_title("Class Composition by Survival")

    sns.countplot(data=df, x="Sex", hue="Survived", ax=axes[1, 2], palette=["#dc2626", "#16a34a"])
    axes[1, 2].set_title("Sex Composition by Survival")

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "distributions_by_survival.png"), bbox_inches="tight")
    plt.close()


def categorical_breakdown(df):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    ct1 = pd.crosstab(df["Pclass"], df["Survived"], normalize="index") * 100
    ct1.plot(kind="bar", stacked=True, ax=axes[0], color=["#dc2626", "#16a34a"])
    axes[0].set_title("Survival % by Class")
    axes[0].set_ylabel("Percentage")
    axes[0].legend(["Died", "Survived"])

    ct2 = pd.crosstab(df["Sex"], df["Survived"], normalize="index") * 100
    ct2.plot(kind="bar", stacked=True, ax=axes[1], color=["#dc2626", "#16a34a"])
    axes[1].set_title("Survival % by Sex")
    axes[1].set_ylabel("Percentage")
    axes[1].legend(["Died", "Survived"])

    ct3 = pd.crosstab(df["Embarked"], df["Survived"], normalize="index") * 100
    ct3.plot(kind="bar", stacked=True, ax=axes[2], color=["#dc2626", "#16a34a"])
    axes[2].set_title("Survival % by Embarkation Port")
    axes[2].set_ylabel("Percentage")
    axes[2].legend(["Died", "Survived"])

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "categorical_breakdown.png"), bbox_inches="tight")
    plt.close()


def pairplot(df):
    subset = df[["Survived", "Age", "Fare", "Pclass", "FamilySize"]].copy()
    g = sns.pairplot(subset, hue="Survived", palette=["#dc2626", "#16a34a"],
                      diag_kind="kde", plot_kws={"alpha": 0.5, "s": 20})
    g.fig.suptitle("Pairwise Relationships by Survival", y=1.02)
    g.savefig(os.path.join(RESULTS_DIR, "pairplot.png"), bbox_inches="tight")
    plt.close()


def main():
    df = pd.read_csv(DATA_PATH)
    if "FamilySize" not in df.columns:
        df["FamilySize"] = df["SibSp"] + df["Parch"] + 1

    statistical_summary(df)
    surv_corr = correlation_analysis(df)
    distribution_plots(df)
    categorical_breakdown(df)
    pairplot(df)

    log("\n" + "=" * 60)
    log("TOP 3 KEY INFLUENCING FACTORS")
    log("=" * 60)
    top3 = surv_corr.abs().sort_values(ascending=False).head(3)
    for i, (feat, _) in enumerate(top3.items(), 1):
        log(f"{i}. {feat} (correlation: {surv_corr[feat]:.3f})")

    with open(os.path.join(RESULTS_DIR, "statistical_summary.txt"), "w") as f:
        f.write("\n".join(report_lines))

    print(f"\nAll outputs saved to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()

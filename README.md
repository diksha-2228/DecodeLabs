<h1 align="center">📊 Advanced EDA & Feature Engineering Pipeline</h1>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white" alt="Python Version">
  <img src="https://img.shields.io/badge/Pandas-2.0%2B-150458?logo=pandas&logoColor=white" alt="Pandas">
  <img src="https://img.shields.io/badge/Scikit--Learn-F7931E?logo=scikit-learn&logoColor=white" alt="Scikit-Learn">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
</p>

<p align="center">
  <strong>An enterprise-grade data preprocessing pipeline that automates cleaning, anomaly mitigation, and structural formatting for messy real estate datasets.</strong>
</p>

---

### 📍 Quick Navigation
[🚀 Quickstart](#-quickstart) | [⚙️ Pipeline Architecture](#%EF%B8%8F-pipeline-architecture) | [📦 Project Structure](#-project-structure) | [📈 Exploratory Visualizations](#-exploratory-visualizations) | [✅ Phase Explanations](#-requirements-coverage--detailed-phase-explanations)

---

## 📌 Overview
Real-world datasets are rarely ready for Machine Learning models; they suffer from unmitigated outliers, multi-collinearity, unstructured categorical text, and severe data skewness. This repository implements an automated, reproducible **Data Science Pipeline** that takes raw, volatile house listing data and transforms it into an optimized, model-ready format using advanced mathematical strategies (IQR clipping, Z-score Winsorization, and Pandas/Pandera validation).

---

## 🛠️ Features & Transformations

| Data Engineering Technique | Why It Matters / Business Impact |
| :--- | :--- |
| **Missingness Decision Matrix** | Dynamically drops or imputes missing entries based on severity (<5% drop, 5-20% imputation, >20% KNN). |
| **Z-Score & IQR Winsorization** | Clips extreme outliers safely without dropping valuable real estate records. |
| **Multi-Collinearity Removal** | Cleans features with high correlation to prevent model over-fitting and variance inflation. |
| **Pandera Schema Validation** | Enforces strict programmatic check constraints to maintain downstream data integrity. |

---

## 🚀 Quickstart

Follow these explicit steps to configure your local environment in **VS Code** and run the data pipeline.

### Step 1: Environment Setup
Clone the repository and navigate to the project directory, then create a clean python virtual environment.

```bash
# Clone and enter directory
git clone [https://github.com/diksha-2228/DecodeLabs.git](https://github.com/diksha-2228/DecodeLabs.git)
cd DecodeLabs

# Create virtual environment
python -m venv venv

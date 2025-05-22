# 📊 Census Income Classification using Logistic Regression and Recursive Feature Elimination (RFE)

This project demonstrates how to use **Logistic Regression**, a classical and interpretable machine learning algorithm, in combination with **Recursive Feature Elimination (RFE)** to predict whether an individual earns **more than \$50,000 annually** based on their demographic and employment attributes.

The model is trained and evaluated on the **UCI Adult Census Income Dataset**, one of the most widely used real-world benchmarks for income classification tasks.

---

## 🎯 Project Goals

- Build a reliable, interpretable binary classification model
- Understand the relationship between features and income bracket
- Apply **RFE** to reduce dimensionality and improve performance
- Use **F1-score** to evaluate model robustness under class imbalance

---

## 📚 Dataset Information

### 📌 Overview

- **Name**: Adult Census Income Dataset (aka "Census Income" or "Adult")
- **Source**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/adult)
- **Format**: `adult.data` (CSV-like, no headers)
- **Size**: 32,561 rows × 15 columns
- **Problem Type**: Binary classification
- **Target Column**: `income` → `<=50K` or `>50K`

### 📋 Original Columns

| Column Name         | Description                                                   |
|---------------------|---------------------------------------------------------------|
| `age`               | Age of the individual (continuous)                            |
| `workclass`         | Type of employment (e.g., Private, Government, Self-Employed) |
| `fnlwgt`            | Sampling weight (used in census calculations)                 |
| `education`         | Highest level of education attained                           |
| `education-num`     | Numeric encoding of education level                           |
| `marital-status`    | Marital status (e.g., Married, Divorced, Single)              |
| `occupation`        | Job type (e.g., Exec-managerial, Tech-support)                |
| `relationship`      | Relationship within household (e.g., Husband, Not-in-family)  |
| `race`              | Race (e.g., White, Black, Asian-Pac-Islander)                 |
| `sex`               | Gender                                                        |
| `capital-gain`      | Investment income gains (continuous)                          |
| `capital-loss`      | Investment income losses (continuous)                         |
| `hours-per-week`    | Work hours per week (continuous)                              |
| `native-country`    | Country of origin                                             |
| `income`            | Target class: `<=50K` or `>50K`                               |

---

## 🧹 Data Preparation

- Loaded `.data` file using `pandas.read_csv()`
- Replaced `" ?"` entries with `NaN` and removed incomplete rows
- Encoded categorical columns using `LabelEncoder`
- Normalized numerical features using `StandardScaler` for model convergence

---

## 🔎 Modeling Pipeline

### 1. Baseline Logistic Regression
- Trained on all features to establish baseline accuracy and F1-score

### 2. Recursive Feature Elimination (RFE)
- Used RFE from `sklearn.feature_selection` to:
  - Rank feature importance
  - Test all subsets from 1 to 14 features
  - Evaluate each using **weighted F1-score** on test data

### 3. Model Evaluation
- Compared models by number of features and their F1-scores
- Identified optimal feature subset
- Final model trained and evaluated on selected features

---

## 📈 Why F1-Score?

- The dataset is slightly imbalanced (more `<=50K` than `>50K`)
- **F1-score balances precision and recall**, providing a better measure of model effectiveness than raw accuracy
- Used `average='weighted'` to account for both classes proportionally

---

## ✅ Results

- Achieved strong F1-score with fewer than all features
- Top predictors included:
  - `education-num`
  - `hours-per-week`
  - `capital-gain`
  - `occupation`
  - `age`
- Reduced dimensionality improved model generalization and interpretability

---

## 💻 Running Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/aarshdesai-ds/census-income-logistic-regression.git
cd census-income-logistic-regression

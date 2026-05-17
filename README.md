# Census Income Prediction (UCI Adult Dataset, 1994)

A binary classification study using the UCI Adult Census dataset to predict whether an individual earns **more or less than $50,000/year**, comparing a **Baseline Logistic Regression** against a **Recursive Feature Elimination (RFE)-optimized** model across 32,561 individuals.

---

## Table of Contents

- [Overview](#overview)
- [Research Questions](#research-questions)
- [Dataset Description](#dataset-description)
- [Methodology](#methodology)
- [Key Findings](#key-findings)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [How to Run](#how-to-run)
- [Results Summary](#results-summary)
- [Ideas for Extension](#ideas-for-extension)

---

## Overview

This project analyzes **32,561 individuals** from the 1994 U.S. Census to evaluate two approaches to binary income classification:

1. **Baseline Logistic Regression** — trained on all 99 features after one-hot encoding
2. **RFE-Optimized Logistic Regression** — trained on the 5 most informative features selected via Recursive Feature Elimination

The core question: *can a dramatically simplified model (5 features vs. 99) match the predictive power of a full-feature baseline — and which individual attributes drive income most?*

---

## Research Questions

- How accurately can Logistic Regression classify income groups using demographic and employment data?
- Does Recursive Feature Elimination reduce model complexity without sacrificing performance?
- Which features are the strongest predictors of earning above $50K/year?
- How does class imbalance (75.9% vs. 24.1%) affect model evaluation and metric selection?
- What do the model's errors reveal about the limits of a linear classifier on census data?

---

## Dataset Description

**Source:** [UCI Machine Learning Repository — Adult Dataset](https://archive.ics.uci.edu/ml/datasets/adult)
**Extracted from:** 1994 U.S. Census Bureau database

| Property | Value |
|----------|-------|
| Total Records | 32,561 |
| Original Features | 15 |
| File | `adult.data` (3.97 MB) |
| Task | Binary Classification |
| Target Variable | `income-group` (`<=50K` = 0, `>50K` = 1) |

### Class Distribution (Imbalanced)

| Class | Label | Count | Share |
|-------|:-----:|------:|------:|
| ≤ $50K | 0 | 24,720 | **75.9%** |
| > $50K | 1 | 7,841 | **24.1%** |

### Gender Distribution

| Gender | Encoded | Count | Share |
|--------|:-------:|------:|------:|
| Male | 0 | 21,790 | 66.9% |
| Female | 1 | 10,771 | 33.1% |

### Feature Descriptions

| Feature | Type | Description |
|---------|------|-------------|
| `age` | Numeric | Age of the individual |
| `workclass` | Categorical | Employment type (Private, Federal-gov, Self-emp, etc.) |
| `fnlwgt` | Numeric | Census sampling weight — *dropped, not predictive* |
| `education` | Categorical | Highest education level (Bachelors, HS-grad, Masters, etc.) |
| `education-years` | Numeric | Numeric representation of education level (1–16) |
| `marital-status` | Categorical | Marital status (7 categories) |
| `occupation` | Categorical | Job role (14 categories incl. Exec-managerial, Prof-specialty) |
| `relationship` | Categorical | Household relationship (Husband, Wife, Own-child, etc.) |
| `race` | Categorical | Race category (5 categories) |
| `sex` | Binary | Gender — encoded Male=0, Female=1 |
| `capital-gain` | Numeric | Investment capital gains |
| `capital-loss` | Numeric | Investment capital losses |
| `hours-per-week` | Numeric | Average weekly hours worked |
| `native-country` | Categorical | Country of origin (42 unique values) |
| `income-group` | **Target** | <=50K → 0, >50K → 1 |

---

## Methodology

### Data Cleaning

Missing values in the raw file are encoded as `"?"` strings in three columns — `workclass`, `occupation`, and `native-country`. These are replaced with `NaN` at load time and the affected rows are dropped. `fnlwgt` (census sampling weight) is removed as it is an estimation artifact rather than an individual-level predictor.

### Feature Encoding

- `sex` and `income-group` are binary-mapped to integers
- The remaining 7 categorical columns are one-hot encoded via `pd.get_dummies()`, producing **93 dummy variables**
- Combined with 5 numeric columns and 1 binary column, the final feature matrix has **99 columns**

### Normalization

Z-score standardization is applied to the 5 numeric columns (`age`, `education-years`, `capital-gain`, `capital-loss`, `hours-per-week`), fit on the training set and applied to both train and test to prevent data leakage.

### Train / Test Split

| Set | Rows | Share |
|-----|-----:|------:|
| Training | 22,792 | 70% |
| Test | 9,769 | 30% |

*`random_state=42`*

### Model 1 — Baseline Logistic Regression

Standard Logistic Regression trained on all 99 normalized features. Evaluated using accuracy, per-class precision / recall / F1, and a confusion matrix on the held-out test set.

### Model 2 — RFE-Optimized Logistic Regression

Recursive Feature Elimination iterates from 1 to 5 features (the 5 numeric columns), training and evaluating a separate Logistic Regression at each step. The feature count that maximizes **weighted F1-score** on the test set is selected as the final model.

**Weighted F1-score** is used as the primary metric throughout — accuracy alone is misleading under class imbalance (a model predicting ≤50K every time scores 75.9% while being entirely useless for the >50K class).

---

## Key Findings

### 1. A 5-Feature Model Matches a 99-Feature Baseline

The RFE model using only 5 numeric features achieves effectively identical test accuracy (~82%) and a slightly better weighted F1 (~0.807 vs 0.80) compared to the full 99-feature baseline — while being **20× simpler**.

### 2. Capital Gain Is the Single Strongest Predictor

With `capital-gain` alone, RFE achieves an F1 of **0.8852** on the majority class and **0.3268** on the minority class. No other single feature comes close, reflecting the strong real-world link between investment income and total earnings bracket.

### 3. Weighted F1 Exposes What Accuracy Hides

The baseline model achieves **0.95 recall on ≤50K** but only **0.40 recall on >50K** — meaning it misclassifies 60% of high earners as low earners. This asymmetry is invisible to accuracy but fully surfaces in the F1 breakdown.

### 4. RFE Feature Progression

Each added feature meaningfully improves minority-class detection:

| # Features | Features Selected | F1 — ≤50K | F1 — >50K |
|:---:|---|:---:|:---:|
| 1 | `capital-gain` | 0.8852 | 0.3268 |
| 2 | `education-years`, `capital-gain` | 0.8877 | 0.3950 |
| 3 | `age`, `education-years`, `capital-gain` | 0.8841 | 0.4569 |
| 4 | `age`, `education-years`, `capital-gain`, `hours-per-week` | 0.8852 | 0.4907 |
| **5 ✅** | **`age`, `education-years`, `capital-gain`, `capital-loss`, `hours-per-week`** | **0.8870** | **0.5071** |

### 5. Top 5 Predictive Features

| Rank | Feature | Insight |
|:---:|---|---|
| 1 | `capital-gain` | Investment returns are the strongest signal of high earner status |
| 2 | `education-years` | Each additional year of education shifts income probability measurably |
| 3 | `age` | Earnings accumulate with career experience — strong monotonic effect |
| 4 | `capital-loss` | Recording investment losses signals active market participation — a high-earner behavior |
| 5 | `hours-per-week` | More hours correlates with salary-driven roles and higher pay bands |

### 6. Confusion Matrix Reveals a Systematic False Negative Problem

```
Baseline model — Test Set (9,769 samples):

                  Predicted ≤50K    Predicted >50K
Actual ≤50K           7,049              406       ← 5.4% false positive rate
Actual >50K           1,390              924       ← 60.2% false negative rate
```

The model correctly identifies 94.6% of low earners but misclassifies **1,390 high earners** as low earners. This is the central limitation of a linear classifier on an imbalanced dataset without resampling or cost-sensitive adjustments.

---

## Project Structure

```
census-income-logistic-regression/
│
├── CensusAnalysis.ipynb        # Main analysis notebook
├── README.md
└── adult.data                  # UCI Adult Census dataset (3.97 MB)
```

---

## Requirements

```
Python 3.8+
pandas
numpy
scikit-learn
jupyter
```

Install dependencies:

```bash
pip install pandas numpy scikit-learn jupyter
```

---

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/aarshdesai-ds/census-income-logistic-regression.git
   cd census-income-logistic-regression
   ```

2. Install dependencies:
   ```bash
   pip install pandas numpy scikit-learn jupyter
   ```

3. Launch the notebook:
   ```bash
   jupyter notebook CensusAnalysis.ipynb
   ```

4. Run all cells sequentially (`Cell > Run All`) to reproduce the full pipeline: data loading → cleaning → encoding → normalization → baseline model → RFE → final evaluation.

---

## Results Summary

| Model | Features Used | Test Accuracy | F1 — ≤50K | F1 — >50K | Weighted F1 |
|---|:---:|:---:|:---:|:---:|:---:|
| Baseline Logistic Regression | 99 | **82%** | 0.89 | 0.51 | 0.80 |
| RFE-Optimized (5 features) | **5** | ~82% | 0.8870 | **0.5071** | **~0.807** |

**Classification Report — Baseline Model:**

| Class | Precision | Recall | F1-Score | Support |
|-------|:---------:|:------:|:--------:|--------:|
| 0 — ≤$50K | 0.84 | 0.95 | 0.89 | 7,455 |
| 1 — >$50K | 0.69 | 0.40 | 0.51 | 2,314 |
| **Accuracy** | | | **0.82** | **9,769** |
| Macro Avg | 0.77 | 0.67 | 0.70 | 9,769 |
| Weighted Avg | 0.80 | 0.82 | 0.80 | 9,769 |

---

## Ideas for Extension

### 1. Handle Class Imbalance Directly
The 75.9% / 24.1% split causes the model to miss 60% of high earners. Applying **SMOTE** (Synthetic Minority Oversampling) or setting `class_weight='balanced'` in Logistic Regression would force the model to pay more attention to the minority class and likely improve >50K recall substantially.

### 2. Add Non-Linear Models for Comparison
Logistic Regression assumes a linear decision boundary. Comparing against **Random Forest**, **Gradient Boosting (XGBoost)**, and **SVM** would quantify how much predictive power is left on the table — especially given the likely interactions between education, occupation, and capital gains.

### 3. Extend RFE to All 99 Features
The current RFE is restricted to the 5 numeric columns. Running it across all 99 features (including dummies) would reveal whether any categorical sub-groups — specific occupations, marital statuses, or countries — add independent signal beyond what the numeric features already capture.

### 4. Coefficient Analysis and Odds Ratios
Logistic Regression coefficients can be exponentiated to produce **odds ratios** — a direct, interpretable measure of how each feature multiplies the odds of earning >50K. Plotting these for the final 5-feature model makes the model's logic transparent and is a strong addition to any portfolio write-up.

### 5. Calibration Analysis
A well-performing classifier isn't necessarily well-calibrated. Plotting a **reliability diagram** (predicted probability vs. actual frequency) tests whether the model's confidence scores are trustworthy — especially useful if the output probabilities are used downstream for decision-making.

### 6. Cross-Validation Instead of a Single Split
The current 70/30 split produces one estimate of generalization. Replacing it with **10-fold stratified cross-validation** would give a distribution of F1 scores and quantify model stability — particularly important for the minority class where a single unlucky split can noticeably swing results.

### 7. Demographic Bias Audit
The dataset contains `sex`, `race`, and `native-country`. Evaluating model performance **separately by demographic group** after training checks whether error rates are uniform or systematically higher for specific populations — a standard fairness analysis and increasingly expected for any model trained on census data.

### 8. Feature Interaction Terms
Income prediction likely involves non-additive effects — for example, `capital-gain` may matter more for highly educated individuals. Adding **interaction terms** (e.g., `capital-gain × education-years`, `age × hours-per-week`) and re-running RFE would test whether interactions add predictive power beyond main effects.

### 9. Threshold Tuning
The default classification threshold is 0.5. Lowering it (e.g., to 0.3) would increase recall for the >50K class at the cost of more false positives. Plotting the **precision-recall curve** and selecting a threshold based on the use case (e.g., minimizing missed high earners) is a practical and often high-impact step.

### 10. Deploy as an Interactive Web App
Wrap the final 5-feature model in a **Streamlit app** with input sliders for age, education years, capital gain/loss, and hours per week — and display a real-time income probability estimate. The simplicity of the 5-feature model makes this straightforward to deploy and a strong portfolio demonstration.

---

*Data sourced from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/adult). Original dataset extracted from the 1994 U.S. Census Bureau database by Ronny Kohavi and Barry Becker.*

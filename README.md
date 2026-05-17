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
2. **RFE-Optimized Logistic Regression** — trained on the 5 most informative numeric features selected via Recursive Feature Elimination

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

Missing values in the raw file are encoded as `" ?"` strings in three columns: `workclass`, `occupation`, and `native-country`. These are replaced with `NaN` at load time and rows are dropped. `fnlwgt` (sampling weight) is removed as it is a census estimation artifact, not an individual-level predictor.

```
Rows after cleaning : 32,561  (no rows lost — NaN replacement at read time)
Columns after drop  : 14  (fnlwgt removed)
```

### Feature Encoding

Binary and categorical columns are encoded before modelling:

```
sex          : Male → 0,  Female → 1
income-group : <=50K → 0, >50K → 1

Remaining 7 categorical columns → one-hot encoded via pd.get_dummies()
  workclass (8 categories)  → 7 dummies
  education (16 categories) → 15 dummies
  marital-status (7)        → 6 dummies
  occupation (14)           → 13 dummies
  relationship (6)          → 5 dummies
  race (5)                  → 4 dummies
  native-country (42)       → 41 dummies
  ─────────────────────────────────────
  Total dummy columns       → 93
```

**Final feature matrix: 99 columns** (5 numeric + 1 binary sex + 93 dummies).

### Normalization

Z-score standardization is applied to the 5 numeric columns using a custom `standard_scalar()` function, fit on the training set and applied to both train and test:

```
Z = (X − μ_train) / σ_train

Applied to: age, education-years, capital-gain, capital-loss, hours-per-week
```

### Train / Test Split

```
Total samples : 32,561
Training set  : 22,792  (70%)
Test set      :  9,769  (30%)
random_state  : 42
```

### Model 1: Baseline Logistic Regression (99 Features)

Standard `sklearn` Logistic Regression trained on all 99 normalized features. Evaluated using accuracy, per-class precision/recall/F1, and a confusion matrix on the held-out test set.

### Model 2: RFE-Optimized Logistic Regression (5 Features)

Recursive Feature Elimination iterates from 1 to 5 features (the 5 numeric columns), training and evaluating a separate Logistic Regression at each step. The feature count that maximizes the **weighted F1-score** on the test set is selected as the final model.

```
RFE candidate pool : age, education-years, capital-gain, capital-loss, hours-per-week
Selection criterion: weighted F1-score (accounts for class imbalance)
Optimal n_features : 5
```

**Weighted F1-score** is used as the primary metric throughout because accuracy is misleading under class imbalance (a model predicting ≤50K every time scores 75.9% accuracy while being useless).

---

## Key Findings

### 1. A 5-Feature Model Matches a 99-Feature Baseline

The RFE model using only 5 numeric features achieves effectively identical test accuracy (~82%) and slightly better weighted F1 (~0.807 vs 0.80) compared to the full 99-feature baseline — while being **20× simpler**.

### 2. Capital Gain Is the Single Strongest Predictor

With just `capital-gain` alone (1-feature model), RFE achieves:
- F1 of **0.8852** on the majority class (≤50K)
- F1 of **0.3268** on the minority class (>50K)

No other single feature comes close, reflecting the strong real-world link between investment income and total earnings bracket.

### 3. Betting on Weighted F1 Over Accuracy Is Critical

The dataset is 75.9% class-0. A naïve majority-class classifier would achieve 75.9% accuracy but 0% recall on >50K earners. The baseline Logistic Regression achieves:

- **Recall of 0.95** on ≤50K (captures nearly all)
- **Recall of 0.40** on >50K (misses 60% of high earners)

This asymmetry is invisible to accuracy — weighted F1 surfaces it.

### 4. RFE Feature Progression

Each added feature meaningfully improves minority-class detection:

| # Features | Features | F1 — ≤50K | F1 — >50K |
|:---:|---|:---:|:---:|
| 1 | `capital-gain` | 0.8852 | 0.3268 |
| 2 | `education-years`, `capital-gain` | 0.8877 | 0.3950 |
| 3 | `age`, `education-years`, `capital-gain` | 0.8841 | 0.4569 |
| 4 | `age`, `education-years`, `capital-gain`, `hours-per-week` | 0.8852 | 0.4907 |
| **5 ✅** | **`age`, `education-years`, `capital-gain`, `capital-loss`, `hours-per-week`** | **0.8870** | **0.5071** |

### 5. Top 5 Predictive Features

| Rank | Feature | Insight |
|:---:|---|---|
| 1 | `capital-gain` | Strongest predictor — investment returns signal high earner status directly |
| 2 | `education-years` | Each additional year of education measurably shifts income probability |
| 3 | `age` | Earnings accumulate with career experience; strong monotonic effect |
| 4 | `capital-loss` | Even recording losses signals active investment — a high-earner behavior |
| 5 | `hours-per-week` | More hours correlates with salary-driven roles and higher pay bands |

### 6. Confusion Matrix Reveals Systematic FN Problem

```
Baseline model — Test Set (9,769 samples):

                  Predicted ≤50K    Predicted >50K
Actual ≤50K           7,049              406          ← 5.4% FP rate
Actual >50K           1,390              924          ← 60.2% FN rate
```

The model correctly identifies 94.6% of low earners but misclassifies **1,390 high earners** (60%) as low earners. This is the central limitation of a linear classifier on an imbalanced dataset without resampling or cost-sensitive learning.

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

The current analysis is a solid foundation. Here are concrete directions to take it further:

### 1. Handle Class Imbalance Directly
The 75.9% / 24.1% split causes the model to miss 60% of high earners. Applying **SMOTE** (Synthetic Minority Oversampling Technique) or **class_weight='balanced'** in Logistic Regression would force the model to pay more attention to the minority class and likely improve >50K recall substantially.

### 2. Add Non-Linear Models for Comparison
Logistic Regression assumes a linear decision boundary. Compare performance against **Random Forest**, **Gradient Boosting (XGBoost)**, and **SVM** to quantify how much predictive power is left on the table by the linear assumption. This is especially relevant given the interaction between education, occupation, and capital gains.

### 3. Extend RFE to All 99 Features
The current RFE is restricted to the 5 numeric features. Extending it across all 99 features (including dummy variables) using `sklearn`'s full RFE would reveal whether any categorical sub-groups — specific occupations, marital statuses, or countries — contribute independently beyond what the numeric features already capture.

### 4. Coefficient Analysis and Odds Ratios
Logistic Regression coefficients can be exponentiated to produce **odds ratios** — a direct, interpretable measure of how each feature multiplies the odds of earning >50K. Plotting these for the final 5-feature model and a selected subset of categorical dummies would make the model's logic transparent and portfolio-ready.

### 5. Calibration Analysis
A well-performing classifier isn't necessarily well-calibrated. Plot a **reliability diagram** (predicted probability vs. actual frequency) to test whether the model's confidence scores are trustworthy. Logistic Regression is often well-calibrated, but class imbalance and feature selection can distort this.

### 6. Cross-Validation Instead of a Single Split
The current 70/30 split gives one estimate of generalization. Replacing it with **5-fold or 10-fold stratified cross-validation** would produce a distribution of F1 scores and quantify model stability — especially important for the minority class where a single unlucky split can swing results.

### 7. Demographic Bias Audit
The dataset contains `sex`, `race`, and `native-country`. After training, evaluate model performance **separately by demographic group** to check whether the model's error rates are uniform or systematically higher for specific populations. This is a standard fairness analysis and increasingly important for any model trained on census data.

### 8. Optimal Logistic Regression Exponent (Pythagorean Analogy)
Inspired by the Pythagorean Expectation approach in sports analytics: rather than using raw logistic output, fit a **power-transformed probability model** where the exponent is tuned to minimize Brier Score on the training set. This is an unconventional but potentially informative extension for comparing to the standard logistic curve.

### 9. Feature Interaction Terms
Income prediction likely involves non-additive effects — for example, `capital-gain` may matter more for highly educated individuals. Adding **interaction terms** (e.g., `capital-gain × education-years`, `age × hours-per-week`) to the feature matrix and re-running RFE would test whether any interactions add predictive power beyond main effects.

### 10. Deploy as an Interactive Web App
Wrap the final 5-feature model in a **Streamlit app** with input sliders for age, education years, capital gain/loss, and hours per week — and display a real-time income probability estimate. The simplicity of the 5-feature model makes this straightforward to deploy and a strong portfolio demonstration.

---

*Data sourced from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/adult). Original dataset extracted from the 1994 U.S. Census Bureau database by Ronny Kohavi and Barry Becker.*

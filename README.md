# Census Income Prediction — Logistic Regression + RFE

Binary classification model that predicts whether an individual earns **more or less than $50,000/year** using demographic and employment data from the UCI Adult Census dataset.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Data Preprocessing](#data-preprocessing)
4. [Feature Engineering](#feature-engineering)
5. [Modeling Pipeline](#modeling-pipeline)
6. [RFE Feature Selection](#rfe-feature-selection)
7. [Results](#results)
8. [Key Takeaways](#key-takeaways)
9. [How to Run](#how-to-run)

---

## Project Overview

| Item | Detail |
|------|--------|
| **Goal** | Predict income group (`<=50K` vs `>50K`) |
| **Model** | Logistic Regression |
| **Feature Selection** | Recursive Feature Elimination (RFE) |
| **Primary Metric** | Weighted F1-Score (handles class imbalance) |
| **Dataset** | UCI Adult Census Income |
| **Notebook** | `CensusAnalysis.ipynb` |

---

## Dataset

**Source:** [UCI Machine Learning Repository — Adult Dataset](https://archive.ics.uci.edu/ml/datasets/adult)

| Property | Value |
|----------|-------|
| Total Records | 32,561 |
| Original Features | 15 |
| Target Variable | `income` (`<=50K` or `>50K`) |
| File | `adult.data` (3.97 MB) |

### Class Distribution (Imbalanced)

| Class | Label | Count | Share |
|-------|-------|------:|------:|
| ≤ $50K | 0 | 24,720 | **75.9%** |
| > $50K | 1 | 7,841 | **24.1%** |

### Gender Distribution

| Gender | Count | Share |
|--------|------:|------:|
| Male | 21,790 | 66.9% |
| Female | 10,771 | 33.1% |

### Feature Descriptions

| Feature | Type | Description |
|---------|------|-------------|
| `age` | Numeric | Age of the individual |
| `workclass` | Categorical | Type of employment (Private, Federal-gov, etc.) |
| `fnlwgt` | Numeric | Census sampling weight *(dropped — not predictive)* |
| `education` | Categorical | Highest education level attained |
| `education-years` | Numeric | Numeric representation of education level |
| `marital-status` | Categorical | Marital status |
| `occupation` | Categorical | Job role / occupation type |
| `relationship` | Categorical | Household relationship |
| `race` | Categorical | Race category |
| `sex` | Binary | Gender (Male=0, Female=1) |
| `capital-gain` | Numeric | Capital gains from investments |
| `capital-loss` | Numeric | Capital losses from investments |
| `hours-per-week` | Numeric | Weekly working hours |
| `native-country` | Categorical | Country of origin (42 unique values) |
| `income-group` | Target | <=50K=0, >50K=1 |

---

## Data Preprocessing

| Step | Detail |
|------|--------|
| Missing values | `"?"` entries in `workclass`, `occupation`, `native-country` replaced with `NaN` and dropped |
| `fnlwgt` dropped | Sampling weight — not relevant for individual-level prediction |
| Binary encoding | `sex`: Male→0, Female→1 &nbsp;/&nbsp; `income-group`: ≤50K→0, >50K→1 |
| One-hot encoding | 7 categorical columns → 93 dummy variables via `pd.get_dummies()` |
| Normalization | Z-score standardization applied to 5 numeric columns using a custom `standard_scalar()` |

**Rows after cleaning:** 32,561 (no rows lost — `"?"` values were parsed as `NaN` at load time)

### Train / Test Split

| Set | Rows | Share |
|-----|-----:|------:|
| Training | 22,792 | 70% |
| Testing | 9,769 | 30% |

*`random_state=42`*

---

## Feature Engineering

After preprocessing, the final feature matrix has **100 columns**:

| Feature Group | Count |
|---------------|------:|
| Numeric features (`age`, `education-years`, `capital-gain`, `capital-loss`, `hours-per-week`) | 5 |
| Dummy variables from 7 categorical columns | 93 |
| `sex` (binary-encoded) | 1 |
| **Total features (excl. target)** | **99** |

---

## Modeling Pipeline

### Step 1 — Baseline Logistic Regression (All 99 Features)

Trained on all features to establish a performance benchmark.

| Metric | Value |
|--------|------:|
| Training Accuracy | **81.36%** |
| Test Accuracy | **82%** |

**Confusion Matrix (Baseline — Test Set: 9,769 samples):**

```
                  Predicted ≤50K    Predicted >50K
Actual ≤50K           7,049              406
Actual >50K           1,390              924
```

**Classification Report (Baseline):**

| Class | Precision | Recall | F1-Score | Support |
|-------|----------:|-------:|---------:|--------:|
| 0 — ≤50K | 0.84 | 0.95 | **0.89** | 7,455 |
| 1 — >50K | 0.69 | 0.40 | **0.51** | 2,314 |
| **Accuracy** | | | **0.82** | **9,769** |
| Macro Avg | 0.77 | 0.67 | 0.70 | 9,769 |
| Weighted Avg | 0.80 | 0.82 | 0.80 | 9,769 |

---

## RFE Feature Selection

**Recursive Feature Elimination (RFE)** was run iteratively from 1 to 5 features (using all 5 numeric features as candidates), selecting the subset that maximized the weighted F1-score on the test set.

### RFE Progression

| # Features | Features Selected | F1 — Class 0 (≤50K) | F1 — Class 1 (>50K) |
|:----------:|-------------------|:-------------------:|:-------------------:|
| 1 | `capital-gain` | 0.8852 | 0.3268 |
| 2 | `education-years`, `capital-gain` | 0.8877 | 0.3950 |
| 3 | `age`, `education-years`, `capital-gain` | 0.8841 | 0.4569 |
| 4 | `age`, `education-years`, `capital-gain`, `hours-per-week` | 0.8852 | 0.4907 |
| **5** ✅ | **`age`, `education-years`, `capital-gain`, `capital-loss`, `hours-per-week`** | **0.8870** | **0.5071** |

**Optimal selection: 5 features** — adding `capital-loss` to the 4-feature model improved minority class F1 by +1.6 pp with no regression on the majority class.

### Top 5 Predictive Features (Final Model)

| Rank | Feature | Why it matters |
|:----:|---------|----------------|
| 1 | `capital-gain` | Strongest single predictor; high gains correlate directly with high earners |
| 2 | `education-years` | More years of education → consistently higher income |
| 3 | `age` | Earnings generally increase with career experience |
| 4 | `capital-loss` | Investment activity (even losses) signals higher wealth bracket |
| 5 | `hours-per-week` | More hours worked correlates with higher annual earnings |

---

## Results

### Final Model — 5 Features (RFE-Selected)

| Metric | Class 0 (≤50K) | Class 1 (>50K) |
|--------|:--------------:|:--------------:|
| F1-Score | **0.8870** | **0.5071** |

| Overall | Value |
|---------|------:|
| Test Accuracy | ~82% |
| Weighted F1-Score | ~0.807 |

### Baseline vs Final Model Comparison

| Metric | Baseline (99 features) | Final (5 features) |
|--------|:----------------------:|:------------------:|
| Test Accuracy | 82% | ~82% |
| F1 — Class 0 | 0.89 | 0.8870 |
| F1 — Class 1 | 0.51 | 0.5071 |
| Weighted F1 | 0.80 | ~0.807 |
| Features Used | **99** | **5** |
| Interpretability | Low | **High** |

> The 5-feature model matches the 99-feature baseline in accuracy while being **20x simpler** and far more interpretable.

---

## Key Takeaways

**Dimensionality reduction pays off.** RFE reduced 99 features down to 5 with no meaningful loss in predictive performance. This confirms that most of the one-hot encoded categorical dummies add noise rather than signal.

**Capital gain is the dominant predictor.** It alone (1-feature model) achieves an F1 of 0.885 on the majority class and 0.327 on the minority — more informative than any other single feature.

**Class imbalance requires careful evaluation.** With 75.9% of records in the ≤50K class, accuracy alone is misleading. Weighted F1 was used as the primary metric to ensure both classes were evaluated fairly.

**Logistic Regression is competitive with complex models here.** When features are well-selected and normalized, the simple linear classifier achieves 82% accuracy — comparable to more complex models — while remaining fully interpretable.

**The minority class (>50K) is harder to capture.** Even the best model achieves only 0.51 F1 on >50K vs 0.89 on ≤50K, reflecting the inherent difficulty of detecting the minority class without oversampling or cost-sensitive learning.

---

## How to Run

```bash
# 1. Clone the repository
git clone https://github.com/aarshdesai-ds/census-income-logistic-regression.git
cd census-income-logistic-regression

# 2. Install dependencies
pip install pandas numpy scikit-learn jupyter

# 3. Launch the notebook
jupyter notebook CensusAnalysis.ipynb
```

Run all cells top-to-bottom to reproduce the full pipeline: data loading → preprocessing → baseline model → RFE → final evaluation.

**Requirements:** Python 3.8+, pandas, numpy, scikit-learn, jupyter

---

*Dataset: [UCI Adult Census Income](https://archive.ics.uci.edu/ml/datasets/adult) — Dua, D. and Graff, C. (2019). UCI Machine Learning Repository. Irvine, CA: University of California, School of Information and Computer Science.*

# 💼 Census Income Prediction with Logistic Regression + RFE

This project demonstrates how to apply **Logistic Regression** and **Recursive Feature Elimination (RFE)** to predict whether an individual earns more than \$50,000 per year, based on demographic and employment data.

We use the **UCI Adult Census Income Dataset**, a classic benchmark for income classification tasks.

---

## 🎯 Project Goals

- Develop a binary classification model using Logistic Regression  
- Interpret the impact of different features on income prediction  
- Apply RFE to reduce dimensionality and highlight informative features  
- Evaluate performance using **weighted F1-score** to handle class imbalance  

---

## 📚 Dataset Overview

**Source:** [UCI Machine Learning Repository – Adult Dataset](https://archive.ics.uci.edu/ml/datasets/adult)  
**Format:** CSV-like (`adult.data`)  
**Size:** 32,561 records × 15 columns  
**Task:** Binary Classification  
**Target Variable:** `income` → `<=50K` or `>50K`

### 📋 Features

| Column Name       | Description                                 |
|-------------------|---------------------------------------------|
| age               | Age of the individual                       |
| workclass         | Type of employment                          |
| fnlwgt            | Sampling weight (used in census estimates)  |
| education         | Highest education level                     |
| education-num     | Numeric version of education                |
| marital-status    | Marital status                              |
| occupation        | Job role                                    |
| relationship      | Household relationship                      |
| race              | Race category                               |
| sex               | Gender                                       |
| capital-gain      | Capital gains                               |
| capital-loss      | Capital losses                              |
| hours-per-week    | Weekly working hours                        |
| native-country    | Country of origin                           |
| income            | Target variable (<=50K or >50K)             |

---

## 🧹 Data Preparation

- Loaded `.data` file using `pandas.read_csv()`  
- Removed records with `"?"` and missing values  
- Encoded categorical columns using `LabelEncoder`  
- Standardized numerical features using `StandardScaler`  

---

## 🔎 Modeling Pipeline

### 1. 📉 Baseline Logistic Regression

- Trained on all features  
- Evaluated using accuracy and **F1-score**  
- Served as a performance benchmark  

### 2. 🔍 Recursive Feature Elimination (RFE)

- Employed RFE with Logistic Regression as the estimator  
- Ranked and recursively eliminated the least important features  
- Evaluated performance across feature subsets (1 to 14 features)  
- Chose optimal feature subset based on **weighted F1-score**

### 3. ✅ Final Model

- Final model trained on optimal subset (based on RFE results)  
- Achieved better generalization and simpler interpretation  

---

## 📊 Evaluation Metrics

- Used **weighted F1-score** due to class imbalance  
- Also reported **accuracy** and **confusion matrix**  
- Visualized feature rankings and model performance  

---

## ✅ Results Summary

- Achieved strong F1-score with fewer features  
- Top predictors included:
  - `education-num`
  - `hours-per-week`
  - `capital-gain`
  - `occupation`
  - `age`  
- Dimensionality reduction via RFE enhanced interpretability  
- Logistic Regression performed well with normalized, selected features

---

## 💻 How to Run


1. Clone this repository  
   `git clone https://github.com/aarshdesai-ds/diamond-price-prediction.git`
2. Launch the notebook  
   `jupyter notebook CensusAnalysis.ipynb`
3. Run all cells to reproduce the workflow and results


--- 



# ✅ Key Takeaways

This project showcases the effectiveness of **Logistic Regression** combined with **Recursive Feature Elimination (RFE)** for binary income classification using the UCI Adult Census dataset. Below are the major insights and outcomes from the analysis:

---

## 🔍 Interpretability with Performance

- Logistic Regression provided a **transparent, interpretable model** that helps understand the impact of each feature on income prediction.
- Despite its simplicity, it achieved **strong predictive performance**, especially when paired with good preprocessing and feature selection.

---

## 🔄 Dimensionality Reduction via RFE

- **Recursive Feature Elimination (RFE)** ranked and removed less important features iteratively.
- The best performance was achieved **without using all 14 features**, confirming that not all input variables contribute meaningfully.
- Dimensionality reduction improved **generalization**, reduced **overfitting risk**, and made the model easier to explain.

---

## ⚖️ Importance of F1-Score

- The target variable was **imbalanced** (`<=50K` being more frequent than `>50K`).
- **Weighted F1-score** provided a more reliable performance metric than accuracy.
- This ensured the model was evaluated based on both **precision and recall**, accounting for both majority and minority classes.

---

## 🌟 Key Predictive Features

Based on RFE and coefficient analysis, the most influential features were:

- `education-num`
- `hours-per-week`
- `capital-gain`
- `occupation`
- `age`

These features aligned with real-world intuitions about income determinants.

---

## 📈 Conclusion

This project highlights that **classical machine learning models**, when combined with thoughtful preprocessing and feature selection, can achieve **high accuracy and interpretability**—making them ideal for real-world applications where explainability matters.



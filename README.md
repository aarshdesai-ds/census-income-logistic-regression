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

## 🧠 Why Logistic Regression?

Logistic Regression is:
- **Statistically robust** and widely used for binary classification
- **Interpretable**, allowing clear insight into which features contribute to high-income predictions
- Efficient and easy to implement with a strong theoretical foundation

---

## ✂️ Why Recursive Feature Elimination (RFE)?

Real-world datasets often contain **redundant or non-informative features**. RFE works by:

- Fitting the model with all features
- Ranking them by importance (using model coefficients)
- Removing the least important one
- Repeating this process recursively until the desired number of features is selected

This ensures the final model uses only the **most predictive subset**, improving performance and interpretability.

---



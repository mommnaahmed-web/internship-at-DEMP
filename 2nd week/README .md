# Week 2 – Machine Learning Fundamentals

Internship task submissions for **Week 2** of the AI Internship Program. This week's work covers the bias-variance tradeoff, data validation, unsupervised learning, hyperparameter tuning, and ensemble learning techniques.

## Table of Contents

- [Overview](#overview)
- [Tasks](#tasks)
- [Technologies Used](#technologies-used)
- [Learning Outcomes](#learning-outcomes)
- [Author](#author)

## Overview

| | |
|---|---|
| **Program** | AI Internship |
| **Week** | 2 |
| **Focus Areas** | Bias-Variance Tradeoff, Data Validation, Unsupervised Learning, Hyperparameter Tuning, Ensemble Learning |

## Tasks

### 1. Bias-Variance Tradeoff

Studied how model complexity affects underfitting, overfitting, and generalization.

- Explained bias and variance using the dartboard analogy
- Broke down why the tradeoff exists and how it affects model accuracy
- Connected the tradeoff to the curse of dimensionality

### 2. Data Validation Techniques

Compared model validation strategies and diagnosed overfitting/underfitting on a finance dataset.

- Implemented Train-Test Split, K-Fold, Stratified K-Fold, and Repeated Stratified K-Fold validation
- Compared validation strategies side by side on a credit card default risk dataset
- Visualized overfitting vs. underfitting using decision boundaries and learning curves

### 3. Unsupervised Learning

Applied dimensionality reduction and anomaly detection to a finance dataset.

- Implemented PCA for dimensionality reduction and visualization
- Applied Isolation Forest for anomaly detection
- Built a fraud detection use case on simulated credit card transaction data

### 4. Hyperparameter Tuning – KNN & Decision Tree

Tuned classification models and evaluated the effect of hyperparameter choices on a gaming churn dataset.

- Trained baseline KNN and Decision Tree models with default hyperparameters
- Ran manual hyperparameter sweeps for `k` (KNN) and `max_depth` (Decision Tree)
- Used GridSearchCV for systematic hyperparameter tuning

### 5. Ensemble Learning – Bagging & Random Forest

Built and compared ensemble models on a gaming player churn dataset.

- Trained a single Decision Tree as a baseline model
- Implemented a Bagging ensemble of decision trees
- Trained a Random Forest and compared accuracy, F1 score, and feature importance across all three models

## Technologies Used

Python · Jupyter Notebook · Scikit-learn · Pandas · NumPy · Matplotlib

## Learning Outcomes

- Analyzed the bias-variance tradeoff and its effect on model generalization
- Applied data validation techniques including K-Fold and Stratified K-Fold cross-validation
- Diagnosed overfitting and underfitting using decision boundaries and learning curves
- Implemented unsupervised learning techniques, including PCA and Isolation Forest
- Tuned KNN and Decision Tree models using manual sweeps and GridSearchCV
- Implemented ensemble learning methods, including Bagging and Random Forest


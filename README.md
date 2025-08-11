# Predictive Analysis for Breast Cancer Diagnosis with SVMs

## Overview

Self-directed ML project building a **predictive classifier for breast cancer diagnosis** (benign vs malignant) with Support Vector Machines (SVMs). The workflow covers data cleaning, exploratory data analysis, feature ranking and selection, model benchmarking (Gradient Boosted Trees vs SVC), ROC/AUC evaluation, and hyperparameter tuning.

### Dataset Sourced from Kaggle
[Dataset Source Link](https://www.kaggle.com/datasets/yasserh/breast-cancer-dataset/data)


## Features

- EDA with correlation heatmap, pairwise scatterplots, distribution checks — dropped features with high correlation (e.g. dropped `perimeter_*` and `area_*`, kept `radius_*`) to handle multicollinearity
- **Feature selection:** RandomForest-based pruning of low-importance predictors (< 0.01), ranked features by importance
- Standardized feature pipeline for linear/nonlinear models
- Explored Models: **GradientBoostingClassifier** and **SVC** (Support Vector Classifier)
- Evaluation: Test Accuracy, ROC curves and AUC score; Plotted ROC curves for both models
- Hyperparameter tuning: **GridSearchCV** (`C`, `gamma`, `kernel`) and comparison to baseline
- Robustness checks: **RepeatedStratifiedKFold** CV

## Learning Outcomes

- Handling multicollinearity by analysing correlation between features
- Practical feature selection via model-based importances
- Building scikit-learn pipelines for clean preprocessing and training of SVC
- Interpreting ROC/AUC of trained models
- Hyperparameter tuning of SVC

## Model Details

### Models & Config
- **SVM:** `Pipeline(StandardScaler(), SVC())`  
  - Grid search explored `C ∈ {0.1,1,10,100}`, `gamma ∈ {scale, auto, 0.01, 0.1, 1}`, `kernel ∈ {rbf, poly, linear}`
- **Gradient Boosted Trees:** `GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=3, random_state=42)`

## Results

Train/test split: **75/25**. All features standardized for model training.

| Model                  | Test Accuracy | AUC_ROC |
|------------------------|---------------|---------|
| **SVM**                | **98.6014%**  | **0.984320** |
| Gradient Boosted Trees | 95.8042%      | 0.947807 |

### Hyperparameter Tuning Summary
- **GridSearchCV (5-fold)** best CV: `~0.9718 ± 0.0141`; best test score: `~0.9650`
- **Baseline SVM** CV: `~0.9679 ± 0.0139`; test: `~0.9860`  
**Conclusion:** tuning didn’t beat the baseline on the test set; kept the simpler baseline SVM.
**Final Test Accuracy Achieved: 98.6014%**

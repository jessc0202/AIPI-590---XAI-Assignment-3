# AIPI-590---XAI-Assignment-3
# Duke-AI-XAI
This repository is for Duke's AI MEng course, AIPI 590: Emerging Trends in Explainable Artificial Intelligence (XAI). This course is taught by [Dr. Brinnae Bent](https://runsdata.org) and launches in Fall 2025. 

In this repository, you will find example Colab notebooks covering the following topics:
This project implements and compares three key explainable AI techniques: **Partial Dependence Plots (PDP)**, **Individual Conditional Expectation (ICE) plots**, and **Accumulated Local Effects (ALE) plots** for a stroke prediction model. The analysis demonstrates how feature correlations impact model interpretability and why choosing the right visualization method matters for causal interpretation.

### Dataset
- **Source**: Healthcare Stroke Prediction Dataset (Kaggle)
- **Size**: 5,110 patients with 12 features
- **Target**: Binary stroke occurrence (5% positive class - highly imbalanced)
- **Key Features**: Age, glucose level, BMI, hypertension, heart disease, smoking status, etc.


## Key Findings

### 1. Correlation Analysis Reveals Strong Confounding
- **age ↔ ever_married: r=0.6791 (STRONG)** - Creates massive PDP bias
- **age ↔ bmi: r=0.3243 (MODERATE)** - Some confounding effects
- **glucose correlations: all <0.2 (WEAK)** - Minimal bias

### 2. PDP vs ALE Comparison
- **Age**: PDP shows 19x larger effect than ALE due to marriage correlation
- **Glucose**: PDP and ALE patterns match (low correlation validates PDP)
- **BMI**: ALE produces smoother signal by removing age confounding

### 3. Model Performance
- **Algorithm**: Gradient Boosting Classifier (chosen for handling class imbalance and capturing non-linear relationships)
- **ROC-AUC**: ~0.85 (strong discrimination)
- **Feature Importance**: Age > Heart Disease > Hypertension > Glucose > BMI


## Installation
```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install pandas numpy scikit-learn matplotlib seaborn scipy


### Methodology
1. Data Preprocessing

Handle missing BMI values (median imputation)
Encode categorical variables (gender, marital status, work type, etc.)
Feature engineering: Create encoded versions of all categorical features

2. Correlation Analysis

Calculate Pearson correlations between all features
Identify multicollinearity (age-marriage r=0.68)
Assess correlation with target variable

3. Model Training

Gradient Boosting Classifier with parameters tuned for imbalanced data:

n_estimators=100
learning_rate=0.1
max_depth=4
subsample=0.8
min_samples_split=50 (important for rare stroke cases)

### Technical Notes
## Why Manual ALE Implementation?
PyALE library has compatibility issues with sklearn's GradientBoostingClassifier:

PyALE expects specific model interfaces not satisfied by sklearn
DataFrame/array conversion causes column name loss
Binary classification handling is problematic

Solution: Implemented ALE from scratch following the algorithm:

Sort data by feature values
Create bins of similar feature values
Compute local differences within bins
Accumulate effects
Center around zero


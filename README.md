# Student Performance Prediction

Regression pipeline predicting student academic performance from demographic, socioeconomic, and preparation attributes.

[![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-orange?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## Overview

This project demonstrates a complete supervised regression workflow on the [UCI Students Performance dataset](https://www.kaggle.com/datasets/spscientist/students-performance-in-exams). A composite **average score** target is constructed from math, reading, and writing scores, then predicted using only categorical demographic and preparation features — with explicit steps taken to prevent data leakage.

---

## Objectives

1. Apply correct train-test split methodology to prevent target leakage
2. Perform thorough exploratory data analysis on categorical demographic features
3. Build a clean preprocessing pipeline using `ColumnTransformer`
4. Benchmark multiple regression algorithms fairly using cross-validation
5. Package the best model as a serialised `joblib` pipeline artifact

---

## Dataset

| Property | Value |
|---|---|
| **File** | `StudentsPerformance.csv` |
| **Rows** | 1,000 |
| **Original Features** | 8 |
| **Target** | `average_score` — mean of math, reading, and writing scores |
| **Missing Data** | None |

> **Data leakage note**: The three original test score columns are dropped before training. The model predicts performance from background attributes only.

---

## Features Used

| Feature | Type | Notes |
|---|---|---|
| `gender` | Categorical | 2 values |
| `race_ethnicity` | Categorical | 5 groups (A–E) |
| `parental_level_of_education` | Categorical | 6 levels |
| `lunch` | Categorical | standard / free-reduced |
| `test_preparation_course` | Categorical | completed / none |

---

## ML Pipeline

```
Raw CSV
   ↓
Column renaming + target creation (average_score)
   ↓
Drop original score columns (prevent leakage)
   ↓
Train/Test Split (80/20, random_state=42)
   ↓
ColumnTransformer (fitted on X_train only)
   └── OneHotEncoder → 5 categorical features
   ↓
Model training + 5-fold cross-validation
   ↓
Final Pipeline → joblib artifact
```

---

## Models Evaluated

| Model | Notes |
|---|---|
| Baseline (Mean predictor) | Lower bound benchmark |
| Linear Regression | Interpretable baseline |
| Random Forest | Ensemble method |
| Gradient Boosting | Best performer |

**Metrics**: MAE and R² on held-out test set and via 5-fold cross-validation.

---

## Repository Structure

```
.
├── StudentsPerformance.csv              # Raw dataset
├── student_performance_notebook.ipynb  # Full analysis notebook
├── student_performance_model.pkl        # Serialised final pipeline
├── requirements.txt
└── README.md
```

---

## Getting Started

```bash
git clone https://github.com/Sambhu69/Student-Performance-Prediction-Model.git
cd Student-Performance-Prediction-Model
pip install -r requirements.txt
jupyter notebook student_performance_notebook.ipynb
```

### Use the Saved Model

```python
import joblib, pandas as pd

model = joblib.load("student_performance_model.pkl")

sample = pd.DataFrame([{
    "gender": "female",
    "race_ethnicity": "group B",
    "parental_level_of_education": "bachelor's degree",
    "lunch": "standard",
    "test_preparation_course": "completed"
}])

print(f"Predicted average score: {model.predict(sample)[0]:.2f}")
```

---

## License

Released under the [MIT License](LICENSE).
# Heart Disease Prediction

A machine learning project that predicts heart disease using **Logistic Regression** and **Decision Tree** classifiers, then compares their performance.

## Features

- Loads and explores the UCI Heart Disease dataset (`heart.csv`)
- Trains two models: Logistic Regression and Decision Tree
- Evaluates accuracy, precision, recall, F1-score, and ROC-AUC
- Generates comparison visualizations (confusion matrices, ROC curves, feature importance)
- Saves trained models for reuse
- Demonstrates prediction on new patient data

## Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
joblib
```

Install with:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib
```

## Usage

Place `heart.csv` in the project root, then run:

```bash
python heart_model3.py
```

If no dataset is found, sample data is generated automatically.

## Output

Results are saved to the `outputs/` directory:

- `heart_disease_comparison.png` — side-by-side model visualizations
- `logistic_regression_model.pkl` — trained Logistic Regression model
- `decision_tree_model.pkl` — trained Decision Tree model
- `scaler.pkl` — fitted StandardScaler for feature normalization

## Dataset

Expected columns: `age`, `sex`, `cp`, `trestbps`, `chol`, `fbs`, `restecg`, `thalach`, `exang`, `oldpeak`, `slope`, `ca`, `thal`, `target`

The `target` column indicates presence (1) or absence (0) of heart disease.

# Diabetes Prediction Using Data Analytics and Machine Learning

**Author:** Aniruddha Rana

A machine learning project that classifies patients as **Diabetic (Y)**, **Non-Diabetic (N)**, or **Pre-Diabetic (P)** using clinical blood-work features.

---

## Dataset

**Name:** Dataset of Diabetes  
**File:** `Dataset of Diabetes .csv`  
**Shape:** 1,000 rows × 14 columns  
**Target column:** `CLASS` — values: `N` (Non-Diabetic), `P` (Pre-Diabetic), `Y` (Diabetic)

### Features

| Column | Description |
|--------|-------------|
| `ID` | Patient ID (dropped before training) |
| `No_Pation` | Patient number (dropped before training) |
| `Gender` | Patient gender (F / M → encoded 0 / 1) |
| `AGE` | Age in years |
| `Urea` | Blood urea level |
| `Cr` | Creatinine ratio |
| `HbA1c` | Glycated haemoglobin (%) |
| `Chol` | Total cholesterol |
| `TG` | Triglycerides |
| `HDL` | High-density lipoprotein |
| `LDL` | Low-density lipoprotein |
| `VLDL` | Very low-density lipoprotein |
| `BMI` | Body mass index |
| `CLASS` | **Target label** |

---

## Project Description

This project builds a supervised multi-class classification pipeline to predict the diabetes status of a patient from routine blood-test measurements. The full workflow covers:

1. **Exploratory Data Analysis (EDA)** — distribution plots, class balance, correlation heatmap, box-plots
2. **Preprocessing** — drop identifier columns, label-encode `Gender` and `CLASS`
3. **Train / Test Split** — 80 / 20 stratified split
4. **Feature Selection** — Mutual Information scoring on the training set; features above 10 % of the maximum MI score are retained (`HbA1c`, `BMI`, `AGE`, `VLDL`, `TG`, `Urea`)
5. **Model Comparison** — SVM, Random Forest, Logistic Regression evaluated via 5-fold Stratified Cross-Validation with default hyperparameters
6. **Hyperparameter Tuning** — `GridSearchCV` with `StratifiedKFold` (5 folds) inside a `Pipeline` (scaler + classifier) to prevent data leakage
7. **Test-Set Evaluation** — accuracy, precision, recall, F1-score, confusion matrix for all three models
8. **Automatic Model Selection** — the model with the highest test accuracy is selected programmatically
9. **Predictive System** — `predict_diabetes()` function wraps the best pipeline for single-patient inference
10. **Deployment Artefact** — `deployment_pipeline.pkl` bundles the pipeline, label encoder, and selected feature list

### Results

| Model | Best CV Acc | Test Acc | F1 Score |
|-------|------------|----------|----------|
| **Random Forest** | **97.50 %** | **97.50 %** | **97.49 %** |
| SVM | 93.25 % | 97.00 % | 97.02 % |
| Logistic Regression | 91.38 % | 94.00 % | 92.82 % |

**Winner: Random Forest** (`n_estimators=20`)  
**Selected features:** `HbA1c`, `BMI`, `AGE`, `VLDL`, `TG`, `Urea`

---

## Technologies Used

| Library | Purpose |
|---------|---------|
| Python 3.13 | Core language |
| NumPy | Numerical operations |
| Pandas | Data loading and manipulation |
| scikit-learn | ML models, pipelines, feature selection, metrics |
| Seaborn | Statistical visualisation |
| Matplotlib | Plotting |
| Jupyter Notebook | Interactive development environment |
| pickle | Model serialisation |

---

## Project Structure

```
├── AniruddhaRana_Diabetes Prediction Using Data Analytics and Machine Learning.ipynb                          # Main notebook — EDA, training, evaluation
├── Dataset of Diabetes .csv   # Raw dataset
├── deployment_pipeline.pkl    # Saved inference bundle
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

---

## Setup & Run Instructions

### 1. Clone / download the project

```bash
git clone https://github.com/AniruddhaRana04/Diabetes-Prediction-Using-Data-Analytics-and-Machine-Learning.git
cd Diabetes-Prediction-Using-Data-Analytics-and-Machine-Learning
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Launch Jupyter Notebook

```bash
jupyter notebook "AniruddhaRana_Diabetes Prediction Using Data Analytics and Machine Learning.ipynb"
```

Run all cells top-to-bottom (**Kernel → Restart & Run All**).

### 5. Use the saved model for inference

```python
import pickle, pandas as pd

bundle   = pickle.load(open("deployment_pipeline.pkl", "rb"))
pipeline = bundle["pipeline"]           # Pipeline: StandardScaler + RandomForest
le       = bundle["label_encoder"]      # LabelEncoder (N / P / Y)
features = bundle["selected_features"]  # ['HbA1c', 'BMI', 'AGE', 'VLDL', 'TG', 'Urea']

# Example patient
input_df = pd.DataFrame([[9.9, 28.0, 61.0, 1.1, 4.2, 2.1]], columns=features)
encoded  = pipeline.predict(input_df)[0]
label    = le.inverse_transform([encoded])[0]   # 'Y' → Diabetic
print(label)
```

---

## Key Design Decisions

- **`StandardScaler` is inside every Pipeline** so scaling is re-fitted on each training fold during CV — no data leakage.
- **Feature selection uses only the training set** (`mutual_info_classif` on `X_train` / `Y_train`).
- **`GridSearchCV` selects hyperparameters automatically** — nothing is hard-coded.
- **Best model is selected programmatically** by highest test-set accuracy.
- **`deployment_pipeline.pkl` is the single inference artefact** — it contains the scaler, classifier, encoder, and feature list so the correct pre-processing is always applied.

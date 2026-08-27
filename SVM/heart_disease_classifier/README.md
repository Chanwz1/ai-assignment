# Heart Disease Classification with SVM

This assignment trains and serves one RBF Support Vector Machine (SVM) for binary heart-disease classification. Training, the saved pipeline, Streamlit, the CLI, and tests share one schema from `project_config.py`.

> This is an academic prototype built from synthetic data. It is not a medical device and must not replace professional diagnosis or advice.

## Current data and feature contract

The only dataset is `synthetic_heart_disease_dataset.csv`:

- 50,000 rows
- 20 raw input columns
- binary target `Heart_Disease` (`0` = absent, `1` = present)
- target counts: 26,827 absent and 23,173 present
- `Alcohol_Intake="None"` is a real category, not a missing value

The model receives these 19 raw fields, in this exact order:

```text
Age, Gender, Weight, Height, Smoking, Alcohol_Intake,
Physical_Activity, Diet, Stress_Level, Hypertension, Diabetes,
Hyperlipidemia, Family_History, Previous_Heart_Attack,
Systolic_BP, Diastolic_BP, Heart_Rate, Blood_Sugar_Fasting,
Cholesterol_Total
```

`BMI` remains in the source data but is excluded from the fitted model. It is inconsistent with the value recalculated from weight and height, and a training-only comparison found that removing it performed slightly better than retaining either version.

## Model and validation

`train.py` uses an 80/20 stratified split with `random_state=42`. The test partition is held back until all feature and parameter decisions are complete.

The saved scikit-learn pipeline contains:

- median imputation and `StandardScaler` for numerical fields
- constant imputation and `OneHotEncoder(handle_unknown="ignore")` for categorical fields
- `SVC(kernel="rbf", C=20, gamma=0.1)`

Preprocessing is fitted inside each cross-validation fold. Sixteen SVM parameter candidates are evaluated with five-fold stratified cross-validation on a fixed 12,000-row training-only subset. The selected pipeline is then cross-validated on all 40,000 training rows, fitted once, and evaluated once on the 10,000-row test set.

Current saved results:

| Metric | Value |
|---|---:|
| Mean five-fold training CV accuracy | 95.95% |
| Test accuracy | 96.15% |
| Test precision | 95.74% |
| Test recall | 95.97% |
| Test F1 | 95.85% |
| Test ROC-AUC | 0.9954 |

The high result reflects a deterministic pattern in this synthetic dataset. A direct audit found that every training label matches a five-condition rule documented in `static/metrics.json`; this is a limitation of the dataset, not target leakage. `Heart_Disease` is excluded from model inputs, preprocessing, and prediction forms.

## Install and run

From this directory:

```bash
pip install -r requirements.txt
python -m streamlit run app.py
```

The compatibility-critical dependency is pinned to `scikit-learn==1.9.0`, matching the version used to create the saved pipeline. The remaining minimum versions are recorded in `requirements.txt`: pandas 2.0, Streamlit 1.25, Matplotlib 3.7, Seaborn 0.12, and joblib 1.3.

The Streamlit app provides dataset overview, exploratory analysis, saved SVM performance, and a 19-field prediction form.

To run the terminal interface instead:

```bash
cd ../heart_disease_cli
python main.py
```

## Retraining and verification

Retraining is intentionally explicit because it takes several minutes and replaces the saved artifacts:

```bash
python train.py
```

It updates:

- `models/svm_pipeline.joblib`
- `static/metrics.json`
- `static/svm_confusion_matrix.png`

Run the regression and inference checks with:

```bash
python -B -m unittest -v test_project_integrity.py
python -B test_inference.py
```

The integrity tests verify dataset identity, preservation of the alcohol category, schema alignment, leakage controls, SVM-only training code, fitted preprocessing, metric provenance, and the minimum 80% accuracy requirement.

## Project files

```text
heart_disease_classifier/
|-- app.py
|-- build_report_artifact.py
|-- project_config.py
|-- synthetic_heart_disease_dataset.csv
|-- train.py
|-- test_inference.py
|-- test_project_integrity.py
|-- models/svm_pipeline.joblib
|-- static/metrics.json
|-- static/svm_confusion_matrix.png
|-- report_artifact.json
`-- svm_analysis_report.html
```

`svm_analysis_report.html` is the portable technical report generated from `report_artifact.json` and the current saved metrics.

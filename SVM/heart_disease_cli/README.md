# Heart Disease SVM CLI

This terminal interface uses the dataset, schema, fitted preprocessing, SVM pipeline, and evaluation artifacts from the sibling `heart_disease_classifier` directory. It does not keep separate copies.

> This is an academic prototype based on synthetic data, not a medical diagnostic system.

## Shared assets

The CLI reads:

- `../heart_disease_classifier/synthetic_heart_disease_dataset.csv`
- `../heart_disease_classifier/project_config.py`
- `../heart_disease_classifier/models/svm_pipeline.joblib`
- `../heart_disease_classifier/static/metrics.json`
- `../heart_disease_classifier/static/svm_confusion_matrix.png`

The source contains 50,000 rows and target `Heart_Disease`. The prediction form collects the exact 19 fields expected by the saved pipeline; `BMI` and the target are not accepted. `Alcohol_Intake="None"` is preserved as a valid category.

## Menu

1. Overview Dashboard
2. Exploratory Data Analysis
3. SVM Model Performance
4. Heart Disease Classification Prediction
5. Model Training
6. Exit

Predictions are displayed as a binary classification. The CLI does not present the SVM decision score as a probability.

## Install and run

From this directory:

```bash
pip install -r requirements.txt
python main.py
```

Paths are resolved relative to the source files, so the CLI can also be launched from another working directory.

## Current saved evaluation

The current shared model uses `SVC(kernel="rbf", C=20, gamma=0.1)` and achieved:

| Metric | Value |
|---|---:|
| Mean five-fold training CV accuracy | 95.95% |
| Test accuracy | 96.15% |
| Test precision | 95.74% |
| Test recall | 95.97% |
| Test F1 | 95.85% |
| Test ROC-AUC | 0.9954 |

The result is unusually high because the synthetic target follows a deterministic pattern documented in the shared metrics and report. It should not be interpreted as clinical performance.

## Retraining

Menu option 5 can explicitly launch the sibling `train.py`. Training takes several minutes and replaces the shared model, metrics, and confusion-matrix image. It is never triggered during normal startup or prediction.

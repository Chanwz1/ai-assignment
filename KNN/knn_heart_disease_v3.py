"""
====================================================================
 Heart Disease Prediction using K-Nearest Neighbours (KNN)
 AI Assignment - Title 1: Machine Learning (Supervised)
====================================================================

Dataset:
    synthetic_heart_disease_dataset.csv (50,000 patients, 20 features
    + target)
    Target column: "Heart_Disease" (0 = No, 1 = Yes)

Dataset characteristics:
    - Large: 50,000 rows -> plenty of data for KNN to learn from
    - Fairly balanced target: 26,827 No (53.7%) vs 23,173 Yes (46.3%)
    - One column with real missing data: Alcohol_Intake (40.2% missing)
    - 6 categorical text columns need encoding
    - Strong real signal: Hypertension, Age, Cholesterol_Total,
      Diabetes and Previous_Heart_Attack all correlate meaningfully
      with the target (unlike the earlier heart_disease.csv dataset,
      which turned out to have no real signal at all)

Pipeline:
    1. Load & inspect data
    2. Clean data (remove duplicates, remove physiologically invalid
       readings such as Systolic_BP <= Diastolic_BP)
    3. Encode categorical text columns into numbers
    4. Exploratory Data Analysis (EDA) -> saved as PNG charts
    5. Train/test split (BEFORE imputing/scaling, to avoid data leakage)
    6. Impute missing values (Alcohol_Intake only, using train statistics)
    7. Scale features (required for KNN, a distance-based algorithm)
    8. Hyperparameter tuning (finding best K via cross-validation)
    9. Train final KNN model
   10. Evaluate: accuracy, precision, recall, F1, confusion matrix, ROC-AUC
   11. Predict on a new / unseen patient
====================================================================
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, roc_auc_score
)

RANDOM_STATE = 42
OUT_DIR = "outputs_knn_v3"
os.makedirs(OUT_DIR, exist_ok=True)

# --------------------------------------------------------------
# 1. LOAD DATA
# --------------------------------------------------------------
df = pd.read_csv("synthetic_heart_disease_dataset.csv")
print("Dataset shape (raw):", df.shape)
print(df.head())
print("\nMissing values per column:\n", df.isnull().sum())
print("\nDuplicate rows:", df.duplicated().sum())

# --------------------------------------------------------------
# 2. DATA CLEANING
# --------------------------------------------------------------
# 2a. Remove exact duplicate rows (none found in this dataset, but
#     always worth checking - see the earlier heart.csv prototype
#     where 723 duplicates caused data leakage).
n_before = len(df)
df = df.drop_duplicates().reset_index(drop=True)
print(f"\nRemoved {n_before - len(df)} duplicate rows.")

# 2b. Remove physiologically IMPOSSIBLE readings.
# Systolic blood pressure (the "top" number) must always be higher
# than diastolic (the "bottom" number) in a real reading. Rows that
# violate this are either data entry errors or corrupted synthetic
# records, and would teach the model a false pattern if kept.
invalid_bp = df["Systolic_BP"] <= df["Diastolic_BP"]
print(f"Rows with Systolic_BP <= Diastolic_BP (invalid): {invalid_bp.sum()} "
      f"({invalid_bp.mean():.1%} of data)")
df = df[~invalid_bp].reset_index(drop=True)

# 2c. Sanity-check other numeric ranges for implausible values
# (none found here, but this is standard practice to verify).
range_checks = {
    "Age": (18, 120),
    "Heart_Rate": (30, 220),
    "BMI": (10, 70),
}
for col, (lo, hi) in range_checks.items():
    n_bad = ((df[col] < lo) | (df[col] > hi)).sum()
    print(f"{col} outside plausible range [{lo}, {hi}]: {n_bad} rows")

print(f"\nDataset shape after cleaning: {df.shape}")

# --------------------------------------------------------------
# SAFETY CHECK: target must have no missing values
# --------------------------------------------------------------
# This guards against a common Jupyter mistake: if a cell above is
# re-run twice (e.g. the encoding cell runs .map() on an already-
# encoded column), values that can no longer be matched become NaN.
# scikit-learn's train_test_split will fail with "Input y contains
# NaN" if the target has any missing values. We check and fix this
# defensively here, and also recommend using "Restart & Run All"
# if you see unexpected NaNs anywhere in this notebook.
if df["Heart_Disease"].isnull().any():
    n_missing_target = df["Heart_Disease"].isnull().sum()
    print(f"\nWARNING: {n_missing_target} rows have a missing target value "
          f"- dropping them. If this number looks large or unexpected, "
          f"restart the kernel and run all cells from the top in order "
          f"(a cell may have been re-run out of sequence).")
    df = df.dropna(subset=["Heart_Disease"]).reset_index(drop=True)

# --------------------------------------------------------------
# 3. ENCODE CATEGORICAL COLUMNS
# --------------------------------------------------------------
# Gender -> binary
df["Gender"] = df["Gender"].map({"Male": 1, "Female": 0})

# Ordinal columns -> there is a natural order, so we map to 0/1/2
# rather than one-hot encoding (keeps the feature count small and
# preserves the "more/less" relationship for KNN's distance metric)
df["Smoking"] = df["Smoking"].map({"Never": 0, "Former": 1, "Current": 2})
df["Alcohol_Intake"] = df["Alcohol_Intake"].map({"Low": 0, "Moderate": 1, "High": 2})
df["Physical_Activity"] = df["Physical_Activity"].map({"Sedentary": 0, "Moderate": 1, "Active": 2})
df["Diet"] = df["Diet"].map({"Unhealthy": 0, "Average": 1, "Healthy": 2})
df["Stress_Level"] = df["Stress_Level"].map({"Low": 0, "Medium": 1, "High": 2})

print("\nAfter encoding, all columns are numeric:\n", df.dtypes)

# Safety check: only Alcohol_Intake should have missing values at this
# point (40.2%, from the raw data). If other columns suddenly show
# missing values too, the encoding cell was likely run more than once
# on already-encoded data (e.g. .map() looking for "Male"/"Female"
# text in a column that's already 1/0 produces NaN for every row).
post_encode_missing = df.isnull().sum()
unexpected_missing = post_encode_missing[post_encode_missing.index != "Alcohol_Intake"]
unexpected_missing = unexpected_missing[unexpected_missing > 0]
if len(unexpected_missing) > 0:
    print("\nWARNING: unexpected missing values found after encoding "
          "(this usually means a cell was run twice). Restart the "
          "kernel and run all cells from the top in order:")
    print(unexpected_missing)

# --------------------------------------------------------------
# 4. EXPLORATORY DATA ANALYSIS (EDA)
# --------------------------------------------------------------
plt.figure(figsize=(5, 4))
sns.countplot(x="Heart_Disease", hue="Heart_Disease",
              data=df, palette=["#4C72B0", "#DD8452"], legend=False)
plt.title("Distribution of Heart Disease (0 = No, 1 = Yes)")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/01_target_distribution.png", dpi=150)
plt.close()
print("\nClass balance:\n", df["Heart_Disease"].value_counts(normalize=True))

plt.figure(figsize=(14, 11))
sns.heatmap(df.corr(), annot=False, cmap="coolwarm", square=True)
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/02_correlation_heatmap.png", dpi=150)
plt.close()

# Bar chart of each feature's correlation with the target - useful
# for the documentation's "Related Work" / "Results" discussion
target_corr = df.corr()["Heart_Disease"].drop("Heart_Disease").sort_values()
plt.figure(figsize=(7, 7))
target_corr.plot(kind="barh", color=["#DD8452" if v < 0 else "#4C72B0" for v in target_corr])
plt.title("Feature Correlation with Heart Disease")
plt.xlabel("Correlation Coefficient")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/03_target_correlation.png", dpi=150)
plt.close()

plt.figure(figsize=(6, 4))
sns.histplot(data=df, x="Age", hue="Heart_Disease", multiple="stack",
             bins=20, palette=["#4C72B0", "#DD8452"])
plt.title("Age Distribution by Heart Disease Status")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/04_age_distribution.png", dpi=150)
plt.close()

# --------------------------------------------------------------
# 5. TRAIN / TEST SPLIT (before imputing, to avoid data leakage)
# --------------------------------------------------------------
X = df.drop(columns=["Heart_Disease"])
y = df["Heart_Disease"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)
print(f"\nTrain size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")

# --------------------------------------------------------------
# 6. IMPUTE MISSING VALUES
# --------------------------------------------------------------
# Only Alcohol_Intake has missing values (40.2%). We fit the imputer
# on TRAINING data only, then apply the same learned median to test
# data, to avoid leaking test-set information into training.
imputer = SimpleImputer(strategy="median")
X_train_imputed = pd.DataFrame(
    imputer.fit_transform(X_train), columns=X_train.columns, index=X_train.index
)
X_test_imputed = pd.DataFrame(
    imputer.transform(X_test), columns=X_test.columns, index=X_test.index
)
print("\nMissing values after imputation (train):", X_train_imputed.isnull().sum().sum())

# --------------------------------------------------------------
# 7. SCALE FEATURES
# --------------------------------------------------------------
# KNN is distance-based -> features must be scaled or large-range
# features (e.g. Cholesterol_Total ~150-350) will dominate small-range
# ones (e.g. Gender 0-1).
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_imputed)
X_test_scaled = scaler.transform(X_test_imputed)

# --------------------------------------------------------------
# 8. HYPERPARAMETER TUNING - find best K
# --------------------------------------------------------------
k_range = range(1, 31, 2)  # odd K only, avoids tie votes
cv_scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train_scaled, y_train, cv=5, scoring="f1")
    cv_scores.append(scores.mean())

best_k = list(k_range)[int(np.argmax(cv_scores))]
print(f"\nBest K found via 5-fold CV: {best_k} (CV F1 = {max(cv_scores):.4f})")

plt.figure(figsize=(7, 4))
plt.plot(list(k_range), cv_scores, marker="o", color="#4C72B0")
plt.axvline(best_k, color="red", linestyle="--", label=f"Best K = {best_k}")
plt.title("KNN Cross-Validation F1 Score vs K")
plt.xlabel("K (Number of Neighbours)")
plt.ylabel("Cross-Validated F1 Score")
plt.legend()
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/05_k_tuning.png", dpi=150)
plt.close()

# --------------------------------------------------------------
# 9. TRAIN FINAL MODEL
# --------------------------------------------------------------
final_model = KNeighborsClassifier(n_neighbors=best_k)
final_model.fit(X_train_scaled, y_train)
y_pred = final_model.predict(X_test_scaled)
y_proba = final_model.predict_proba(X_test_scaled)[:, 1]

# --------------------------------------------------------------
# 10. EVALUATION
# --------------------------------------------------------------
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_proba)

report = classification_report(y_test, y_pred, target_names=["No Disease", "Disease"])
cm = confusion_matrix(y_test, y_pred)

print("\n================ RESULTS ================")
print(f"Best K            : {best_k}")
print(f"Accuracy           : {acc:.4f}")
print(f"Precision          : {prec:.4f}")
print(f"Recall             : {rec:.4f}")
print(f"F1 Score           : {f1:.4f}")
print(f"ROC-AUC            : {auc:.4f}")
print("\nClassification Report:\n", report)
print("Confusion Matrix:\n", cm)

plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["No Disease", "Disease"],
            yticklabels=["No Disease", "Disease"])
plt.title(f"Confusion Matrix (KNN, K={best_k})")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/06_confusion_matrix.png", dpi=150)
plt.close()

fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.figure(figsize=(5, 4))
plt.plot(fpr, tpr, label=f"KNN (AUC = {auc:.3f})", color="#4C72B0")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.title("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/07_roc_curve.png", dpi=150)
plt.close()

# --------------------------------------------------------------
# 11. SAVE TEXT REPORT
# --------------------------------------------------------------
with open(f"{OUT_DIR}/results_report.txt", "w") as f:
    f.write("HEART DISEASE PREDICTION - KNN MODEL RESULTS (synthetic_heart_disease_dataset.csv)\n")
    f.write("=" * 70 + "\n")
    f.write(f"Best K (from 5-fold CV, optimised for F1): {best_k}\n")
    f.write(f"Accuracy : {acc:.4f}\n")
    f.write(f"Precision: {prec:.4f}\n")
    f.write(f"Recall   : {rec:.4f}\n")
    f.write(f"F1 Score : {f1:.4f}\n")
    f.write(f"ROC-AUC  : {auc:.4f}\n\n")
    f.write("Classification Report:\n")
    f.write(report + "\n")
    f.write("Confusion Matrix:\n")
    f.write(str(cm) + "\n")

print(f"\nAll charts and report saved to '{OUT_DIR}/' folder.")

# --------------------------------------------------------------
# 12. PREDICT ON A NEW / UNSEEN PATIENT
# --------------------------------------------------------------
def predict_new_patient(patient_dict):
    """
    patient_dict: a dictionary with the RAW (un-encoded) feature values,
    using the same text categories as the original CSV, e.g.:
        {
            "Age": 60, "Gender": "Male", "Weight": 90, "Height": 172,
            "BMI": 30.4, "Smoking": "Current", "Alcohol_Intake": "High",
            "Physical_Activity": "Sedentary", "Diet": "Unhealthy",
            "Stress_Level": "High", "Hypertension": 1, "Diabetes": 1,
            "Hyperlipidemia": 1, "Family_History": 1,
            "Previous_Heart_Attack": 0, "Systolic_BP": 150,
            "Diastolic_BP": 95, "Heart_Rate": 90,
            "Blood_Sugar_Fasting": 140, "Cholesterol_Total": 260
        }
    Returns: (predicted_label, probability_of_disease)
    """
    p = pd.DataFrame([patient_dict])
    p["Gender"] = p["Gender"].map({"Male": 1, "Female": 0})
    p["Smoking"] = p["Smoking"].map({"Never": 0, "Former": 1, "Current": 2})
    p["Alcohol_Intake"] = p["Alcohol_Intake"].map({"Low": 0, "Moderate": 1, "High": 2})
    p["Physical_Activity"] = p["Physical_Activity"].map({"Sedentary": 0, "Moderate": 1, "Active": 2})
    p["Diet"] = p["Diet"].map({"Unhealthy": 0, "Average": 1, "Healthy": 2})
    p["Stress_Level"] = p["Stress_Level"].map({"Low": 0, "Medium": 1, "High": 2})
    p = p[X.columns]  # keep same column order
    p_imputed = pd.DataFrame(imputer.transform(p), columns=X.columns)
    p_scaled = scaler.transform(p_imputed)
    pred = final_model.predict(p_scaled)[0]
    proba = final_model.predict_proba(p_scaled)[0][1]
    return pred, proba


example_patient = {
    "Age": 60, "Gender": "Male", "Weight": 90, "Height": 172,
    "BMI": 30.4, "Smoking": "Current", "Alcohol_Intake": "High",
    "Physical_Activity": "Sedentary", "Diet": "Unhealthy",
    "Stress_Level": "High", "Hypertension": 1, "Diabetes": 1,
    "Hyperlipidemia": 1, "Family_History": 1,
    "Previous_Heart_Attack": 0, "Systolic_BP": 150,
    "Diastolic_BP": 95, "Heart_Rate": 90,
    "Blood_Sugar_Fasting": 140, "Cholesterol_Total": 260
}
pred_label, pred_proba = predict_new_patient(example_patient)

print("\n================ NEW PATIENT PREDICTION ================")
print("Patient data:", example_patient)
print(f"Prediction: {'Heart Disease' if pred_label == 1 else 'No Heart Disease'}")
print(f"Probability of heart disease: {pred_proba:.2%}")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, roc_auc_score
)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# -----------------------------
# 1. Load data
# -----------------------------
DATA_PATH = "synthetic_heart_disease_dataset.csv"

try:
    df = pd.read_csv(DATA_PATH)
except FileNotFoundError:
    raise SystemExit(
        f"ERROR: '{DATA_PATH}' not found. "
        "Make sure the CSV is in the same folder as this script."
    )
except pd.errors.EmptyDataError:
    raise SystemExit(f"ERROR: '{DATA_PATH}' is empty or corrupted.")
except pd.errors.ParserError as e:
    raise SystemExit(f"ERROR: Could not parse '{DATA_PATH}' as CSV. Details: {e}")

if df.empty:
    raise SystemExit("ERROR: Loaded dataset has 0 rows. Check the CSV file.")

REQUIRED_COLUMNS = [
    "Age", "Gender", "Weight", "Height", "BMI", "Smoking", "Alcohol_Intake",
    "Physical_Activity", "Diet", "Stress_Level", "Hypertension", "Diabetes",
    "Hyperlipidemia", "Family_History", "Previous_Heart_Attack", "Systolic_BP",
    "Diastolic_BP", "Heart_Rate", "Blood_Sugar_Fasting", "Cholesterol_Total",
    "Heart_Disease",
]
missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
if missing_cols:
    raise SystemExit(
        f"ERROR: Dataset is missing required columns: {missing_cols}. "
        f"Found columns: {list(df.columns)}"
    )

print("Dataset shape:", df.shape)
print(df.head())
print("\nMissing values per column:\n", df.isnull().sum()[df.isnull().sum() > 0])
print("\nTarget distribution:\n", df["Heart_Disease"].value_counts(normalize=True))

# -----------------------------
# 2. Encode categorical text columns
# -----------------------------
GENDER_MAP = {"Male": 1, "Female": 0}
SMOKING_MAP = {"Never": 0, "Former": 1, "Current": 2}
ALCOHOL_MAP = {"None": 0, "Low": 1, "Moderate": 2, "High": 3}
ACTIVITY_MAP = {"Sedentary": 0, "Moderate": 1, "Active": 2}
DIET_MAP = {"Unhealthy": 0, "Average": 1, "Healthy": 2}
STRESS_MAP = {"Low": 0, "Medium": 1, "High": 2}


def encode_categorical_columns(data: pd.DataFrame) -> pd.DataFrame:
    """Encode text category columns to numbers. Raises ValueError if an
    unexpected category value is found (helps catch typos/bad data early)."""
    data = data.copy()
    # Alcohol_Intake's blanks mean "does not drink" — fill before mapping.
    if "Alcohol_Intake" in data.columns:
        data["Alcohol_Intake"] = data["Alcohol_Intake"].fillna("None")
    mappings = {
        "Gender": GENDER_MAP,
        "Smoking": SMOKING_MAP,
        "Alcohol_Intake": ALCOHOL_MAP,
        "Physical_Activity": ACTIVITY_MAP,
        "Diet": DIET_MAP,
        "Stress_Level": STRESS_MAP,
    }
    for col, mapping in mappings.items():
        if col not in data.columns:
            continue
        unknown = set(data[col].dropna().unique()) - set(mapping.keys())
        if unknown:
            raise ValueError(
                f"Column '{col}' contains unexpected category values {unknown}. "
                f"Expected one of {list(mapping.keys())}."
            )
        data[col] = data[col].map(mapping)
    return data


try:
    df = encode_categorical_columns(df)
except ValueError as e:
    raise SystemExit(f"ERROR while encoding categorical columns: {e}")

print("\nAfter encoding, dtypes:\n", df.dtypes)
print("\nMissing values remaining after encoding:", df.isnull().sum().sum())

# -----------------------------
# 3. Correlation check (sanity check the data has real signal)
# -----------------------------
try:
    plt.figure(figsize=(12, 10))
    sns.heatmap(df.corr(numeric_only=True), cmap="coolwarm", annot=False, square=True)
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig("correlation_heatmap.png", dpi=150)
    plt.close()
    print("\nSaved correlation_heatmap.png")
except Exception as e:
    print(f"WARNING: Could not generate correlation heatmap: {e}")

# -----------------------------
# 4. Train/test split (BEFORE scaling — avoids data leakage)
# -----------------------------
if "Heart_Disease" not in df.columns:
    raise SystemExit("ERROR: Target column 'Heart_Disease' not found in dataset.")

X = df.drop(columns=["Heart_Disease"])
y = df["Heart_Disease"]
feature_names = X.columns.tolist()

if y.isnull().any():
    raise SystemExit("ERROR: Target column 'Heart_Disease' contains missing values.")

if y.nunique() < 2:
    raise SystemExit("ERROR: Target column has only one class — cannot train a classifier.")

try:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
except ValueError as e:
    raise SystemExit(f"ERROR during train/test split: {e}")

print(f"\nTrain size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")

# -----------------------------
# 5. Scale features
# -----------------------------
if X_train.isnull().sum().sum() > 0 or X_test.isnull().sum().sum() > 0:
    raise SystemExit(
        "ERROR: Unexpected missing values remain after categorical encoding. "
        "Check the encode_categorical_columns() mappings."
    )

try:
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
except ValueError as e:
    raise SystemExit(f"ERROR during feature scaling: {e}")

# -----------------------------
# 6. Hyperparameter tuning: compare ANN architectures via 5-fold CV
# -----------------------------
print("\n--- Hyperparameter tuning (this may take a minute) ---")
architectures = [(8,), (16,), (32,), (16, 8), (32, 16), (64, 32)]
cv_scores = []
for arch in architectures:
    try:
        mlp = MLPClassifier(
            hidden_layer_sizes=arch, activation="relu", solver="adam",
            max_iter=300, random_state=RANDOM_STATE, early_stopping=True,
        )
        scores = cross_val_score(mlp, X_train_scaled, y_train, cv=5, scoring="f1")
        cv_scores.append(scores.mean())
        print(f"Architecture {arch}: CV F1 = {scores.mean():.4f}")
    except Exception as e:
        print(f"WARNING: Architecture {arch} failed during CV ({e}); skipping.")
        cv_scores.append(-1.0)  # so it's never picked as "best"

if max(cv_scores) < 0:
    raise SystemExit("ERROR: All architectures failed during hyperparameter tuning.")

best_idx = int(np.argmax(cv_scores))
best_arch = architectures[best_idx]
print(f"\nBest architecture: {best_arch} (CV F1 = {max(cv_scores):.4f})")

plt.figure(figsize=(8, 4))
plt.bar([str(a) for a in architectures], cv_scores, color="#4C72B0")
plt.title("ANN Cross-Validation F1 Score vs Hidden Layer Architecture")
plt.xlabel("Hidden Layer Sizes")
plt.ylabel("Cross-Validated F1 Score")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig("hyperparameter_tuning.png", dpi=150)
plt.close()
print("Saved hyperparameter_tuning.png")

# -----------------------------
# 7. Train final model with best architecture
# -----------------------------
final_model = MLPClassifier(
    hidden_layer_sizes=best_arch, activation="relu", solver="adam",
    max_iter=500, random_state=RANDOM_STATE, early_stopping=True,
)
final_model.fit(X_train_scaled, y_train)

y_pred = final_model.predict(X_test_scaled)
y_proba = final_model.predict_proba(X_test_scaled)[:, 1]


# -----------------------------
# 8. Evaluation
# -----------------------------
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_proba)

print("\n===== Test Set Performance =====")
print(f"Accuracy  : {acc:.4f}")
print(f"Precision : {prec:.4f}")
print(f"Recall    : {rec:.4f}")
print(f"F1 Score  : {f1:.4f}")
print(f"ROC-AUC   : {auc:.4f}")
print("\nClassification Report:\n",
      classification_report(y_test, y_pred, target_names=["No Disease", "Disease"]))

# Confusion matrix + ROC curve
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["No Disease", "Disease"],
            yticklabels=["No Disease", "Disease"], ax=axes[0])
axes[0].set_title("ANN Confusion Matrix")
axes[0].set_xlabel("Predicted")
axes[0].set_ylabel("Actual")

fpr, tpr, _ = roc_curve(y_test, y_proba)
axes[1].plot(fpr, tpr, label=f"AUC = {auc:.3f}", color="#4C72B0")
axes[1].plot([0, 1], [0, 1], linestyle="--", color="gray")
axes[1].set_title("ROC Curve")
axes[1].set_xlabel("False Positive Rate")
axes[1].set_ylabel("True Positive Rate")
axes[1].legend()

plt.tight_layout()
plt.savefig("ann_results.png", dpi=150)
plt.close()
print("Saved ann_results.png")

# -----------------------------
# 9. Save model + preprocessing objects for reuse (e.g. in the Streamlit UI)
# -----------------------------
try:
    joblib.dump(final_model, "heart_disease_ann_model.pkl")
    joblib.dump(scaler, "scaler.pkl")
    joblib.dump(feature_names, "feature_names.pkl")
    print("\nModel and preprocessing objects saved.")
except Exception as e:
    print(f"WARNING: Could not save model/preprocessing objects: {e}")
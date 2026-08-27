"""Single source of truth for the current heart-disease dataset and model schema."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "synthetic_heart_disease_dataset.csv"
MODEL_PATH = ROOT / "models" / "svm_pipeline.joblib"
METRICS_PATH = ROOT / "static" / "metrics.json"
CONFUSION_MATRIX_PATH = ROOT / "static" / "svm_confusion_matrix.png"
TARGET = "Heart_Disease"

CATEGORICAL_FEATURES = [
    "Gender",
    "Smoking",
    "Alcohol_Intake",
    "Physical_Activity",
    "Diet",
    "Stress_Level",
]

DATA_NUMERICAL_FEATURES = [
    "Age",
    "Weight",
    "Height",
    "BMI",
    "Hypertension",
    "Diabetes",
    "Hyperlipidemia",
    "Family_History",
    "Previous_Heart_Attack",
    "Systolic_BP",
    "Diastolic_BP",
    "Heart_Rate",
    "Blood_Sugar_Fasting",
    "Cholesterol_Total",
]

# BMI is deliberately excluded from the fitted model. In this CSV it is almost
# independent of Weight / Height**2, and training-only CV showed that removing it
# was slightly better than either retaining it or replacing it with recalculated BMI.
REMOVED_FEATURES = {"BMI": "Inconsistent with Weight and Height; redundant after recalculation."}
NUMERICAL_FEATURES = [name for name in DATA_NUMERICAL_FEATURES if name not in REMOVED_FEATURES]
MODEL_FEATURES = [
    "Age",
    "Gender",
    "Weight",
    "Height",
    "Smoking",
    "Alcohol_Intake",
    "Physical_Activity",
    "Diet",
    "Stress_Level",
    "Hypertension",
    "Diabetes",
    "Hyperlipidemia",
    "Family_History",
    "Previous_Heart_Attack",
    "Systolic_BP",
    "Diastolic_BP",
    "Heart_Rate",
    "Blood_Sugar_Fasting",
    "Cholesterol_Total",
]
DATA_COLUMNS = [
    "Age",
    "Gender",
    "Weight",
    "Height",
    "BMI",
    "Smoking",
    "Alcohol_Intake",
    "Physical_Activity",
    "Diet",
    "Stress_Level",
    "Hypertension",
    "Diabetes",
    "Hyperlipidemia",
    "Family_History",
    "Previous_Heart_Attack",
    "Systolic_BP",
    "Diastolic_BP",
    "Heart_Rate",
    "Blood_Sugar_Fasting",
    "Cholesterol_Total",
    TARGET,
]

CATEGORY_OPTIONS = {
    "Gender": ["Female", "Male"],
    "Smoking": ["Never", "Former", "Current"],
    "Alcohol_Intake": ["None", "Low", "Moderate", "High"],
    "Physical_Activity": ["Sedentary", "Moderate", "Active"],
    "Diet": ["Unhealthy", "Average", "Healthy"],
    "Stress_Level": ["Low", "Medium", "High"],
}


def load_dataset() -> pd.DataFrame:
    """Load and validate only the current CSV while preserving alcohol 'None'."""
    if not DATA_PATH.is_file():
        raise FileNotFoundError(f"Required dataset not found: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH, keep_default_na=False)
    if df.columns.tolist() != DATA_COLUMNS:
        raise ValueError(
            "Unexpected CSV schema. "
            f"Expected {DATA_COLUMNS}; found {df.columns.tolist()}"
        )
    if len(df) != 50_000:
        raise ValueError(f"Expected 50,000 records, found {len(df):,}")
    if set(df[TARGET].unique()) != {0, 1}:
        raise ValueError(f"{TARGET} must contain only integer labels 0 and 1")
    if df["Alcohol_Intake"].isna().any() or "None" not in set(df["Alcohol_Intake"]):
        raise ValueError("Alcohol_Intake category 'None' was not preserved")
    return df


def model_input(df: pd.DataFrame) -> pd.DataFrame:
    """Return raw inputs in the exact order expected by the saved pipeline."""
    missing = [name for name in MODEL_FEATURES if name not in df.columns]
    if missing:
        raise ValueError(f"Missing model input features: {missing}")
    if TARGET in df.columns:
        raise ValueError(f"Target leakage guard: {TARGET} must not be passed to inference")
    return df.loc[:, MODEL_FEATURES]

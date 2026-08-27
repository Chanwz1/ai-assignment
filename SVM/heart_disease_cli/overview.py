"""Dataset and SVM project overview."""

from utils import (
    CATEGORICAL_FEATURES, NUMERICAL_FEATURES, STATUS_COLUMN,
    banner, load_dataset, press_enter, print_table, section,
)


FEATURE_DESCRIPTIONS = {
    "Age": "Age in years",
    "Gender": "Female or Male",
    "Weight": "Weight in kilograms",
    "Height": "Height in centimetres",
    "Smoking": "Never, Former, or Current",
    "Alcohol_Intake": "None, Low, Moderate, or High",
    "Physical_Activity": "Sedentary, Moderate, or Active",
    "Diet": "Unhealthy, Average, or Healthy",
    "Stress_Level": "Low, Medium, or High",
    "Hypertension": "0 or 1",
    "Diabetes": "0 or 1",
    "Hyperlipidemia": "0 or 1",
    "Family_History": "0 or 1",
    "Previous_Heart_Attack": "0 or 1",
    "Systolic_BP": "Systolic blood pressure",
    "Diastolic_BP": "Diastolic blood pressure",
    "Heart_Rate": "Heart rate",
    "Blood_Sugar_Fasting": "Fasting blood sugar",
    "Cholesterol_Total": "Total cholesterol",
}


def run():
    banner("OVERVIEW DASHBOARD")
    section("Project")
    print("""
  This academic project uses one Support Vector Machine classifier to predict
  Heart_Disease from 19 model input attributes from the current 50,000-row CSV. The pipeline
  performs preprocessing and classification consistently for both interfaces.

  This is an academic AI prototype and is not a medical diagnosis.
    """)

    section("Objectives")
    for goal in [
        "1. Preserve Alcohol_Intake='None' as a legitimate category.",
        "2. Fit scaling and one-hot encoding only inside the SVM pipeline.",
        "3. Tune RBF SVM parameters using training-only cross-validation.",
        "4. Evaluate once using Accuracy, Precision, Recall, F1 and ROC-AUC.",
    ]:
        print(f"  {goal}")

    section("Dataset Summary")
    df = load_dataset()
    if df is None:
        print("  [ERROR] The shared dataset is missing.")
    else:
        counts = df[STATUS_COLUMN].value_counts()
        print_table({
            "Dataset records": len(df),
            "Input attributes": len(df.columns) - 1,
            "Numerical attributes": len(NUMERICAL_FEATURES),
            "Categorical attributes": len(CATEGORICAL_FEATURES),
            "Target": STATUS_COLUMN,
            "0 - No": int(counts.get(0, 0)),
            "1 - Yes": int(counts.get(1, 0)),
            "Missing input values": int(df.drop(columns=STATUS_COLUMN).isna().sum().sum()),
        })

    section("Input Attribute Dictionary")
    for name, description in FEATURE_DESCRIPTIONS.items():
        kind = "Numerical" if name in NUMERICAL_FEATURES else "Categorical"
        print(f"  {name:<28} {kind:<12} {description}")
    press_enter()

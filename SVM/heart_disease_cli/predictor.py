"""Interactive raw-input prediction with the shared preprocessing + SVM pipeline."""

import pandas as pd

from utils import (
    FEATURES, PIPELINE_PATH, WIDTH, banner, load_pipeline, press_enter,
    print_table, prompt_choice, prompt_float, prompt_yn, section,
)


def _choice(label: str, options: list[str]) -> str:
    print(f"\n  {label}")
    return options[prompt_choice("Select", options)]


def _binary(label: str) -> int:
    return prompt_choice(label, ["0 - No", "1 - Yes"])


def _collect_patient_data() -> pd.DataFrame:
    section("Enter All 19 Model Input Attributes")
    print("  Enter numerical measurements and select each categorical value.\n")
    data = {
        "Age": prompt_float("1. Age", 30, 79),
        "Gender": _choice("2. Gender", ["Female", "Male"]),
        "Weight": prompt_float("3. Weight (kg)", 50, 119),
        "Height": prompt_float("4. Height (cm)", 150, 199),
        "Smoking": _choice("5. Smoking", ["Never", "Former", "Current"]),
        "Alcohol_Intake": _choice("6. Alcohol Intake", ["None", "Low", "Moderate", "High"]),
        "Physical_Activity": _choice("7. Physical Activity", ["Sedentary", "Moderate", "Active"]),
        "Diet": _choice("8. Diet", ["Unhealthy", "Average", "Healthy"]),
        "Stress_Level": _choice("9. Stress Level", ["Low", "Medium", "High"]),
        "Hypertension": _binary("10. Hypertension"),
        "Diabetes": _binary("11. Diabetes"),
        "Hyperlipidemia": _binary("12. Hyperlipidemia"),
        "Family_History": _binary("13. Family History"),
        "Previous_Heart_Attack": _binary("14. Previous Heart Attack"),
        "Systolic_BP": prompt_float("15. Systolic BP", 100, 179),
        "Diastolic_BP": prompt_float("16. Diastolic BP", 60, 119),
        "Heart_Rate": prompt_float("17. Heart Rate", 60, 109),
        "Blood_Sugar_Fasting": prompt_float("18. Fasting Blood Sugar", 70, 179),
        "Cholesterol_Total": prompt_float("19. Total Cholesterol", 150, 299),
    }
    return pd.DataFrame([data], columns=FEATURES)


def _print_result(prediction: int, patient_df: pd.DataFrame):
    label = "Yes" if prediction == 1 else "No"
    print(f"\n{'=' * WIDTH}")
    print("  HEART DISEASE CLASSIFICATION PREDICTION")
    print(f"  Result: {label}")
    print("  This is an academic AI prototype and is not a medical diagnosis.")
    print("=" * WIDTH)
    section("Entered Attribute Summary")
    print_table(patient_df.iloc[0].to_dict())


def predict_patient(patient_df: pd.DataFrame) -> int:
    """Pass the raw, exact-schema DataFrame directly to the shared pipeline."""
    return int(load_pipeline().predict(patient_df)[0])


def run():
    banner("HEART DISEASE CLASSIFICATION PREDICTION")
    if not PIPELINE_PATH.is_file():
        print("\n  [ERROR] The fitted SVM pipeline is missing. Run Model Training first.")
        press_enter()
        return
    while True:
        patient_df = _collect_patient_data()
        _print_result(predict_patient(patient_df), patient_df)
        if not prompt_yn("Run another classification prediction?"):
            break

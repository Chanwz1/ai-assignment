"""Smoke-test raw-data inference through the fitted preprocessing + SVM pipeline."""

import joblib
import pandas as pd

from project_config import MODEL_FEATURES, MODEL_PATH, TARGET, model_input


pipeline = joblib.load(MODEL_PATH)

sample_patient = pd.DataFrame([{
    "Age": 62,
    "Gender": "Male",
    "Weight": 82,
    "Height": 176,
    "Smoking": "Former",
    "Alcohol_Intake": "None",
    "Physical_Activity": "Moderate",
    "Diet": "Average",
    "Stress_Level": "Medium",
    "Hypertension": 1,
    "Diabetes": 0,
    "Hyperlipidemia": 0,
    "Family_History": 1,
    "Previous_Heart_Attack": 0,
    "Systolic_BP": 145,
    "Diastolic_BP": 90,
    "Heart_Rate": 82,
    "Blood_Sugar_Fasting": 110,
    "Cholesterol_Total": 235,
}])

sample_patient = model_input(sample_patient)
assert sample_patient.columns.tolist() == MODEL_FEATURES
assert TARGET not in sample_patient.columns
prediction = int(pipeline.predict(sample_patient)[0])
print(f"Inference completed successfully with all {len(MODEL_FEATURES)} model inputs.")
print(f"Prediction: {prediction} ({'Yes' if prediction == 1 else 'No'})")

"""Smoke-test the combined ANN/KNN/SVM predictors with one sample patient."""

from app import predict_ann, predict_knn, predict_svm


SAMPLE_PATIENT = {
    "Age": 60,
    "Gender": "Male",
    "Weight": 90,
    "Height": 172,
    "BMI": 30.4,
    "Smoking": "Current",
    "Alcohol_Intake": "High",
    "Physical_Activity": "Sedentary",
    "Diet": "Unhealthy",
    "Stress_Level": "High",
    "Hypertension": 1,
    "Diabetes": 1,
    "Hyperlipidemia": 1,
    "Family_History": 1,
    "Previous_Heart_Attack": 0,
    "Systolic_BP": 150,
    "Diastolic_BP": 95,
    "Heart_Rate": 90,
    "Blood_Sugar_Fasting": 140,
    "Cholesterol_Total": 260,
}


def main() -> None:
    for name, predictor in {
        "ANN": predict_ann,
        "KNN": predict_knn,
        "SVM": predict_svm,
    }.items():
        prediction, score = predictor(SAMPLE_PATIENT)
        print(f"{name}: prediction={prediction}, score={score:.4f}")


if __name__ == "__main__":
    main()

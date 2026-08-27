"""Combined Streamlit app for ANN, KNN, and SVM heart-disease prediction."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
TARGET = "Heart_Disease"
RANDOM_STATE = 42

DATA_PATHS = [
    ROOT / "synthetic_heart_disease_dataset.csv",
    PROJECT_ROOT / "SVM" / "heart_disease_classifier" / "synthetic_heart_disease_dataset.csv",
    PROJECT_ROOT / "KNN" / "synthetic_heart_disease_dataset.csv",
    PROJECT_ROOT / "ANN" / "AI Assignment" / "synthetic_heart_disease_dataset.csv",
]

ANN_MODEL_PATHS = [
    ROOT / "models" / "heart_disease_ann_model.pkl",
    PROJECT_ROOT / "ANN" / "AI Assignment" / "heart_disease_ann_model.pkl",
]
ANN_SCALER_PATHS = [
    ROOT / "models" / "ann_scaler.pkl",
    PROJECT_ROOT / "ANN" / "AI Assignment" / "scaler.pkl",
]
ANN_FEATURE_PATHS = [
    ROOT / "models" / "ann_feature_names.pkl",
    PROJECT_ROOT / "ANN" / "AI Assignment" / "feature_names.pkl",
]
SVM_MODEL_PATHS = [
    ROOT / "models" / "svm_pipeline.joblib",
    PROJECT_ROOT / "SVM" / "heart_disease_classifier" / "models" / "svm_pipeline.joblib",
]
SVM_METRICS_PATHS = [
    ROOT / "static" / "svm_metrics.json",
    PROJECT_ROOT / "SVM" / "heart_disease_classifier" / "static" / "metrics.json",
]
KNN_MODEL_PATH = ROOT / "models" / "knn_pipeline.joblib"

DATA_COLUMNS = [
    "Age", "Gender", "Weight", "Height", "BMI", "Smoking", "Alcohol_Intake",
    "Physical_Activity", "Diet", "Stress_Level", "Hypertension", "Diabetes",
    "Hyperlipidemia", "Family_History", "Previous_Heart_Attack", "Systolic_BP",
    "Diastolic_BP", "Heart_Rate", "Blood_Sugar_Fasting", "Cholesterol_Total",
    TARGET,
]
MODEL_FEATURES = [name for name in DATA_COLUMNS if name not in {TARGET, "BMI"}]
ANN_FEATURES = [name for name in DATA_COLUMNS if name != TARGET]
CATEGORICAL_FEATURES = [
    "Gender", "Smoking", "Alcohol_Intake", "Physical_Activity", "Diet", "Stress_Level",
]
NUMERICAL_FEATURES = [name for name in MODEL_FEATURES if name not in CATEGORICAL_FEATURES]

CATEGORY_OPTIONS = {
    "Gender": ["Female", "Male"],
    "Smoking": ["Never", "Former", "Current"],
    "Alcohol_Intake": ["None", "Low", "Moderate", "High"],
    "Physical_Activity": ["Sedentary", "Moderate", "Active"],
    "Diet": ["Unhealthy", "Average", "Healthy"],
    "Stress_Level": ["Low", "Medium", "High"],
}
ANN_MAPS = {
    "Gender": {"Male": 1, "Female": 0},
    "Smoking": {"Never": 0, "Former": 1, "Current": 2},
    "Alcohol_Intake": {"None": 0, "Low": 1, "Moderate": 2, "High": 3},
    "Physical_Activity": {"Sedentary": 0, "Moderate": 1, "Active": 2},
    "Diet": {"Unhealthy": 0, "Average": 1, "Healthy": 2},
    "Stress_Level": {"Low": 0, "Medium": 1, "High": 2},
}
KNN_MAPS = {
    "Gender": {"Male": 1, "Female": 0},
    "Smoking": {"Never": 0, "Former": 1, "Current": 2},
    "Alcohol_Intake": {"Low": 0, "Moderate": 1, "High": 2},
    "Physical_Activity": {"Sedentary": 0, "Moderate": 1, "Active": 2},
    "Diet": {"Unhealthy": 0, "Average": 1, "Healthy": 2},
    "Stress_Level": {"Low": 0, "Medium": 1, "High": 2},
}


def first_existing(paths: list[Path], label: str) -> Path:
    for path in paths:
        if path.exists():
            return path
    raise FileNotFoundError(f"Could not find {label}. Checked: {', '.join(str(p) for p in paths)}")


@st.cache_data
def load_dataset() -> pd.DataFrame:
    path = first_existing(DATA_PATHS, "dataset")
    return pd.read_csv(path, keep_default_na=False)


@st.cache_resource
def load_ann_artifacts():
    model = joblib.load(first_existing(ANN_MODEL_PATHS, "ANN model"))
    scaler = joblib.load(first_existing(ANN_SCALER_PATHS, "ANN scaler"))
    features = joblib.load(first_existing(ANN_FEATURE_PATHS, "ANN feature list"))
    return model, scaler, features


@st.cache_resource
def load_svm_pipeline():
    return joblib.load(first_existing(SVM_MODEL_PATHS, "SVM pipeline"))


@st.cache_data
def load_svm_metrics() -> dict | None:
    try:
        path = first_existing(SVM_METRICS_PATHS, "SVM metrics")
    except FileNotFoundError:
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def encode_with_maps(data: pd.DataFrame, maps: dict[str, dict[str, int]]) -> pd.DataFrame:
    encoded = data.copy()
    for column, mapping in maps.items():
        encoded[column] = encoded[column].map(mapping)
    return encoded


@st.cache_resource(show_spinner="Training KNN model for first use...")
def load_or_train_knn_pipeline():
    if KNN_MODEL_PATH.exists():
        return joblib.load(KNN_MODEL_PATH)

    df = load_dataset().drop_duplicates().reset_index(drop=True)
    df = df[df["Systolic_BP"] > df["Diastolic_BP"]].reset_index(drop=True)
    df = encode_with_maps(df, KNN_MAPS)

    X = df.drop(columns=[TARGET])
    y = df[TARGET].astype(int)
    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", KNeighborsClassifier(n_neighbors=25)),
    ])
    pipeline.fit(X_train, y_train)

    KNN_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, KNN_MODEL_PATH)
    return pipeline


def build_patient_input() -> dict:
    values: dict[str, object] = {}
    with st.form("patient_form"):
        st.subheader("Patient Details")
        col1, col2, col3 = st.columns(3)
        with col1:
            values["Age"] = st.number_input("Age", 18, 120, 50)
            values["Gender"] = st.selectbox("Gender", CATEGORY_OPTIONS["Gender"])
            values["Weight"] = st.number_input("Weight (kg)", 20, 300, 75)
            values["Height"] = st.number_input("Height (cm)", 100, 250, 170)
        with col2:
            values["Smoking"] = st.selectbox("Smoking", CATEGORY_OPTIONS["Smoking"])
            values["Alcohol_Intake"] = st.selectbox("Alcohol Intake", CATEGORY_OPTIONS["Alcohol_Intake"])
            values["Physical_Activity"] = st.selectbox(
                "Physical Activity", CATEGORY_OPTIONS["Physical_Activity"]
            )
            values["Diet"] = st.selectbox("Diet", CATEGORY_OPTIONS["Diet"])
            values["Stress_Level"] = st.selectbox("Stress Level", CATEGORY_OPTIONS["Stress_Level"])
        with col3:
            values["Hypertension"] = int(st.checkbox("Hypertension"))
            values["Diabetes"] = int(st.checkbox("Diabetes"))
            values["Hyperlipidemia"] = int(st.checkbox("Hyperlipidemia"))
            values["Family_History"] = int(st.checkbox("Family History"))
            values["Previous_Heart_Attack"] = int(st.checkbox("Previous Heart Attack"))

        st.subheader("Vitals and Lab Results")
        col4, col5, col6 = st.columns(3)
        with col4:
            values["Systolic_BP"] = st.number_input("Systolic BP", 60, 250, 120)
            values["Diastolic_BP"] = st.number_input("Diastolic BP", 30, 150, 80)
        with col5:
            values["Heart_Rate"] = st.number_input("Heart Rate", 30, 220, 72)
            values["Blood_Sugar_Fasting"] = st.number_input("Fasting Blood Sugar", 40, 500, 95)
        with col6:
            values["Cholesterol_Total"] = st.number_input("Total Cholesterol", 80, 500, 190)
            bmi = round(values["Weight"] / ((values["Height"] / 100) ** 2), 1)
            values["BMI"] = bmi
            st.metric("Calculated BMI", bmi)

        submitted = st.form_submit_button("Predict", use_container_width=True)
    return values, submitted


def predict_ann(patient: dict) -> tuple[int, float]:
    model, scaler, feature_names = load_ann_artifacts()
    raw = pd.DataFrame([patient], columns=ANN_FEATURES)
    encoded = encode_with_maps(raw, ANN_MAPS)
    scaled = scaler.transform(encoded.loc[:, feature_names])
    prediction = int(model.predict(scaled)[0])
    probability = float(model.predict_proba(scaled)[0][1])
    return prediction, probability


def predict_knn(patient: dict) -> tuple[int, float]:
    raw = pd.DataFrame([patient], columns=ANN_FEATURES)
    encoded = encode_with_maps(raw, KNN_MAPS)
    pipeline = load_or_train_knn_pipeline()
    prediction = int(pipeline.predict(encoded)[0])
    probability = float(pipeline.predict_proba(encoded)[0][1])
    return prediction, probability


def predict_svm(patient: dict) -> tuple[int, float | None]:
    raw = pd.DataFrame([patient], columns=ANN_FEATURES).loc[:, MODEL_FEATURES]
    pipeline = load_svm_pipeline()
    prediction = int(pipeline.predict(raw)[0])
    if hasattr(pipeline, "decision_function"):
        score = float(pipeline.decision_function(raw)[0])
        probability_like = 1.0 / (1.0 + np.exp(-score))
        return prediction, probability_like
    return prediction, None


PREDICTORS = {
    "ANN": predict_ann,
    "KNN": predict_knn,
    "SVM": predict_svm,
}


def show_result(model_name: str, prediction: int, probability: float | None) -> None:
    label = "Higher risk of heart disease" if prediction == 1 else "Lower risk of heart disease"
    message = f"{model_name}: {label}"
    if prediction == 1:
        st.error(message)
    else:
        st.success(message)
    if probability is not None:
        st.metric("Heart disease score", f"{probability:.1%}")
        st.progress(min(max(probability, 0.0), 1.0))


def show_overview() -> None:
    df = load_dataset()
    st.subheader("Combined Project Overview")
    col1, col2, col3 = st.columns(3)
    col1.metric("Dataset rows", f"{len(df):,}")
    col2.metric("Input features", len(MODEL_FEATURES))
    col3.metric("Models", "ANN, KNN, SVM")
    st.dataframe(df.head(10), width="stretch")

    metrics = load_svm_metrics()
    if metrics:
        svm = metrics["test_metrics"]
        st.subheader("Saved SVM Performance")
        cols = st.columns(5)
        for col, key in zip(cols, ["accuracy", "precision", "recall", "f1", "roc_auc"]):
            col.metric(key.replace("_", " ").title(), f"{svm[key] * 100:.2f}%")


def main() -> None:
    st.set_page_config(page_title="Heart Disease Model Menu", page_icon="heart", layout="wide")
    st.title("Heart Disease Prediction")
    st.caption("Combined ANN, KNN, and SVM prediction menu")
    st.warning(
        "Academic demonstration only. These synthetic-data models are not medical advice "
        "or a real diagnosis."
    )

    menu = st.sidebar.radio("Menu", ["Overview", "Predict"])
    selected_model = st.sidebar.selectbox("Prediction model", ["ANN", "KNN", "SVM", "Compare all"])

    if menu == "Overview":
        show_overview()
        return

    patient, submitted = build_patient_input()
    if not submitted:
        return

    if patient["Systolic_BP"] <= patient["Diastolic_BP"]:
        st.error("Systolic BP must be higher than Diastolic BP.")
        return

    st.divider()
    st.subheader("Prediction Result")
    model_names = list(PREDICTORS) if selected_model == "Compare all" else [selected_model]
    for model_name in model_names:
        try:
            prediction, probability = PREDICTORS[model_name](patient)
            show_result(model_name, prediction, probability)
        except Exception as exc:
            st.error(f"{model_name} could not run: {exc}")


if __name__ == "__main__":
    main()

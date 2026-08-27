"""
Heart Disease Risk Predictor — Streamlit Demo UI
AI Assignment - Machine Learning (Supervised) - ANN Method

Run with:
    streamlit run app.py

Loads the model + preprocessing objects saved by train_ann_final.py
(heart_disease_ann_model.pkl, scaler.pkl, feature_names.pkl)
and lets the user enter a patient's data to get a live prediction.
"""

import streamlit as st
import pandas as pd
import joblib
import os

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="Heart Disease Risk Predictor",
    page_icon="\u2764\ufe0f",
    layout="centered",
)

# -----------------------------
# Same encoding maps used during training — must match train_ann_final.py exactly
# -----------------------------
GENDER_MAP = {"Male": 1, "Female": 0}
SMOKING_MAP = {"Never": 0, "Former": 1, "Current": 2}
# "None" represents patients who do not drink alcohol at all (not missing data).
ALCOHOL_MAP = {"None": 0, "Low": 1, "Moderate": 2, "High": 3}
ACTIVITY_MAP = {"Sedentary": 0, "Moderate": 1, "Active": 2}
DIET_MAP = {"Unhealthy": 0, "Average": 1, "Healthy": 2}
STRESS_MAP = {"Low": 0, "Medium": 1, "High": 2}

NUMERIC_RANGES = {
    "Age": (0, 120), "Weight": (20, 300), "Height": (100, 250), "BMI": (10, 70),
    "Systolic_BP": (60, 250), "Diastolic_BP": (30, 150), "Heart_Rate": (30, 220),
    "Blood_Sugar_Fasting": (40, 500), "Cholesterol_Total": (80, 500),
}

REQUIRED_FILES = [
    "heart_disease_ann_model.pkl", "scaler.pkl", "feature_names.pkl"
]


# -----------------------------
# Load model + preprocessing objects (cached so it only loads once)
# -----------------------------
@st.cache_resource
def load_artifacts():
    missing = [f for f in REQUIRED_FILES if not os.path.exists(f)]
    if missing:
        raise FileNotFoundError(
            f"Missing required file(s): {missing}. "
            "Run train_ann_final.py first to generate the trained model."
        )
    model = joblib.load("heart_disease_ann_model.pkl")
    scaler = joblib.load("scaler.pkl")
    feature_names = joblib.load("feature_names.pkl")
    return model, scaler, feature_names


def encode_categorical(patient: dict) -> pd.DataFrame:
    p = pd.DataFrame([patient])
    p["Gender"] = p["Gender"].map(GENDER_MAP)
    p["Smoking"] = p["Smoking"].map(SMOKING_MAP)
    p["Alcohol_Intake"] = p["Alcohol_Intake"].map(ALCOHOL_MAP)
    p["Physical_Activity"] = p["Physical_Activity"].map(ACTIVITY_MAP)
    p["Diet"] = p["Diet"].map(DIET_MAP)
    p["Stress_Level"] = p["Stress_Level"].map(STRESS_MAP)
    return p


def validate_numeric(patient: dict) -> list:
    """Returns a list of human-readable error messages (empty if all valid)."""
    errors = []
    for field, (low, high) in NUMERIC_RANGES.items():
        value = patient.get(field)
        if value is None:
            errors.append(f"'{field}' is missing.")
            continue
        if not (low <= value <= high):
            errors.append(f"'{field}' = {value} is outside the expected range [{low}, {high}].")
    return errors


def predict(patient: dict, model, scaler, feature_names):
    errors = validate_numeric(patient)
    if errors:
        raise ValueError(" / ".join(errors))

    p = encode_categorical(patient)
    if p.isnull().any().any():
        raise ValueError("One or more selections could not be processed. Please check your inputs.")

    p = p[feature_names]
    p_scaled = scaler.transform(p)

    pred = model.predict(p_scaled)[0]
    proba = model.predict_proba(p_scaled)[0][1]
    return pred, proba


# -----------------------------
# UI
# -----------------------------
st.title("\u2764\ufe0f Heart Disease Risk Predictor")
st.caption(
    "AI Assignment \u2014 Machine Learning (Supervised) \u2014 Artificial Neural Network (ANN)"
)
st.markdown(
    "Enter a patient's information below to estimate their risk of heart disease. "
)

try:
    model, scaler, feature_names = load_artifacts()
except FileNotFoundError as e:
    st.error(f"\u26a0\ufe0f {e}")
    st.stop()

st.divider()

with st.form("patient_form"):
    st.subheader("Demographics")
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", min_value=0, max_value=120, value=50)
        gender = st.selectbox("Gender", list(GENDER_MAP.keys()))
    with col2:
        weight = st.number_input("Weight (kg)", min_value=20, max_value=300, value=75)
        height = st.number_input("Height (cm)", min_value=100, max_value=250, value=170)

    bmi = round(weight / ((height / 100) ** 2), 1)
    st.caption(f"Calculated BMI: **{bmi}**")

    st.subheader("Lifestyle")
    col3, col4 = st.columns(2)
    with col3:
        smoking = st.selectbox("Smoking", list(SMOKING_MAP.keys()))
        alcohol = st.selectbox("Alcohol Intake", list(ALCOHOL_MAP.keys()))
    with col4:
        activity = st.selectbox("Physical Activity", list(ACTIVITY_MAP.keys()))
        diet = st.selectbox("Diet", list(DIET_MAP.keys()))
    stress = st.selectbox("Stress Level", list(STRESS_MAP.keys()))

    st.subheader("Medical History")
    col5, col6 = st.columns(2)
    with col5:
        hypertension = st.checkbox("Hypertension")
        diabetes = st.checkbox("Diabetes")
        hyperlipidemia = st.checkbox("Hyperlipidemia")
    with col6:
        family_history = st.checkbox("Family History of Heart Disease")
        previous_attack = st.checkbox("Previous Heart Attack")

    st.subheader("Vitals & Lab Results")
    col7, col8 = st.columns(2)
    with col7:
        systolic = st.number_input("Systolic BP (mmHg)", min_value=60, max_value=250, value=120)
        diastolic = st.number_input("Diastolic BP (mmHg)", min_value=30, max_value=150, value=80)
        heart_rate = st.number_input("Heart Rate (bpm)", min_value=30, max_value=220, value=72)
    with col8:
        blood_sugar = st.number_input("Fasting Blood Sugar (mg/dL)", min_value=40, max_value=500, value=95)
        cholesterol = st.number_input("Total Cholesterol (mg/dL)", min_value=80, max_value=500, value=190)

    submitted = st.form_submit_button("Predict Heart Disease Risk", use_container_width=True)

if submitted:
    patient = {
        "Age": age, "Gender": gender, "Weight": weight, "Height": height, "BMI": bmi,
        "Smoking": smoking, "Alcohol_Intake": alcohol, "Physical_Activity": activity,
        "Diet": diet, "Stress_Level": stress,
        "Hypertension": int(hypertension), "Diabetes": int(diabetes),
        "Hyperlipidemia": int(hyperlipidemia), "Family_History": int(family_history),
        "Previous_Heart_Attack": int(previous_attack),
        "Systolic_BP": systolic, "Diastolic_BP": diastolic, "Heart_Rate": heart_rate,
        "Blood_Sugar_Fasting": blood_sugar, "Cholesterol_Total": cholesterol,
    }

    try:
        pred_label, pred_proba = predict(patient, model, scaler, feature_names)
    except ValueError as e:
        st.error(f"\u26a0\ufe0f Invalid input: {e}")
    except Exception as e:
        st.error(f"\u26a0\ufe0f Unexpected error while predicting: {e}")
    else:
        st.divider()
        st.subheader("Result")

        if pred_label == 1:
            st.error(f"### \u26a0\ufe0f Higher Risk of Heart Disease")
        else:
            st.success(f"### \u2705 Lower Risk of Heart Disease")

        st.metric("Predicted probability of heart disease", f"{pred_proba:.1%}")
        st.progress(min(max(pred_proba, 0.0), 1.0))

        st.caption(
            "This prediction comes from an ANN trained on a synthetic dataset for an "
            "academic assignment. It is not medical advice — please consult a qualified "
            "healthcare professional for any real health concerns."
        )
"""Streamlit interface for the SVM-only heart disease classification prototype."""

import json

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

from project_config import (
    CATEGORY_OPTIONS, CONFUSION_MATRIX_PATH, METRICS_PATH, MODEL_FEATURES,
    MODEL_PATH, TARGET, load_dataset as load_current_dataset, model_input,
)

st.set_page_config(page_title="Heart Disease Classification Prediction", page_icon="❤️", layout="wide")


@st.cache_data
def cached_dataset() -> pd.DataFrame:
    return load_current_dataset()


@st.cache_data
def load_metrics() -> dict:
    return json.loads(METRICS_PATH.read_text(encoding="utf-8"))


@st.cache_resource
def load_pipeline():
    return joblib.load(MODEL_PATH)


st.title("Heart Disease Classification Prediction")
st.caption("TARUMT Artificial Intelligence assignment — leakage-safe RBF SVM prototype")
st.warning(
    "Academic demonstration only. This synthetic-data classifier is not a medical "
    "diagnostic system and must not replace professional evaluation."
)

page = st.sidebar.radio(
    "Navigation",
    ["Overview", "Exploratory Data Analysis", "SVM Model Performance", "Prediction"],
)

try:
    df = cached_dataset()
except Exception as exc:
    st.error(f"Unable to load the current dataset: {exc}")
    st.stop()


if page == "Overview":
    st.header("Current Dataset Overview")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Records", f"{len(df):,}")
    col2.metric("CSV input attributes", len(df.columns) - 1)
    col3.metric("Model input attributes", len(MODEL_FEATURES))
    col4.metric("Target", TARGET)
    st.write(
        "The app loads only `synthetic_heart_disease_dataset.csv`. Target values are "
        "0 = no heart disease and 1 = heart disease. BMI is excluded from model input "
        "because it conflicts with Weight and Height."
    )
    counts = df[TARGET].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(x=["0 - No", "1 - Yes"], y=counts.values, ax=ax, color="#2878b5")
    ax.set_title("Heart_Disease Class Distribution")
    ax.set_ylabel("Records")
    st.pyplot(fig)
    st.caption("Alcohol_Intake='None' is preserved as a real category, not treated as missing.")

elif page == "Exploratory Data Analysis":
    st.header("Exploratory Data Analysis")
    preview, signal, bmi = st.tabs(["Dataset quality", "Target relationships", "BMI consistency"])
    with preview:
        st.dataframe(df.head(10), width="stretch")
        st.write(f"Shape: `{df.shape}`")
        st.write(f"Duplicate rows: `{df.duplicated().sum():,}`")
        st.write(f"True missing values: `{df.isna().sum().sum():,}`")
        st.write("Alcohol categories:", sorted(df["Alcohol_Intake"].unique().tolist()))
    with signal:
        feature = st.selectbox(
            "Feature",
            ["Age", "Hypertension", "Diabetes", "Previous_Heart_Attack", "Cholesterol_Total"],
        )
        if set(df[feature].unique()).issubset({0, 1}):
            rates = df.groupby(feature)[TARGET].agg(Records="count", Heart_Disease_Rate="mean")
            st.dataframe(rates, width="stretch")
        else:
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.histplot(data=df, x=feature, hue=TARGET, element="step", stat="density", ax=ax)
            ax.set_title(f"{feature} distribution by {TARGET}")
            st.pyplot(fig)
        correlations = (
            df.select_dtypes("number").corr()[TARGET].drop(TARGET).sort_values(key=abs, ascending=False)
        )
        st.subheader("Numerical correlation with Heart_Disease")
        st.dataframe(correlations.rename("Pearson correlation").to_frame(), width="stretch")
    with bmi:
        recalculated = df["Weight"] / (df["Height"] / 100.0) ** 2
        difference = (df["BMI"] - recalculated).abs()
        st.error(
            "Dataset-quality issue: supplied BMI is inconsistent with Weight and Height. "
            f"Median absolute difference: {difference.median():.2f}; correlation: "
            f"{df['BMI'].corr(recalculated):.3f}."
        )
        st.dataframe(pd.DataFrame({"Supplied BMI": df["BMI"], "Recalculated BMI": recalculated}).head(20))

elif page == "SVM Model Performance":
    st.header("Final SVM Model Performance")
    if not METRICS_PATH.exists():
        st.info("Run `python train.py` to create the current evaluation results.")
    else:
        metrics = load_metrics()
        values = metrics["test_metrics"]
        columns = st.columns(5)
        for column, label, key in zip(
            columns,
            ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"],
            ["accuracy", "precision", "recall", "f1", "roc_auc"],
        ):
            column.metric(label, f"{values[key] * 100:.2f}%")
        cv = metrics["full_training_cv"]
        st.write(
            f"**Five-fold training CV accuracy:** {cv['mean_accuracy']:.4f} "
            f"+/- {cv['std_accuracy']:.4f}  "
            f"\n**Majority baseline:** {values['majority_class_baseline_accuracy']:.4f}  "
            f"\n**Best parameters:** `{metrics['tuning']['best_parameters']}`"
        )
        st.dataframe(
            pd.DataFrame(values["confusion_matrix"], index=["Actual 0", "Actual 1"],
                         columns=["Predicted 0", "Predicted 1"]),
            width="stretch",
        )
        if CONFUSION_MATRIX_PATH.exists():
            st.image(str(CONFUSION_MATRIX_PATH), caption="Final untouched-test confusion matrix")
        st.info(metrics["synthetic_rule_audit"]["recovered_rule"])

else:
    st.header("Heart Disease Classification Prediction")
    if not MODEL_PATH.exists():
        st.info("Run `python train.py` before submitting a prediction.")
        st.stop()

    numeric_specs = {
        "Age": (30, 79, 54), "Weight": (50, 119, 85), "Height": (150, 199, 174),
        "Systolic_BP": (100, 179, 139), "Diastolic_BP": (60, 119, 90),
        "Heart_Rate": (60, 109, 85), "Blood_Sugar_Fasting": (70, 179, 125),
        "Cholesterol_Total": (150, 299, 225),
    }
    binary_features = [
        "Hypertension", "Diabetes", "Hyperlipidemia", "Family_History",
        "Previous_Heart_Attack",
    ]
    values = {}
    with st.form("prediction_form"):
        form_columns = st.columns(3)
        for index, feature in enumerate(MODEL_FEATURES):
            with form_columns[index % 3]:
                if feature in numeric_specs:
                    low, high, default = numeric_specs[feature]
                    values[feature] = st.number_input(feature, low, high, default)
                elif feature in binary_features:
                    values[feature] = st.selectbox(
                        feature, [0, 1],
                        format_func=lambda x: f"{x} - {'Yes' if x else 'No'}",
                    )
                else:
                    values[feature] = st.selectbox(feature, CATEGORY_OPTIONS[feature])
        submitted = st.form_submit_button("Submit classification prediction")

    if submitted:
        raw_input = model_input(pd.DataFrame([values]))
        prediction = int(load_pipeline().predict(raw_input)[0])
        if prediction == 1:
            st.error("SVM classification result: 1 - Heart disease")
        else:
            st.success("SVM classification result: 0 - No heart disease")
        st.caption("Academic demonstration only; this output is not a diagnosis.")

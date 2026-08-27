"""Train, validate, and save the assignment's leakage-safe SVM pipeline."""

from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
from time import perf_counter

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import (
    RandomizedSearchCV, StratifiedKFold, cross_val_score, train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

from project_config import (
    CATEGORICAL_FEATURES, CONFUSION_MATRIX_PATH, DATA_PATH, METRICS_PATH,
    MODEL_FEATURES, MODEL_PATH, NUMERICAL_FEATURES, REMOVED_FEATURES, TARGET,
    load_dataset,
)


RANDOM_STATE = 42
TEST_SIZE = 0.20
TUNING_SAMPLE_SIZE = 12_000
N_JOBS = 4
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)


def build_pipeline(
    numerical_features: list[str] | None = None,
    categorical_features: list[str] | None = None,
) -> Pipeline:
    """Build preprocessing and SVC as one estimator fitted inside each CV fold."""
    numerical_features = numerical_features or NUMERICAL_FEATURES
    categorical_features = categorical_features or CATEGORICAL_FEATURES
    numerical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessing = ColumnTransformer([
        ("numerical", numerical_pipeline, numerical_features),
        ("categorical", categorical_pipeline, categorical_features),
    ])
    return Pipeline([
        ("preprocessing", preprocessing),
        ("classifier", SVC(kernel="rbf", cache_size=1000)),
    ])


def stratified_tuning_sample(X_train: pd.DataFrame, y_train: pd.Series):
    """Draw a fixed training-only subset so SVC tuning is reproducible and practical."""
    if len(X_train) <= TUNING_SAMPLE_SIZE:
        return X_train.copy(), y_train.copy()
    X_tune, _, y_tune, _ = train_test_split(
        X_train, y_train, train_size=TUNING_SAMPLE_SIZE, stratify=y_train,
        random_state=RANDOM_STATE,
    )
    return X_tune, y_tune


def compare_bmi_variants(X_tune_raw: pd.DataFrame, y_tune: pd.Series) -> list[dict]:
    """Compare supplied, recalculated, and removed BMI with training-only CV."""
    results = []
    variants = {
        "supplied": X_tune_raw.copy(),
        "recalculated": X_tune_raw.assign(
            BMI=X_tune_raw["Weight"] / (X_tune_raw["Height"] / 100.0) ** 2
        ),
        "removed": X_tune_raw.drop(columns=["BMI"]),
    }
    for name, X_variant in variants.items():
        numeric = [column for column in X_variant.columns if column not in CATEGORICAL_FEATURES]
        pipeline = build_pipeline(numeric, CATEGORICAL_FEATURES).set_params(
            classifier__C=5, classifier__gamma="scale", classifier__class_weight=None,
        )
        scores = cross_val_score(
            pipeline, X_variant, y_tune, scoring="accuracy", cv=CV, n_jobs=N_JOBS
        )
        results.append({
            "variant": name,
            "cv_scores": [float(score) for score in scores],
            "mean_cv_accuracy": float(scores.mean()),
            "std_cv_accuracy": float(scores.std()),
        })
    return results


def serialize_search_results(search: RandomizedSearchCV) -> list[dict]:
    """Keep every tested candidate and its five validation scores."""
    rows = []
    results = search.cv_results_
    for index, params in enumerate(results["params"]):
        rows.append({
            "rank": int(results["rank_test_score"][index]),
            "mean_cv_accuracy": float(results["mean_test_score"][index]),
            "std_cv_accuracy": float(results["std_test_score"][index]),
            "fold_scores": [
                float(results[f"split{fold}_test_score"][index]) for fold in range(5)
            ],
            "parameters": params,
        })
    return sorted(rows, key=lambda row: row["rank"])


def main() -> None:
    started = perf_counter()
    df = load_dataset()
    print(f"Dataset: {DATA_PATH.resolve()}")
    print(df.shape)
    print(df.columns.tolist())
    print(df[TARGET].value_counts().sort_index())

    X_all_raw = df.drop(columns=[TARGET])
    y = df[TARGET].astype(int)
    assert TARGET not in X_all_raw.columns
    assert TARGET not in MODEL_FEATURES

    recalculated_bmi = df["Weight"] / (df["Height"] / 100.0) ** 2
    bmi_error = (df["BMI"] - recalculated_bmi).abs()
    bmi_quality = {
        "correlation_supplied_vs_recalculated": float(df["BMI"].corr(recalculated_bmi)),
        "mean_absolute_error": float(bmi_error.mean()),
        "median_absolute_error": float(bmi_error.median()),
        "within_0_1_share": float((bmi_error <= 0.1).mean()),
        "within_1_0_share": float((bmi_error <= 1.0).mean()),
    }

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_all_raw, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE,
    )
    X_tune_raw, y_tune = stratified_tuning_sample(X_train_raw, y_train)

    print("Comparing BMI variants with five-fold CV on the training-only tuning subset...")
    bmi_variants = compare_bmi_variants(X_tune_raw, y_tune)
    for result in bmi_variants:
        print(
            f"  {result['variant']}: {result['mean_cv_accuracy']:.4f} "
            f"(+/- {result['std_cv_accuracy']:.4f})"
        )

    X_train = X_train_raw.loc[:, MODEL_FEATURES]
    X_test = X_test_raw.loc[:, MODEL_FEATURES]
    X_tune = X_tune_raw.loc[:, MODEL_FEATURES]
    assert TARGET not in X_train.columns
    assert X_train.columns.tolist() == MODEL_FEATURES

    parameter_distributions = {
        "classifier__C": [0.5, 1, 2, 5, 10, 20],
        "classifier__gamma": ["scale", 0.01, 0.03, 0.05, 0.1],
        "classifier__class_weight": [None, "balanced"],
    }
    search = RandomizedSearchCV(
        build_pipeline(), param_distributions=parameter_distributions,
        n_iter=16, scoring="accuracy", cv=CV, random_state=RANDOM_STATE,
        n_jobs=N_JOBS, refit=False, return_train_score=False, verbose=1,
    )
    print("Tuning 16 RBF SVM candidates using training-only five-fold CV...")
    search.fit(X_tune, y_tune)
    candidates = serialize_search_results(search)
    best_parameters = search.best_params_
    print(f"Best parameters: {best_parameters}")

    final_pipeline = build_pipeline().set_params(**best_parameters)
    print("Computing five-fold accuracy on the complete 40,000-row training set...")
    full_cv_scores = cross_val_score(
        final_pipeline, X_train, y_train, scoring="accuracy", cv=CV, n_jobs=N_JOBS
    )

    # The held-out test set is first used here, after all feature and parameter choices.
    print("Fitting the selected SVM on all training rows and evaluating the test set once...")
    final_pipeline.fit(X_train, y_train)
    predictions = final_pipeline.predict(X_test)
    decision_scores = final_pipeline.decision_function(X_test)
    matrix = confusion_matrix(y_test, predictions)
    report = classification_report(
        y_test, predictions, target_names=["No heart disease", "Heart disease"],
        output_dict=True, zero_division=0,
    )

    strong_features = [
        "Age", "Hypertension", "Diabetes", "Previous_Heart_Attack",
        "Cholesterol_Total",
    ]
    deterministic_risk_count = (
        (X_train["Age"] > 55).astype(int)
        + X_train["Hypertension"].astype(int)
        + X_train["Diabetes"].astype(int)
        + X_train["Previous_Heart_Attack"].astype(int)
        + (X_train["Cholesterol_Total"] > 240).astype(int)
    )
    rule_predictions = (deterministic_risk_count >= 2).astype(int)

    relationships = {}
    for feature in strong_features:
        if set(df[feature].unique()).issubset({0, 1}):
            relationships[feature] = {
                str(level): {
                    "count": int(group["count"]),
                    "heart_disease_rate": float(group["mean"]),
                }
                for level, group in df.groupby(feature)[TARGET].agg(["count", "mean"]).iterrows()
            }
        else:
            relationships[feature] = {
                "pearson_correlation": float(df[[feature, TARGET]].corr().iloc[0, 1])
            }

    test_accuracy = float(accuracy_score(y_test, predictions))
    test_precision = float(precision_score(y_test, predictions, zero_division=0))
    test_recall = float(recall_score(y_test, predictions, zero_division=0))
    test_f1 = float(f1_score(y_test, predictions, zero_division=0))
    test_auc = float(roc_auc_score(y_test, decision_scores))
    metrics = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "path": DATA_PATH.name,
            "sha256": sha256(DATA_PATH.read_bytes()).hexdigest(),
            "shape": list(df.shape),
            "columns": df.columns.tolist(),
            "target": TARGET,
            "target_counts": {
                str(label): int(count)
                for label, count in df[TARGET].value_counts().sort_index().items()
            },
            "alcohol_categories": sorted(df["Alcohol_Intake"].unique().tolist()),
            "alcohol_none_count": int((df["Alcohol_Intake"] == "None").sum()),
            "duplicate_rows": int(df.duplicated().sum()),
            "true_missing_values": int(df.isna().sum().sum()),
        },
        "split": {
            "random_state": RANDOM_STATE, "stratified": True,
            "train_size": int(len(X_train)), "test_size": int(len(X_test)),
            "test_used_for_tuning": False, "final_test_evaluations": 1,
        },
        "input_features": MODEL_FEATURES,
        "numerical_features": NUMERICAL_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "removed_features": REMOVED_FEATURES,
        "target_leakage_checks": {
            "target_absent_from_X": TARGET not in X_train.columns,
            "target_absent_from_manual_feature_lists": TARGET not in MODEL_FEATURES,
            "preprocessing_fitted_inside_pipeline": True,
            "hardcoded_predictions": False,
        },
        "bmi_quality": bmi_quality,
        "bmi_variant_comparison": bmi_variants,
        "tuning": {
            "method": "RandomizedSearchCV", "selection_metric": "accuracy",
            "sample_source": "fixed stratified subset of training partition only",
            "sample_size": int(len(X_tune)), "folds": 5,
            "candidates": candidates, "best_parameters": best_parameters,
            "best_subset_cv_accuracy": float(search.best_score_),
        },
        "full_training_cv": {
            "scores": [float(score) for score in full_cv_scores],
            "mean_accuracy": float(full_cv_scores.mean()),
            "std_accuracy": float(full_cv_scores.std()),
        },
        "test_metrics": {
            "majority_class_baseline_accuracy": float(y_test.value_counts(normalize=True).max()),
            "accuracy": test_accuracy, "precision": test_precision,
            "recall": test_recall, "f1": test_f1, "roc_auc": test_auc,
            "confusion_matrix": matrix.tolist(), "classification_report": report,
        },
        "model_comparison": [{
            "model": "SVM (RBF)", "cv_accuracy": float(full_cv_scores.mean()),
            "test_accuracy": test_accuracy, "precision": test_precision,
            "recall": test_recall, "f1": test_f1, "roc_auc": test_auc,
        }],
        "relationships": relationships,
        "synthetic_rule_audit": {
            "features": strong_features,
            "recovered_rule": (
                "Heart_Disease = 1 when at least two conditions are true: Age > 55, "
                "Hypertension = 1, Diabetes = 1, Previous_Heart_Attack = 1, "
                "Cholesterol_Total > 240."
            ),
            "recovered_rule_training_accuracy": float(accuracy_score(y_train, rule_predictions)),
            "recovered_rule_matching_rows": int((y_train == rule_predictions).sum()),
            "audited_training_rows": int(len(y_train)),
            "interpretation": (
                "This is evidence of deterministic synthetic label generation, not target leakage: "
                "all inputs are legitimate pre-outcome feature columns and the target itself is excluded."
            ),
        },
        "runtime_seconds": float(perf_counter() - started),
    }

    if test_accuracy < 0.80:
        raise RuntimeError("Final SVM test accuracy is below the required 0.80 threshold")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_pipeline, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        matrix, annot=True, fmt="d", cmap="Blues", cbar=False,
        xticklabels=["No", "Yes"], yticklabels=["No", "Yes"],
    )
    plt.title("Final SVM Confusion Matrix")
    plt.xlabel("Predicted Heart_Disease")
    plt.ylabel("Actual Heart_Disease")
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_PATH, dpi=200)
    plt.close()

    print(f"Five-fold CV scores: {metrics['full_training_cv']['scores']}")
    print(f"Mean CV accuracy: {metrics['full_training_cv']['mean_accuracy']:.4f}")
    print(f"CV standard deviation: {metrics['full_training_cv']['std_accuracy']:.4f}")
    for name, value in metrics["test_metrics"].items():
        if name not in {"classification_report", "confusion_matrix"}:
            print(f"{name}: {value:.4f}")
    print(f"Confusion matrix: {matrix.tolist()}")
    print(f"Saved fitted pipeline to {MODEL_PATH}")
    print(f"Saved metrics to {METRICS_PATH}")


if __name__ == "__main__":
    main()

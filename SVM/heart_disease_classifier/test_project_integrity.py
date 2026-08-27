"""Regression checks for dataset identity, leakage controls, and saved artifacts."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import unittest

import joblib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

from project_config import (
    CATEGORICAL_FEATURES,
    DATA_COLUMNS,
    DATA_PATH,
    METRICS_PATH,
    MODEL_FEATURES,
    MODEL_PATH,
    TARGET,
    load_dataset,
)


ROOT = Path(__file__).resolve().parent


class ProjectIntegrityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = load_dataset()
        cls.metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
        cls.pipeline = joblib.load(MODEL_PATH)

    def test_current_dataset_identity_and_target(self):
        self.assertEqual(DATA_PATH.name, "synthetic_heart_disease_dataset.csv")
        obsolete_name = "heart_" + "disease.csv"
        self.assertFalse((ROOT / obsolete_name).exists())
        self.assertEqual(self.df.shape, (50_000, 21))
        self.assertEqual(self.df.columns.tolist(), DATA_COLUMNS)
        self.assertEqual(TARGET, "Heart_Disease")
        self.assertEqual(self.df[TARGET].value_counts().sort_index().to_dict(), {0: 26_827, 1: 23_173})

    def test_alcohol_none_is_a_real_category(self):
        self.assertEqual(int(self.df["Alcohol_Intake"].isna().sum()), 0)
        self.assertEqual(int((self.df["Alcohol_Intake"] == "None").sum()), 20_109)
        self.assertEqual(
            set(self.df["Alcohol_Intake"].unique()),
            {"None", "Low", "Moderate", "High"},
        )

    def test_model_schema_excludes_target_and_rejected_bmi(self):
        self.assertNotIn(TARGET, MODEL_FEATURES)
        self.assertNotIn("BMI", MODEL_FEATURES)
        self.assertEqual(len(MODEL_FEATURES), 19)
        self.assertEqual(self.pipeline.feature_names_in_.tolist(), MODEL_FEATURES)

    def test_pipeline_contains_fold_safe_preprocessing(self):
        preprocessing = self.pipeline.named_steps["preprocessing"]
        classifier = self.pipeline.named_steps["classifier"]
        self.assertIsInstance(preprocessing, ColumnTransformer)
        self.assertIsInstance(classifier, SVC)
        self.assertEqual(classifier.kernel, "rbf")
        self.assertEqual(classifier.C, 20)
        self.assertEqual(classifier.gamma, 0.1)
        self.assertIsNone(classifier.class_weight)

        numerical = preprocessing.named_transformers_["numerical"]
        categorical = preprocessing.named_transformers_["categorical"]
        self.assertIsInstance(numerical.named_steps["scaler"], StandardScaler)
        encoder = categorical.named_steps["encoder"]
        self.assertIsInstance(encoder, OneHotEncoder)
        self.assertEqual(encoder.handle_unknown, "ignore")
        self.assertEqual(CATEGORICAL_FEATURES, [
            "Gender", "Smoking", "Alcohol_Intake", "Physical_Activity", "Diet", "Stress_Level"
        ])

    def test_saved_metrics_match_dataset_and_requirements(self):
        current_hash = sha256(DATA_PATH.read_bytes()).hexdigest()
        self.assertEqual(self.metrics["dataset"]["sha256"], current_hash)
        self.assertFalse(self.metrics["split"]["test_used_for_tuning"])
        self.assertEqual(self.metrics["split"]["final_test_evaluations"], 1)
        self.assertGreaterEqual(self.metrics["test_metrics"]["accuracy"], 0.80)
        self.assertGreater(self.metrics["test_metrics"]["roc_auc"], 0.50)
        self.assertEqual(len(self.metrics["full_training_cv"]["scores"]), 5)
        audit = self.metrics["synthetic_rule_audit"]
        self.assertEqual(audit["recovered_rule_training_accuracy"], 1.0)
        self.assertEqual(audit["recovered_rule_matching_rows"], 40_000)
        self.assertEqual(audit["audited_training_rows"], 40_000)
        checks = self.metrics["target_leakage_checks"]
        self.assertTrue(checks["target_absent_from_X"])
        self.assertTrue(checks["target_absent_from_manual_feature_lists"])
        self.assertTrue(checks["preprocessing_fitted_inside_pipeline"])
        self.assertFalse(checks["hardcoded_predictions"])

    def test_executable_sources_do_not_reference_legacy_dataset_or_target(self):
        legacy_dataset_literal = '"heart_' + 'disease.csv"'
        legacy_target_literal = '"Heart Disease' + ' Status"'
        for folder in (ROOT, ROOT.parent / "heart_disease_cli"):
            for path in folder.glob("*.py"):
                text = path.read_text(encoding="utf-8")
                self.assertNotIn(legacy_dataset_literal, text, path.name)
                self.assertNotIn(legacy_target_literal, text, path.name)

    def test_training_source_uses_no_competing_classifier(self):
        text = (ROOT / "train.py").read_text(encoding="utf-8")
        forbidden = [
            "Decision" + "TreeClassifier",
            "Random" + "ForestClassifier",
            "Logistic" + "Regression",
            "KNeighbors" + "Classifier",
        ]
        for classifier_name in forbidden:
            self.assertNotIn(classifier_name, text)


if __name__ == "__main__":
    unittest.main(verbosity=2)

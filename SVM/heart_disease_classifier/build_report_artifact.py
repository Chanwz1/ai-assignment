"""Build the canonical portable technical-report artifact from saved metrics."""

from __future__ import annotations

import json
from pathlib import Path
import sqlite3


ROOT = Path(__file__).resolve().parent
METRICS_PATH = ROOT / "static" / "metrics.json"
OUTPUT_PATH = ROOT / "report_artifact.json"


def queried_rows(columns: list[str], rows: list[tuple], table: str, sql: str) -> list[dict]:
    """Materialize reviewed rows through the exact SQLite query recorded as provenance."""
    connection = sqlite3.connect(":memory:")
    column_sql = ", ".join(f'"{column}"' for column in columns)
    connection.execute(
        f'CREATE TABLE "{table}" ({column_sql})'
    )
    placeholders = ", ".join("?" for _ in columns)
    connection.executemany(f'INSERT INTO "{table}" VALUES ({placeholders})', rows)
    cursor = connection.execute(sql)
    result = [dict(zip([item[0] for item in cursor.description], row)) for row in cursor.fetchall()]
    connection.close()
    return result


def source(source_id: str, label: str, sql: str, description: str) -> dict:
    return {
        "id": source_id,
        "label": label,
        "path": "static/metrics.json",
        "query": {
            "engine": "SQLite in-memory transformation",
            "language": "sql",
            "sql": sql,
            "description": description,
            "tables_used": [sql.split("FROM ", 1)[1].split()[0]],
            "filters": ["Metrics generated from the fixed random_state=42 training run"],
            "metric_definitions": [
                "Accuracy = correct classifications / evaluated records.",
                "Precision, recall, and F1 use Heart_Disease=1 as the positive class.",
                "ROC-AUC is calculated from SVC decision_function scores.",
            ],
        },
    }


def main() -> None:
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    test = metrics["test_metrics"]
    cv = metrics["full_training_cv"]

    summary_sql = "SELECT * FROM report_summary"
    summary_rows = queried_rows(
        ["accuracy", "precision", "recall", "f1", "roc_auc", "cv_accuracy"],
        [(test["accuracy"], test["precision"], test["recall"], test["f1"],
          test["roc_auc"], cv["mean_accuracy"])],
        "report_summary",
        summary_sql,
    )
    performance_sql = (
        "SELECT metric, accuracy, scope, records, display_order "
        "FROM report_performance ORDER BY display_order"
    )
    performance_rows = queried_rows(
        ["metric", "accuracy", "scope", "records", "display_order"],
        [
            ("Majority baseline", test["majority_class_baseline_accuracy"], "Held-out test", 10_000, 1),
            ("5-fold CV mean", cv["mean_accuracy"], "Training only", 40_000, 2),
            ("Final SVM", test["accuracy"], "Held-out test", 10_000, 3),
        ],
        "report_performance",
        performance_sql,
    )
    bmi_sql = (
        "SELECT variant, mean_cv_accuracy, std_cv_accuracy, sample_size, display_order "
        "FROM bmi_variants ORDER BY display_order"
    )
    bmi_rows = queried_rows(
        ["variant", "mean_cv_accuracy", "std_cv_accuracy", "sample_size", "display_order"],
        [
            (row["variant"].title(), row["mean_cv_accuracy"], row["std_cv_accuracy"], 12_000, index)
            for index, row in enumerate(metrics["bmi_variant_comparison"], start=1)
        ],
        "bmi_variants",
        bmi_sql,
    )
    tuning_sql = (
        "SELECT rank, C, gamma, class_weight, mean_cv_accuracy, std_cv_accuracy "
        "FROM tuning_candidates ORDER BY rank"
    )
    tuning_rows = queried_rows(
        ["rank", "C", "gamma", "class_weight", "mean_cv_accuracy", "std_cv_accuracy"],
        [
            (
                row["rank"],
                row["parameters"]["classifier__C"],
                str(row["parameters"]["classifier__gamma"]),
                str(row["parameters"]["classifier__class_weight"]),
                row["mean_cv_accuracy"],
                row["std_cv_accuracy"],
            )
            for row in metrics["tuning"]["candidates"]
        ],
        "tuning_candidates",
        tuning_sql,
    )
    feature_sql = (
        "SELECT feature, treatment, reason, display_order FROM feature_schema ORDER BY display_order"
    )
    feature_rows = queried_rows(
        ["feature", "treatment", "reason", "display_order"],
        [
            (feature, "Used", "Raw input; preprocessing fitted inside Pipeline", index)
            for index, feature in enumerate(metrics["input_features"], start=1)
        ] + [
            ("BMI", "Removed", metrics["removed_features"]["BMI"], len(metrics["input_features"]) + 1)
        ],
        "feature_schema",
        feature_sql,
    )
    confusion_sql = (
        "SELECT actual_class, predicted_0, predicted_1 FROM confusion_matrix ORDER BY actual_class"
    )
    confusion_rows = queried_rows(
        ["actual_class", "predicted_0", "predicted_1"],
        [
            (0, test["confusion_matrix"][0][0], test["confusion_matrix"][0][1]),
            (1, test["confusion_matrix"][1][0], test["confusion_matrix"][1][1]),
        ],
        "confusion_matrix",
        confusion_sql,
    )

    sources = [
        source("summary_source", "Final saved SVM metrics", summary_sql, "Headline model metrics."),
        source("performance_source", "SVM accuracy and baseline", performance_sql, "Accuracy comparison rows."),
        source("bmi_source", "Training-only BMI comparison", bmi_sql, "Five-fold CV results for BMI treatments."),
        source("tuning_source", "Training-only SVM candidate results", tuning_sql, "All 16 randomized-search candidates."),
        source("feature_source", "Final model feature schema", feature_sql, "Used and removed input fields."),
        source("confusion_source", "Final held-out confusion matrix", confusion_sql, "Untouched-test confusion counts."),
    ]

    cards = [
        {
            "id": f"card_{field}", "dataset": "summary", "sourceId": "summary_source",
            "description": description,
            "metrics": [{"label": label, "field": field, "format": "percent"}],
        }
        for field, label, description in [
            ("accuracy", "Test accuracy", "Correct classifications on the one-time 10,000-row test evaluation."),
            ("precision", "Precision", "Positive predictive value for Heart_Disease=1."),
            ("recall", "Recall", "Sensitivity for Heart_Disease=1."),
            ("f1", "F1", "Harmonic mean of precision and recall."),
            ("roc_auc", "ROC-AUC", "Ranking performance from SVC decision scores."),
            ("cv_accuracy", "CV accuracy", "Mean accuracy across five full-training folds."),
        ]
    ]

    charts = [
        {
            "id": "performance_chart",
            "title": "SVM accuracy compared with the majority baseline",
            "subtitle": "Full-training CV mean and one-time held-out test accuracy",
            "showDescription": True,
            "type": "bar",
            "dataset": "performance",
            "sourceId": "performance_source",
            "encodings": {
                "x": {"field": "metric", "type": "nominal", "label": "Evaluation"},
                "y": {"field": "accuracy", "type": "quantitative", "format": "percent", "label": "Accuracy"},
                "tooltip": [
                    {"field": "scope", "type": "text", "label": "Scope"},
                    {"field": "records", "type": "quantitative", "label": "Records"},
                ],
            },
            "yAxisTitle": "Accuracy",
            "valueFormat": "percent",
            "referenceLines": [{"value": 0.80, "label": "Assignment requirement"}],
        },
        {
            "id": "bmi_chart",
            "title": "Five-fold SVM accuracy by BMI treatment",
            "subtitle": "Fixed 12,000-row stratified subset of the training partition",
            "showDescription": True,
            "type": "bar",
            "dataset": "bmi_variants",
            "sourceId": "bmi_source",
            "encodings": {
                "x": {"field": "variant", "type": "nominal", "label": "BMI treatment"},
                "y": {"field": "mean_cv_accuracy", "type": "quantitative", "format": "percent", "label": "Mean CV accuracy"},
                "tooltip": [
                    {"field": "std_cv_accuracy", "type": "quantitative", "label": "CV standard deviation"},
                    {"field": "sample_size", "type": "quantitative", "label": "Training rows"},
                ],
            },
            "yAxisTitle": "Mean CV accuracy",
            "valueFormat": "percent",
        },
    ]

    tables = [
        {
            "id": "confusion_table", "title": "Final SVM confusion matrix",
            "subtitle": "Rows are actual labels; columns are predicted labels",
            "showDescription": True, "dataset": "confusion", "sourceId": "confusion_source",
            "density": "spacious", "defaultSort": {"field": "actual_class", "direction": "asc"},
            "columns": [
                {"field": "actual_class", "label": "Actual class", "type": "number"},
                {"field": "predicted_0", "label": "Predicted 0", "type": "number"},
                {"field": "predicted_1", "label": "Predicted 1", "type": "number"},
            ],
        },
        {
            "id": "tuning_table", "title": "Training-only randomized-search candidates",
            "subtitle": "16 RBF configurations, five stratified folds per candidate",
            "showDescription": True, "dataset": "tuning", "sourceId": "tuning_source",
            "density": "compact", "defaultSort": {"field": "rank", "direction": "asc"},
            "columns": [
                {"field": "rank", "label": "Rank", "type": "number"},
                {"field": "C", "label": "C", "type": "number"},
                {"field": "gamma", "label": "Gamma", "type": "text"},
                {"field": "class_weight", "label": "Class weight", "type": "text"},
                {"field": "mean_cv_accuracy", "label": "Mean CV accuracy", "type": "percent", "format": "percent"},
                {"field": "std_cv_accuracy", "label": "CV standard deviation", "type": "percent", "format": "percent"},
            ],
        },
        {
            "id": "feature_table", "title": "Exact input feature disposition",
            "subtitle": "19 model inputs plus the rejected BMI field",
            "showDescription": True, "dataset": "features", "sourceId": "feature_source",
            "density": "comfortable", "defaultSort": {"field": "display_order", "direction": "asc"},
            "columns": [
                {"field": "display_order", "label": "Order", "type": "number"},
                {"field": "feature", "label": "Feature", "type": "text"},
                {"field": "treatment", "label": "Treatment", "type": "text"},
                {"field": "reason", "label": "Reason", "type": "text"},
            ],
        },
    ]

    title = "Leakage-Safe SVM Heart Disease Analysis"
    blocks = [
        {"id": "title", "type": "markdown", "body": f"# {title}"},
        {
            "id": "technical_summary", "type": "markdown", "sourceId": "summary_source",
            "body": (
                "## Technical Summary\n\n"
                "The final RBF SVM legitimately exceeds the 80% requirement: **96.15% test accuracy** "
                "and **0.9954 ROC-AUC** on a single evaluation of the held-out 10,000-row test set. "
                "Five-fold accuracy on the complete training partition averaged **95.95% ± 0.18 percentage points**. "
                "The previous near-random result came from loading the unrelated legacy CSV and its stale model, not from a lack of signal in the current dataset."
            ),
        },
        {"id": "metric_strip", "type": "metric-strip", "cardIds": [card["id"] for card in cards]},
        {
            "id": "performance_finding", "type": "markdown", "sourceId": "performance_source",
            "body": (
                "## The selected SVM clears both the requirement and the baseline\n\n"
                "Test accuracy is 42.50 percentage points above the 53.65% majority-class baseline and "
                "is consistent with training-only cross-validation. The chart shows the selection evidence "
                "and final evaluation on the same fractional scale; the 80% line is the assignment threshold."
            ),
        },
        {"id": "performance_chart_block", "type": "chart", "chartId": "performance_chart"},
        {
            "id": "confusion_finding", "type": "markdown", "sourceId": "confusion_source",
            "body": (
                "## Errors are balanced across the two outcome classes\n\n"
                "The SVM produced 198 false positives and 187 false negatives. This balance is consistent "
                "with precision (95.74%) and recall (95.97%) being nearly equal."
            ),
        },
        {"id": "confusion_table_block", "type": "table", "tableId": "confusion_table"},
        {
            "id": "bmi_finding", "type": "markdown", "sourceId": "bmi_source",
            "body": (
                "## Removing inconsistent BMI is the defensible feature treatment\n\n"
                "Supplied BMI has only 0.003 correlation with BMI recalculated from Weight and Height, and "
                "the median absolute mismatch is 7.43 BMI units. Training-only CV was 93.82% with supplied "
                "BMI, 93.84% with recalculated BMI, and 94.00% with BMI removed. Removal is slightly best "
                "and avoids retaining a corrupted or redundant field."
            ),
        },
        {"id": "bmi_chart_block", "type": "chart", "chartId": "bmi_chart"},
        {
            "id": "scope_definitions", "type": "markdown",
            "body": (
                "## Scope, Data, and Metric Definitions\n\n"
                "The source is `synthetic_heart_disease_dataset.csv`: 50,000 person-level synthetic rows, "
                "20 raw input columns, and binary target `Heart_Disease` (0 = no disease, 1 = disease). "
                "`Alcohol_Intake='None'` is preserved as a category; its 20,109 rows are not missing. "
                "Precision, recall, and F1 treat label 1 as positive. ROC-AUC uses `decision_function`, not probabilities."
            ),
        },
        {
            "id": "methodology", "type": "markdown",
            "body": (
                "## Model Specification and Validation\n\n"
                "A fixed stratified 80/20 split with `random_state=42` created 40,000 training and 10,000 test rows. "
                "`ColumnTransformer` applies median imputation and `StandardScaler` to numeric fields, plus constant "
                "imputation and `OneHotEncoder(handle_unknown='ignore')` to six categorical fields. All preprocessing "
                "is fitted inside each pipeline fold. Sixteen RBF candidates were evaluated with five-fold stratified "
                "CV on a fixed 12,000-row training-only tuning subset. The selected parameters are C=20, gamma=0.1, "
                "class_weight=None. Full-training five-fold validation preceded a single test evaluation."
            ),
        },
        {"id": "tuning_table_block", "type": "table", "tableId": "tuning_table"},
        {
            "id": "synthetic_limit", "type": "markdown",
            "body": (
                "## The target is deterministically rule-generated, not leaked\n\n"
                "A direct rule audit across all 40,000 training rows found an exact match using Age, Hypertension, "
                "Diabetes, Previous_Heart_Attack, and Cholesterol_Total. The recovered rule is: label 1 "
                "when at least two conditions hold—Age > 55, Hypertension=1, Diabetes=1, Previous_Heart_Attack=1, "
                "or Cholesterol_Total > 240. This explains the high predictability and limits real-world validity. "
                "It is not target leakage because `Heart_Disease` is absent from X, manual feature lists, preprocessing, "
                "and inference inputs."
            ),
        },
        {
            "id": "feature_schema", "type": "markdown",
            "body": (
                "## The web app, CLI, and saved pipeline use one exact schema\n\n"
                "All interfaces import the shared schema, submit raw values in the saved pipeline's 19-column order, "
                "and never accept the target. The table is the auditable feature contract."
            ),
        },
        {"id": "feature_table_block", "type": "table", "tableId": "feature_table"},
        {
            "id": "next_steps", "type": "markdown",
            "body": (
                "## Recommended Next Steps\n\n"
                "- Keep the dataset hash, schema assertions, alcohol-category assertion, and >=80% accuracy guard in automated tests.\n"
                "- Retrain whenever the CSV changes; do not reuse a model whose saved feature schema or dataset hash differs.\n"
                "- Treat these results as evidence about the assignment pipeline, not clinical performance."
            ),
        },
        {
            "id": "further_questions", "type": "markdown",
            "body": (
                "## Further Questions\n\n"
                "Would the assignment benefit from a second evaluation on a non-rule-generated dataset? "
                "If the synthetic generator is available, its source should be cited so the recovered threshold rule can be confirmed directly."
            ),
        },
    ]

    artifact = {
        "surface": "report",
        "manifest": {
            "version": 1, "surface": "report", "title": title,
            "description": "Technical validation of the final SVM classifier and current synthetic dataset.",
            "generatedAt": metrics["generated_at_utc"], "filters": [],
            "cards": cards, "charts": charts, "tables": tables,
            "sources": sources, "blocks": blocks,
        },
        "snapshot": {
            "version": 1, "generatedAt": metrics["generated_at_utc"], "status": "ready",
            "datasets": {
                "summary": summary_rows, "performance": performance_rows,
                "bmi_variants": bmi_rows, "tuning": tuning_rows,
                "features": feature_rows, "confusion": confusion_rows,
            },
            "accessIssues": [],
        },
        "sources": sources,
    }
    OUTPUT_PATH.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

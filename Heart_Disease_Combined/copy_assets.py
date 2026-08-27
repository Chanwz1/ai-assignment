"""Copy existing dataset and saved model artifacts into this combined folder."""

from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent

ASSETS = [
    (
        PROJECT_ROOT / "SVM" / "heart_disease_classifier" / "synthetic_heart_disease_dataset.csv",
        ROOT / "synthetic_heart_disease_dataset.csv",
    ),
    (
        PROJECT_ROOT / "ANN" / "AI Assignment" / "heart_disease_ann_model.pkl",
        ROOT / "models" / "heart_disease_ann_model.pkl",
    ),
    (
        PROJECT_ROOT / "ANN" / "AI Assignment" / "scaler.pkl",
        ROOT / "models" / "ann_scaler.pkl",
    ),
    (
        PROJECT_ROOT / "ANN" / "AI Assignment" / "feature_names.pkl",
        ROOT / "models" / "ann_feature_names.pkl",
    ),
    (
        PROJECT_ROOT / "SVM" / "heart_disease_classifier" / "models" / "svm_pipeline.joblib",
        ROOT / "models" / "svm_pipeline.joblib",
    ),
    (
        PROJECT_ROOT / "SVM" / "heart_disease_classifier" / "static" / "metrics.json",
        ROOT / "static" / "svm_metrics.json",
    ),
    (
        PROJECT_ROOT / "KNN" / "outputs_knn_v3" / "results_report.txt",
        ROOT / "static" / "knn_results_report.txt",
    ),
]


def main() -> None:
    for source, destination in ASSETS:
        if not source.exists():
            print(f"Missing source: {source}")
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        print(f"Copied {source.name} -> {destination}")


if __name__ == "__main__":
    main()

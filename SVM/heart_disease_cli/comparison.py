"""Display the saved SVM evaluation results."""

from utils import banner, load_metrics, press_enter, print_confusion_matrix, print_table, section


def show_performance(metrics: dict):
    test = metrics["test_metrics"]
    cv = metrics["full_training_cv"]
    section("Test-Set Metrics")
    print_table({
        "Accuracy": f"{test['accuracy'] * 100:.2f}%",
        "Precision": f"{test['precision'] * 100:.2f}%",
        "Recall": f"{test['recall'] * 100:.2f}%",
        "F1-score": f"{test['f1'] * 100:.2f}%",
        "ROC-AUC": f"{test['roc_auc'] * 100:.2f}%",
        "Majority baseline": f"{test['majority_class_baseline_accuracy'] * 100:.2f}%",
        "5-fold CV mean": f"{cv['mean_accuracy'] * 100:.2f}%",
        "5-fold CV standard deviation": f"{cv['std_accuracy'] * 100:.2f}%",
        "Training records": metrics["split"]["train_size"],
        "Test records": metrics["split"]["test_size"],
    })
    print(f"  CV scores             {[round(score, 6) for score in cv['scores']]}")
    section("Best SVM Parameters")
    for name, value in metrics["tuning"]["best_parameters"].items():
        print(f"  {name.removeprefix('classifier__'):<16} {value}")
    section("SVM Confusion Matrix")
    print_confusion_matrix(test["confusion_matrix"])
    print(
        "  The final test partition was evaluated once after training-only tuning.\n"
        "  The target is excluded from all model inputs, and preprocessing is fitted\n"
        "  inside the pipeline during each cross-validation fold."
    )
    section("Synthetic-Data Limitation")
    print(f"  {metrics['synthetic_rule_audit']['recovered_rule']}")


def run():
    banner("SVM MODEL PERFORMANCE")
    metrics = load_metrics()
    if metrics is None:
        print("\n  [ERROR] metrics.json is missing. Run Model Training first.")
    else:
        show_performance(metrics)
    press_enter()

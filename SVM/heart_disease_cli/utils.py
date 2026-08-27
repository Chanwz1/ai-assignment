"""Shared paths, data loaders, and terminal helpers for the SVM-only CLI."""

from pathlib import Path
import json
import sys

import joblib


CLI_DIR = Path(__file__).resolve().parent
CLASSIFIER_DIR = (CLI_DIR / ".." / "heart_disease_classifier").resolve()
if str(CLASSIFIER_DIR) not in sys.path:
    sys.path.insert(0, str(CLASSIFIER_DIR))

from project_config import (  # noqa: E402
    CATEGORICAL_FEATURES, CONFUSION_MATRIX_PATH, DATA_PATH as DATASET_PATH,
    METRICS_PATH, MODEL_FEATURES as FEATURES, MODEL_PATH as PIPELINE_PATH,
    NUMERICAL_FEATURES, TARGET as STATUS_COLUMN, load_dataset as load_current_dataset,
)

MODELS_DIR = CLASSIFIER_DIR / "models"
STATIC_DIR = CLASSIFIER_DIR / "static"
TRAIN_SCRIPT_PATH = CLASSIFIER_DIR / "train.py"
WIDTH = 72


def check_assets() -> list[str]:
    """Return missing required asset names without terminating the CLI."""
    required = [
        (DATASET_PATH, "synthetic_heart_disease_dataset.csv"),
        (PIPELINE_PATH, "models/svm_pipeline.joblib"),
    ]
    return [label for path, label in required if not path.is_file()]


def load_pipeline():
    return joblib.load(PIPELINE_PATH)


def load_dataset():
    return load_current_dataset() if DATASET_PATH.is_file() else None


def load_metrics():
    if not METRICS_PATH.is_file():
        return None
    return json.loads(METRICS_PATH.read_text(encoding="utf-8"))


def banner(title: str, width: int = WIDTH):
    print(f"\n{'=' * width}\n  {title}\n{'=' * width}")


def section(title: str, width: int = WIDTH):
    print(f"\n{'-' * width}\n  {title}\n{'-' * width}")


def print_table(data: dict, col_width: int = 32):
    for key, value in data.items():
        if isinstance(value, float):
            value = f"{value:.4f}"
        print(f"  {str(key):<{col_width}} {value}")


def print_confusion_matrix(matrix: list):
    print("\n                         Predicted No   Predicted Yes")
    print(f"  Actual No              {matrix[0][0]:<14} {matrix[0][1]}")
    print(f"  Actual Yes             {matrix[1][0]:<14} {matrix[1][1]}\n")


def press_enter():
    input("\n  Press ENTER to return to the menu...")


def prompt_float(prompt: str, low: float, high: float) -> float:
    while True:
        raw = input(f"  {prompt} [{low:g}-{high:g}]: ").strip()
        try:
            value = float(raw)
            if low <= value <= high:
                return value
        except ValueError:
            pass
        print(f"  [!] Enter a number from {low:g} to {high:g}.")


def prompt_choice(prompt: str, options: list[str]) -> int:
    for index, option in enumerate(options, start=1):
        print(f"    {index}. {option}")
    while True:
        raw = input(f"  {prompt} [1-{len(options)}]: ").strip()
        try:
            index = int(raw) - 1
            if 0 <= index < len(options):
                return index
        except ValueError:
            pass
        print(f"  [!] Enter a number from 1 to {len(options)}.")


def prompt_yn(prompt: str) -> bool:
    while True:
        value = input(f"  {prompt} [Y/N]: ").strip().lower()
        if value in {"y", "yes"}:
            return True
        if value in {"n", "no"}:
            return False
        print("  [!] Enter Y or N.")

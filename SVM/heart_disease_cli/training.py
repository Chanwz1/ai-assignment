"""Run the Streamlit project's single source-of-truth SVM training script."""

import subprocess
import sys

from comparison import show_performance
from utils import (
    CONFUSION_MATRIX_PATH, DATASET_PATH, METRICS_PATH, PIPELINE_PATH,
    TRAIN_SCRIPT_PATH, banner, load_metrics, press_enter, prompt_choice,
    prompt_yn, section,
)


ARTEFACTS = {
    "Dataset": DATASET_PATH,
    "Fitted SVM pipeline": PIPELINE_PATH,
    "Evaluation metrics": METRICS_PATH,
    "SVM confusion matrix": CONFUSION_MATRIX_PATH,
}


def _show_status():
    section("Current Training Artefacts")
    for label, path in ARTEFACTS.items():
        status = "Found" if path.is_file() else "Missing"
        size = f" ({path.stat().st_size / 1024:.1f} KB)" if path.is_file() else ""
        print(f"  {label:<26} {status}{size}")


def execute_training() -> int:
    """Execute the sibling training script and stream its console output."""
    section("SVM Training Progress")
    process = subprocess.Popen(
        [sys.executable, str(TRAIN_SCRIPT_PATH)],
        cwd=str(TRAIN_SCRIPT_PATH.parent),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        print(f"  {line.rstrip()}")
    return process.wait()


def run():
    while True:
        banner("MODEL TRAINING")
        choice = prompt_choice("Choose an option", [
            "Check training artefacts", "Run shared SVM training", "Back to Main Menu",
        ])
        if choice == 0:
            _show_status()
            press_enter()
        elif choice == 1:
            _show_status()
            print("\n  Training uses a fixed 80/20 stratified split and training-only five-fold")
            print("  accuracy tuning. The final test partition is evaluated exactly once.")
            if not prompt_yn("Start SVM training now?"):
                continue
            return_code = execute_training()
            if return_code != 0:
                print(f"\n  [ERROR] Training exited with status {return_code}.")
            else:
                print("\n  Training completed successfully.")
                metrics = load_metrics()
                if metrics:
                    show_performance(metrics)
            press_enter()
        else:
            break

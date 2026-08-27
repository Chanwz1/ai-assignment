"""Entry point for the SVM-only heart disease classification CLI."""

import comparison
import eda
import menu
import overview
import predictor
import training
from utils import banner, check_assets


def main():
    missing = check_assets()
    if missing:
        print("\n  [WARNING] Missing shared assets: " + ", ".join(missing))
        print("  Overview and training remain available; prediction requires the fitted pipeline.")
    actions = {
        1: overview.run,
        2: eda.run,
        3: comparison.run,
        4: predictor.run,
        5: training.run,
    }
    while True:
        choice = menu.display()
        if choice == 6:
            banner("GOODBYE")
            print("  This academic SVM prototype is not a medical diagnosis.\n")
            return
        actions[choice]()


if __name__ == "__main__":
    main()

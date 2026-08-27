"""Top-level CLI menu."""

from utils import banner


MENU_ITEMS = [
    "Overview Dashboard",
    "Exploratory Data Analysis",
    "SVM Model Performance",
    "Heart Disease Classification Prediction",
    "Model Training",
    "Exit",
]


def display() -> int:
    banner("HEART DISEASE CLASSIFICATION PREDICTION")
    print("  Academic SVM prototype - not a medical diagnosis.\n")
    for index, item in enumerate(MENU_ITEMS, start=1):
        print(f"    {index}. {item}")
    while True:
        raw = input(f"\n  Select option [1-{len(MENU_ITEMS)}]: ").strip()
        try:
            choice = int(raw)
            if 1 <= choice <= len(MENU_ITEMS):
                return choice
        except ValueError:
            pass
        print(f"  [!] Enter a number from 1 to {len(MENU_ITEMS)}.")

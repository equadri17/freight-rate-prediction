from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


# ============================================================
# PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"

PREDICTIONS_PATH = (
    OUTPUT_DIR / "validation_predictions.csv"
)

# Change this if your local development holdout
# predictions are stored somewhere else.
HOLDOUT_PATH = DATA_DIR / "holdout_predictions.csv"


# ============================================================
# EVALUATION
# ============================================================

def evaluate_predictions(
    actual,
    predicted
):
    """
    Calculate standard regression metrics.
    """

    mae = mean_absolute_error(
        actual,
        predicted
    )

    rmse = mean_squared_error(
        actual,
        predicted
    ) ** 0.5

    r2 = r2_score(
        actual,
        predicted
    )

    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
    }


def main():

    print("=" * 60)
    print("FREIGHT RATE PREDICTION - EVALUATION")
    print("=" * 60)

    if not HOLDOUT_PATH.exists():

        print(
            "\nNo holdout prediction file found."
        )

        print(
            "\nThis evaluation script is intended for "
            "development/validation predictions where "
            "actual posted_rate values are available."
        )

        print(
            "\nFor the assessment validation.csv, "
            "ground-truth posted_rate values are not available."
        )

        return

    predictions = pd.read_csv(
        HOLDOUT_PATH
    )

    required_columns = [
        "posted_rate",
        "predicted_rate",
    ]

    missing = [
        col
        for col in required_columns
        if col not in predictions.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns: {missing}"
        )

    metrics = evaluate_predictions(
        predictions["posted_rate"],
        predictions["predicted_rate"]
    )

    print("\nValidation Metrics")
    print("-" * 30)

    print(
        f"MAE  : {metrics['MAE']:.4f}"
    )

    print(
        f"RMSE : {metrics['RMSE']:.4f}"
    )

    print(
        f"R²   : {metrics['R2']:.4f}"
    )


if __name__ == "__main__":
    main()

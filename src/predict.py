from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from features import create_features


# ============================================================
# PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "outputs"

MODEL_PATH = MODEL_DIR / "catboost_freight_rate_model.cbm"

VALIDATION_PATH = DATA_DIR / "validation.csv"
TEMPLATE_PATH = DATA_DIR / "validation_predictions_template.csv"

VALIDATION_OUTPUT = (
    OUTPUT_DIR / "validation_predictions.csv"
)


# ============================================================
# MODEL FEATURES
# ============================================================

FEATURES = [
    "pickup",
    "delivery",
    "pickup_lat",
    "pickup_lon",
    "delivery_lat",
    "delivery_lon",
    "distance",
    "equipment",
    "weight",
    "date",
    "market_index",
    "quote_signal",
    "year",
    "month",
    "day",
    "dayofweek",
    "dayofyear",
    "weekofyear",
    "sin_doy",
    "cos_doy",
    "sin_dow",
    "cos_dow",
    "distance_log",
    "distance_sq",
    "weight_log",
    "weight_distance_ratio",
    "route",
]


CATEGORICAL_FEATURES = [
    "pickup",
    "delivery",
    "equipment",
    "route",
]


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():
    """Load the trained CatBoost model."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}\n"
            "Run src/train.py first."
        )

    model = CatBoostRegressor()

    model.load_model(MODEL_PATH)

    return model


# ============================================================
# PREDICTION
# ============================================================

def predict_rates(model, X):
    """
    Generate predictions and convert them from log scale
    back to the original freight-rate scale.
    """

    prediction_log = model.predict(X)

    prediction = np.expm1(
        prediction_log
    )

    prediction = np.maximum(
        prediction,
        0.01
    )

    return prediction


# ============================================================
# MAIN PREDICTION PIPELINE
# ============================================================

def main():

    print("=" * 60)
    print("FREIGHT RATE PREDICTION - INFERENCE")
    print("=" * 60)

    OUTPUT_DIR.mkdir(exist_ok=True)

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    print("\nLoading trained model...")

    model = load_model()

    print("Model loaded successfully.")

    # --------------------------------------------------------
    # Load validation data
    # --------------------------------------------------------

    print("\nLoading validation data...")

    validation = pd.read_csv(
        VALIDATION_PATH
    )

    template = pd.read_csv(
        TEMPLATE_PATH
    )

    print(
        f"Validation rows: {len(validation):,}"
    )

    # --------------------------------------------------------
    # Feature engineering
    # --------------------------------------------------------

    print("\nCreating prediction features...")

    validation_features = create_features(
        validation
    )

    # --------------------------------------------------------
    # Prepare model input
    # --------------------------------------------------------

    missing_features = [
        col
        for col in FEATURES
        if col not in validation_features.columns
    ]

    if missing_features:
        raise ValueError(
            f"Missing features: {missing_features}"
        )

    X_validation = validation_features[
        FEATURES
    ]

    # --------------------------------------------------------
    # Generate predictions
    # --------------------------------------------------------

    print("\nGenerating predictions...")

    predictions = predict_rates(
        model,
        X_validation
    )

    # --------------------------------------------------------
    # Create output
    # --------------------------------------------------------

    output = template.copy()

    output["predicted_rate"] = predictions

    # --------------------------------------------------------
    # Validation checks
    # --------------------------------------------------------

    if len(output) != len(validation):
        raise ValueError(
            "Prediction row count does not match "
            "validation data."
        )

    if output["load_id"].nunique() != len(output):
        raise ValueError(
            "Validation load_id values are not unique."
        )

    if output["predicted_rate"].isna().any():
        raise ValueError(
            "Missing predictions found."
        )

    if (output["predicted_rate"] <= 0).any():
        raise ValueError(
            "Non-positive predictions found."
        )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    output.to_csv(
        VALIDATION_OUTPUT,
        index=False
    )

    print(
        f"\nSaved predictions to: "
        f"{VALIDATION_OUTPUT}"
    )

    print(
        f"Prediction range: "
        f"{output['predicted_rate'].min():.2f}"
        f" to "
        f"{output['predicted_rate'].max():.2f}"
    )

    print(
        f"Total predictions: {len(output):,}"
    )

    print("\nInference completed successfully.")


if __name__ == "__main__":
    main()

from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

TRAIN_PATH = DATA_DIR / "train_test.csv"
VALIDATION_PATH = DATA_DIR / "validation.csv"
TEMPLATE_PATH = DATA_DIR / "validation_predictions_template.csv"
DECEMBER_PATH = DATA_DIR / "december_chart_inputs.csv"

VALIDATION_OUTPUT = ROOT / "validation_predictions.csv"
DECEMBER_OUTPUT = ROOT / "december_chart_inputs_completed.csv"


def create_features(df):
    """
    Apply the same feature engineering used during
    model development.
    """

    df = df.copy()

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce"
    )

    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["dayofweek"] = df["date"].dt.dayofweek
    df["dayofyear"] = df["date"].dt.dayofyear

    df["weekofyear"] = (
        df["date"]
        .dt.isocalendar()
        .week
        .astype(int)
    )


    df["sin_doy"] = np.sin(
        2 * np.pi * df["dayofyear"] / 365
    )

    df["cos_doy"] = np.cos(
        2 * np.pi * df["dayofyear"] / 365
    )

    df["sin_dow"] = np.sin(
        2 * np.pi * df["dayofweek"] / 7
    )

    df["cos_dow"] = np.cos(
        2 * np.pi * df["dayofweek"] / 7
    )


    if "distance" in df.columns:

        df["distance_log"] = np.log1p(
            df["distance"].clip(lower=0)
        )

        df["distance_sq"] = (
            df["distance"] ** 2
        )


    if "weight" in df.columns:

        df["weight_log"] = np.log1p(
            df["weight"].clip(lower=0)
        )

    if (
        "weight" in df.columns
        and "distance" in df.columns
    ):

        df["weight_distance_ratio"] = (
            df["weight"]
            / (df["distance"] + 1)
        )


    if (
        "pickup" in df.columns
        and "delivery" in df.columns
    ):

        df["route"] = (
            df["pickup"].astype(str)
            + "_"
            + df["delivery"].astype(str)
        )

    return df

def prepare_training_data(train):

    train_features = create_features(train)

    target = "posted_rate"

    features = [
        col
        for col in train_features.columns
        if col != target
    ]

    if "load_id" in features:
        features.remove("load_id")

    categorical_features = [
        "pickup",
        "delivery",
        "equipment",
        "route"
    ]

    categorical_features = [
        col
        for col in categorical_features
        if col in features
    ]

    cat_indices = [
        features.index(col)
        for col in categorical_features
    ]

    X_full = train_features[features]
    y_full = train_features[target]

    return (
        train_features,
        X_full,
        y_full,
        features,
        cat_indices
    )


def train_final_model(
    X_full,
    y_full,
    cat_indices
):

    model = CatBoostRegressor(
        iterations=300,
        depth=6,
        learning_rate=0.05,
        l2_leaf_reg=5,
        loss_function="RMSE",
        random_seed=42,
        verbose=100
    )

    y_full_log = np.log1p(y_full)

    model.fit(
        X_full,
        y_full_log,
        cat_features=cat_indices
    )

    return model

def predict_rates(model, X):

    prediction_log = model.predict(X)

    prediction = np.expm1(
        prediction_log
    )

    prediction = np.maximum(
        prediction,
        0.01
    )

    return prediction

def create_validation_predictions(
    model,
    validation,
    template,
    features
):

    validation_features = create_features(
        validation
    )

    X_validation = validation_features[
        features
    ]

    predictions = predict_rates(
        model,
        X_validation
    )

    output = template.copy()

    output["predicted_rate"] = predictions

    return output


def create_december_predictions(
    model,
    train,
    december,
    features
):

    december_model = december.copy()

    december_model["date"] = pd.to_datetime(
        december_model["date"],
        errors="coerce"
    )


    pickup_coordinates = (
        train.groupby("pickup")[
            ["pickup_lat", "pickup_lon"]
        ]
        .median()
    )

    delivery_coordinates = (
        train.groupby("delivery")[
            ["delivery_lat", "delivery_lon"]
        ]
        .median()
    )

    december_model["pickup_lat"] = (
        december_model["pickup"]
        .map(
            pickup_coordinates["pickup_lat"]
        )
    )

    december_model["pickup_lon"] = (
        december_model["pickup"]
        .map(
            pickup_coordinates["pickup_lon"]
        )
    )

    december_model["delivery_lat"] = (
        december_model["delivery"]
        .map(
            delivery_coordinates["delivery_lat"]
        )
    )

    december_model["delivery_lon"] = (
        december_model["delivery"]
        .map(
            delivery_coordinates["delivery_lon"]
        )
    )


    weight_median = train["weight"].median()

    december_model["weight"] = (
        december_model["weight"]
        .fillna(weight_median)
    )


    recent_data = train[
        train["date"] >= "2025-10-01"
    ].copy()

    market_index_recent_median = (
        recent_data["market_index"].median()
    )

    quote_signal_recent_median = (
        recent_data["quote_signal"].median()
    )

    december_model["market_index"] = (
        market_index_recent_median
    )

    december_model["quote_signal"] = (
        quote_signal_recent_median
    )


    december_model = create_features(
        december_model
    )

    missing_features = [
        col
        for col in features
        if col not in december_model.columns
    ]

    if missing_features:
        raise ValueError(
            "Missing December features: "
            + str(missing_features)
        )

    X_december = december_model[
        features
    ]

    predictions = predict_rates(
        model,
        X_december
    )

    output = december.copy()

    output["predicted_rate"] = predictions

    return output


def validate_validation_output(
    predictions
):

    if len(predictions) != 12000:
        raise ValueError(
            f"Expected 12,000 predictions, "
            f"got {len(predictions)}"
        )

    if predictions["load_id"].nunique() != 12000:
        raise ValueError(
            "Validation load_id values are not unique."
        )

    if predictions[
        "predicted_rate"
    ].isna().any():

        raise ValueError(
            "Missing validation predictions found."
        )

    if (
        predictions["predicted_rate"] <= 0
    ).any():

        raise ValueError(
            "Non-positive validation predictions found."
        )


def validate_december_output(
    predictions
):

    if len(predictions) != 31:
        raise ValueError(
            f"Expected 31 December rows, "
            f"got {len(predictions)}"
        )

    if predictions["date"].nunique() != 31:
        raise ValueError(
            "December dates are not unique."
        )

    if predictions[
        "predicted_rate"
    ].isna().any():

        raise ValueError(
            "Missing December predictions found."
        )

    if (
        predictions["predicted_rate"] <= 0
    ).any():

        raise ValueError(
            "Non-positive December predictions found."
        )


def main():

    print("=" * 60)
    print("FREIGHT RATE PREDICTION")
    print("=" * 60)


    print("\nLoading data...")

    train = pd.read_csv(
        TRAIN_PATH
    )

    validation = pd.read_csv(
        VALIDATION_PATH
    )

    template = pd.read_csv(
        TEMPLATE_PATH
    )

    december = pd.read_csv(
        DECEMBER_PATH
    )

    print(
        f"Training rows   : {len(train):,}"
    )

    print(
        f"Validation rows : {len(validation):,}"
    )

    print(
        f"December rows   : {len(december):,}"
    )


    print("\nCreating training features...")

    (
        train_features,
        X_full,
        y_full,
        features,
        cat_indices
    ) = prepare_training_data(train)

    print(
        f"Number of features: {len(features)}"
    )

    print(
        "Categorical feature indices:",
        cat_indices
    )


    print("\nTraining final CatBoost model...")

    model = train_final_model(
        X_full,
        y_full,
        cat_indices
    )

    print("\nModel training complete.")

    print(
        "\nGenerating validation predictions..."
    )

    validation_predictions = (
        create_validation_predictions(
            model,
            validation,
            template,
            features
        )
    )

    validate_validation_output(
        validation_predictions
    )

    validation_predictions.to_csv(
        VALIDATION_OUTPUT,
        index=False
    )

    print(
        f"Saved: {VALIDATION_OUTPUT}"
    )

    print(
        "Prediction range:",
        validation_predictions[
            "predicted_rate"
        ].min(),
        "to",
        validation_predictions[
            "predicted_rate"
        ].max()
    )

    print(
        "\nGenerating December predictions..."
    )

    december_predictions = (
        create_december_predictions(
            model,
            train,
            december,
            features
        )
    )

    validate_december_output(
        december_predictions
    )

    december_predictions.to_csv(
        DECEMBER_OUTPUT,
        index=False
    )

    print(
        f"Saved: {DECEMBER_OUTPUT}"
    )

 
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 60)

    print(
        "\nOutput files:"
    )

    print(
        f"1. {VALIDATION_OUTPUT}"
    )

    print(
        f"2. {DECEMBER_OUTPUT}"
    )


if __name__ == "__main__":
    main()

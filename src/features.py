import numpy as np
import pandas as pd


def create_features(df):
    """
    Apply the feature engineering used by the freight-rate
    prediction model.
    """

    df = df.copy()

    # --------------------------------------------------------
    # Date features
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Cyclical date features
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Distance features
    # --------------------------------------------------------

    if "distance" in df.columns:

        df["distance_log"] = np.log1p(
            df["distance"].clip(lower=0)
        )

        df["distance_sq"] = (
            df["distance"] ** 2
        )

    # --------------------------------------------------------
    # Weight features
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Route feature
    # --------------------------------------------------------

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

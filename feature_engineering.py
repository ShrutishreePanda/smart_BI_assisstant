from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


FEATURE_COLUMNS = [
    "TransactionAmount",
    "TransactionHour",
    "TransactionDayOfWeek",
    "IsWeekend",
    "TransactionDistanceKm",
    "CustomerAge",
    "CityPopulation",
    "TransactionCategoryEncoded",
    "IsFraud",
]

FEATURE_OUTPUT_PATH = Path("data/UnifiedTransactionFeatures.csv")


def generate_unified_features(raw_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    df = raw_df.copy()
    mappings: dict[str, str] = {}

    if all(column in df.columns for column in FEATURE_COLUMNS):
        feature_df = df[FEATURE_COLUMNS].copy()
        return clean_feature_frame(feature_df), {
            column: column for column in FEATURE_COLUMNS
        }

    amount = numeric_series(df, ["TransactionAmount", "transaction_amount", "amt", "amount"])
    mappings["TransactionAmount"] = amount.name or "numeric amount-like column"

    transaction_date = datetime_series(
        df,
        [
            "trans_date_trans_time",
            "transaction_datetime",
            "transaction_date",
            "date",
            "datetime",
            "timestamp",
            "time",
        ],
    )

    hour = existing_or_derived_hour(df, transaction_date)
    mappings["TransactionHour"] = hour.name or "derived from transaction datetime"

    day_of_week = existing_or_derived_day(df, transaction_date)
    mappings["TransactionDayOfWeek"] = day_of_week.name or "derived from transaction datetime"

    is_weekend = existing_or_derived_weekend(df, day_of_week)
    mappings["IsWeekend"] = is_weekend.name or "derived from transaction day"

    distance = distance_series(df)
    mappings["TransactionDistanceKm"] = distance.name or "derived from customer and merchant coordinates"

    customer_age = age_series(df, transaction_date)
    mappings["CustomerAge"] = customer_age.name or "derived from date of birth"

    city_population = numeric_series(
        df,
        ["CityPopulation", "city_pop", "population", "customer_city_population"],
        fallback=0,
    )
    mappings["CityPopulation"] = city_population.name or "defaulted when population was unavailable"

    category_encoded = category_series(df)
    mappings["TransactionCategoryEncoded"] = category_encoded.name or "encoded from category-like column"

    fraud = fraud_series(df)
    mappings["IsFraud"] = fraud.name or "binary target inferred from data"

    feature_df = pd.DataFrame(
        {
            "TransactionAmount": amount,
            "TransactionHour": hour,
            "TransactionDayOfWeek": day_of_week,
            "IsWeekend": is_weekend,
            "TransactionDistanceKm": distance,
            "CustomerAge": customer_age,
            "CityPopulation": city_population,
            "TransactionCategoryEncoded": category_encoded,
            "IsFraud": fraud,
        }
    )

    return clean_feature_frame(feature_df), mappings


def save_unified_features(feature_df: pd.DataFrame) -> Path:
    FEATURE_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    feature_df.to_csv(FEATURE_OUTPUT_PATH, index=False)
    return FEATURE_OUTPUT_PATH


def clean_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df[FEATURE_COLUMNS].copy()
    for column in FEATURE_COLUMNS:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")

    for column in cleaned.columns:
        if cleaned[column].isna().all():
            cleaned[column] = 0
        else:
            cleaned[column] = cleaned[column].fillna(cleaned[column].median())

    cleaned["TransactionHour"] = cleaned["TransactionHour"].clip(0, 23).round().astype(int)
    cleaned["TransactionDayOfWeek"] = cleaned["TransactionDayOfWeek"].clip(0, 6).round().astype(int)
    cleaned["IsWeekend"] = cleaned["IsWeekend"].clip(0, 1).round().astype(int)
    cleaned["IsFraud"] = cleaned["IsFraud"].clip(0, 1).round().astype(int)
    cleaned["TransactionCategoryEncoded"] = cleaned["TransactionCategoryEncoded"].round().astype(int)

    if cleaned["IsFraud"].nunique(dropna=True) < 2:
        raise ValueError(
            "Could not build a usable IsFraud target. Upload raw data with a fraud flag or another binary target column."
        )

    return cleaned


def first_existing_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lookup = {column.lower(): column for column in df.columns}
    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]
    return None


def numeric_series(df: pd.DataFrame, candidates: list[str], fallback: float | None = None) -> pd.Series:
    column = first_existing_column(df, candidates)
    if column:
        series = pd.to_numeric(df[column], errors="coerce")
        series.name = column
        return series

    numeric_columns = list(df.select_dtypes(include="number").columns)
    if numeric_columns:
        series = pd.to_numeric(df[numeric_columns[0]], errors="coerce")
        series.name = numeric_columns[0]
        return series

    if fallback is not None:
        series = pd.Series(fallback, index=df.index)
        series.name = None
        return series

    raise ValueError(f"Could not find a numeric source column for any of: {candidates}")


def datetime_series(df: pd.DataFrame, candidates: list[str]) -> pd.Series | None:
    for column in candidates:
        source = first_existing_column(df, [column])
        if not source:
            continue

        parsed = pd.to_datetime(df[source], errors="coerce")
        if parsed.notna().sum() > 0:
            parsed.name = source
            return parsed

    return None


def existing_or_derived_hour(df: pd.DataFrame, transaction_date: pd.Series | None) -> pd.Series:
    column = first_existing_column(df, ["TransactionHour", "transaction_hour", "hour"])
    if column:
        series = pd.to_numeric(df[column], errors="coerce")
        series.name = column
        return series
    if transaction_date is not None:
        series = transaction_date.dt.hour
        series.name = None
        return series
    series = pd.Series(0, index=df.index)
    series.name = None
    return series


def existing_or_derived_day(df: pd.DataFrame, transaction_date: pd.Series | None) -> pd.Series:
    column = first_existing_column(df, ["TransactionDayOfWeek", "transaction_day_of_week", "day_of_week"])
    if column:
        series = pd.to_numeric(df[column], errors="coerce")
        series.name = column
        return series
    if transaction_date is not None:
        series = transaction_date.dt.dayofweek
        series.name = None
        return series
    series = pd.Series(0, index=df.index)
    series.name = None
    return series


def existing_or_derived_weekend(df: pd.DataFrame, day_of_week: pd.Series) -> pd.Series:
    column = first_existing_column(df, ["IsWeekend", "is_weekend", "weekend"])
    if column:
        series = pd.to_numeric(df[column], errors="coerce")
        series.name = column
        return series
    series = day_of_week.isin([5, 6]).astype(int)
    series.name = None
    return series


def distance_series(df: pd.DataFrame) -> pd.Series:
    column = first_existing_column(df, ["TransactionDistanceKm", "transaction_distance_km", "distance_km"])
    if column:
        series = pd.to_numeric(df[column], errors="coerce")
        series.name = column
        return series

    required = ["lat", "long", "merch_lat", "merch_long"]
    if all(first_existing_column(df, [column]) for column in required):
        lat1 = pd.to_numeric(df[first_existing_column(df, ["lat"])], errors="coerce")
        lon1 = pd.to_numeric(df[first_existing_column(df, ["long"])], errors="coerce")
        lat2 = pd.to_numeric(df[first_existing_column(df, ["merch_lat"])], errors="coerce")
        lon2 = pd.to_numeric(df[first_existing_column(df, ["merch_long"])], errors="coerce")
        series = haversine_km(lat1, lon1, lat2, lon2)
        series.name = None
        return series

    series = pd.Series(0, index=df.index)
    series.name = None
    return series


def haversine_km(lat1: pd.Series, lon1: pd.Series, lat2: pd.Series, lon2: pd.Series) -> pd.Series:
    radius_km = 6371.0
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)
    a = np.sin(delta_phi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2) ** 2
    return pd.Series(2 * radius_km * np.arctan2(np.sqrt(a), np.sqrt(1 - a)), index=lat1.index)


def age_series(df: pd.DataFrame, transaction_date: pd.Series | None) -> pd.Series:
    column = first_existing_column(df, ["CustomerAge", "customer_age", "age"])
    if column:
        series = pd.to_numeric(df[column], errors="coerce")
        series.name = column
        return series

    dob_column = first_existing_column(df, ["dob", "date_of_birth", "birth_date"])
    if dob_column:
        dob = pd.to_datetime(df[dob_column], errors="coerce")
        reference = transaction_date if transaction_date is not None else pd.Series(pd.Timestamp.today(), index=df.index)
        series = ((reference - dob).dt.days / 365.25).clip(lower=0)
        series.name = dob_column
        return series

    series = pd.Series(0, index=df.index)
    series.name = None
    return series


def category_series(df: pd.DataFrame) -> pd.Series:
    column = first_existing_column(
        df,
        ["TransactionCategoryEncoded", "transaction_category_encoded", "category_encoded"],
    )
    if column:
        series = pd.to_numeric(df[column], errors="coerce")
        series.name = column
        return series

    category_column = first_existing_column(df, ["category", "transaction_category", "merchant_category"])
    if not category_column:
        object_columns = list(df.select_dtypes(include=["object", "category"]).columns)
        category_column = object_columns[0] if object_columns else None

    if category_column:
        codes = pd.Categorical(df[category_column].fillna("Missing")).codes
        series = pd.Series(codes, index=df.index)
        series.name = category_column
        return series

    series = pd.Series(0, index=df.index)
    series.name = None
    return series


def fraud_series(df: pd.DataFrame) -> pd.Series:
    column = first_existing_column(df, ["IsFraud", "is_fraud", "fraud", "fraud_flag", "target"])
    if column:
        series = pd.to_numeric(df[column], errors="coerce")
        series.name = column
        return series

    binary_columns = [
        column
        for column in df.columns
        if pd.to_numeric(df[column], errors="coerce").dropna().nunique() == 2
    ]
    if binary_columns:
        column = binary_columns[0]
        series = pd.to_numeric(df[column], errors="coerce")
        series.name = column
        return series

    raise ValueError("Could not infer IsFraud. Upload raw data with a fraud flag or binary target column.")

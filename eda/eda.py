import os
import pandas as pd
import numpy as np

RAW_DATA_PATH = "../data/5000_rows.csv"
OUTPUT_DATA_PATH = "../data/UnifiedTransactionFeatures.csv"

os.makedirs("../data", exist_ok=True)



REQUIRED_RAW_COLUMNS = {
    "trans_date_trans_time",
    "dob",
    "lat",
    "long",
    "merch_lat",
    "merch_long",
    "category",
    "city_pop",
    "amt",
    "is_fraud",
}

df = pd.read_csv(RAW_DATA_PATH)

missing_raw = REQUIRED_RAW_COLUMNS - set(df.columns)
if missing_raw:
    raise ValueError(f"Missing required raw columns: {missing_raw}")



df["TransactionDateTime"] = pd.to_datetime(
    df["trans_date_trans_time"], errors="coerce"
)

df["TransactionHour"] = df["TransactionDateTime"].dt.hour
df["TransactionDayOfWeek"] = df["TransactionDateTime"].dt.dayofweek
df["IsWeekend"] = df["TransactionDayOfWeek"].isin([5, 6]).astype(int)

df["DateOfBirth"] = pd.to_datetime(df["dob"], errors="coerce")
df["CustomerAge"] = (
    (pd.Timestamp.today() - df["DateOfBirth"]).dt.days / 365.25
).round(0)

R = 6371  # Earth radius in km

lat1 = np.radians(df["lat"])
lon1 = np.radians(df["long"])
lat2 = np.radians(df["merch_lat"])
lon2 = np.radians(df["merch_long"])

df["TransactionDistanceKm"] = (
    2 * R *
    np.arcsin(
        np.sqrt(
            np.sin((lat2 - lat1) / 2) ** 2
            + np.cos(lat1)
            * np.cos(lat2)
            * np.sin((lon2 - lon1) / 2) ** 2
        )
    )
)

df["TransactionCategoryEncoded"] = (
    df["category"].astype("category").cat.codes
)

numeric_cols = [
    "TransactionHour",
    "TransactionDayOfWeek",
    "IsWeekend",
    "TransactionDistanceKm",
    "CustomerAge",
    "city_pop",
    "TransactionCategoryEncoded",
    "amt",
]

df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())



final_df = df[
    [
        "amt",
        "TransactionHour",
        "TransactionDayOfWeek",
        "IsWeekend",
        "TransactionDistanceKm",
        "CustomerAge",
        "city_pop",
        "TransactionCategoryEncoded",
        "is_fraud",
    ]
].rename(
    columns={
        "amt": "TransactionAmount",
        "city_pop": "CityPopulation",
        "is_fraud": "IsFraud",
    }
)



REQUIRED_FEATURE_COLUMNS = {
    "TransactionAmount",
    "TransactionHour",
    "TransactionDayOfWeek",
    "IsWeekend",
    "TransactionDistanceKm",
    "CustomerAge",
    "CityPopulation",
    "TransactionCategoryEncoded",
    "IsFraud",
}

missing_features = REQUIRED_FEATURE_COLUMNS - set(final_df.columns)
if missing_features:
    raise ValueError(f"Missing required feature columns: {missing_features}")

if final_df.isnull().any().any():
    raise ValueError("Final dataset contains NaN values")



final_df.to_csv(OUTPUT_DATA_PATH, index=False)
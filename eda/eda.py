import warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")

MODEL_TYPE: str  = "logistic"   # "logistic" | "linear" | "kmeans"
EXPORT_CSV: bool = True        # True → writes CSV for selected model

DATA_PATH: str = r"C:\smart_BI_assisstant\eda\credit_card_transactions.csv"

COL_MAP = {
    "id_cols": ["trans_num", "trans_id", "cc_num"],
    "entity_col": "cc_num",

    "fraud_target": "is_fraud",
    "regression_target": "amt",

    "datetime_col": "trans_date_trans_time",
    "amount_col": "amt",

    "lat_col": "lat",
    "long_col": "long",
    "merch_lat_col": "merch_lat",
    "merch_long_col": "merch_long",
    "city_pop_col": "city_pop",

    "category_col": "category",
    "gender_col": "gender",
    "state_col": "state",
    "job_col": "job",

    "dob_col": "dob",
    "merch_col": "merchant",
}

def _col(k): return COL_MAP.get(k)

def load_data(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    df = pd.read_csv(path, low_memory=False)
    print(f"\nLoaded dataset → {df.shape[0]:,} rows × {df.shape[1]} columns")
    return df

def shared_eda(df: pd.DataFrame) -> None:
    print("\nRunning shared EDA...")
    print(df.dtypes)
    print("\nMissing values:")
    print(df.isnull().sum())
    print("\nDuplicates:", df.duplicated().sum())

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    drop_cols = [c for c in COL_MAP["id_cols"] + [COL_MAP["merch_col"]] if c in df.columns]
    df.drop(columns=drop_cols, inplace=True, errors="ignore")

    dt_col = _col("datetime_col")
    if dt_col in df.columns:
        df[dt_col] = pd.to_datetime(df[dt_col], errors="coerce")
        df["hour"] = df[dt_col].dt.hour
        df["day_of_week"] = df[dt_col].dt.dayofweek
        df["month"] = df[dt_col].dt.month
        df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

    dob_col = _col("dob_col")
    if dob_col in df.columns:
        dob = pd.to_datetime(df[dob_col], errors="coerce")
        df["age"] = ((pd.Timestamp("today") - dob).dt.days / 365.25).round(1)
        df.drop(columns=[dob_col], inplace=True)

    if all(k in df.columns for k in [
        _col("lat_col"), _col("long_col"),
        _col("merch_lat_col"), _col("merch_long_col")
    ]):
        R = 6371
        lat1, lon1 = np.radians(df[_col("lat_col")]), np.radians(df[_col("long_col")])
        lat2, lon2 = np.radians(df[_col("merch_lat_col")]), np.radians(df[_col("merch_long_col")])
        dlat, dlon = lat2 - lat1, lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
        df["distance_km"] = 2 * R * np.arcsin(np.sqrt(a))

    cat_cols = [_col("category_col"), _col("gender_col"), _col("state_col")]
    cat_cols = [c for c in cat_cols if c in df.columns]
    if cat_cols:
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    df[df.select_dtypes(np.number).columns] = df.select_dtypes(np.number).fillna(
        df.select_dtypes(np.number).median()
    )

    return df

# ---------------- MODEL-SPECIFIC EXTRACTION ---------------- #

def extract_logistic_features(df: pd.DataFrame) -> pd.DataFrame:
    target = _col("fraud_target")
    features = [
        _col("amount_col"), "hour", "day_of_week", "is_weekend",
        "distance_km", "age", _col("city_pop_col")
    ]
    features = [f for f in features if f and f in df.columns]
    df = df[features + [target]]
    return df

def extract_linear_features(df: pd.DataFrame) -> pd.DataFrame:
    target = _col("regression_target")
    features = [
        "hour", "day_of_week", "month", "is_weekend",
        "distance_km", "age", _col("city_pop_col")
    ]
    features = [f for f in features if f and f in df.columns]
    df = df[features + [target]]
    return df

def extract_kmeans_features(df: pd.DataFrame) -> pd.DataFrame:
    entity = _col("entity_col")
    amt = _col("amount_col")

    agg = df.groupby(entity).agg(
        tx_count=(amt, "count"),
        total_spend=(amt, "sum"),
        avg_tx_amt=(amt, "mean"),
        std_tx_amt=(amt, "std"),
    ).reset_index()

    agg["std_tx_amt"] = agg["std_tx_amt"].fillna(0)
    return agg.drop(columns=[entity])

# ---------------- RUN PIPELINE ---------------- #

def run_eda(model_type: str):
    df = load_data(DATA_PATH)
    shared_eda(df)
    df = clean_data(df)

    if model_type == "logistic":
        feat_df = extract_logistic_features(df)
        out_name = "features_logistic.csv"

    elif model_type == "linear":
        feat_df = extract_linear_features(df)
        out_name = "features_linear.csv"

    else:
        feat_df = extract_kmeans_features(df)
        out_name = "features_kmeans.csv"

    if EXPORT_CSV:
        feat_df.to_csv(out_name, index=False)
        print(f"\n✅ Exported {out_name} → {feat_df.shape}")

    return feat_df

if __name__ == "__main__":
    final_features = run_eda(MODEL_TYPE)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option("display.max_columns", None)
sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (8, 5)

df = pd.read_csv("credit_card_transactions.csv")

print("Dataset Shape:", df.shape)
print("\nData Types:")
print(df.dtypes)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:", df.duplicated().sum())

drop_cols = [
    "Unnamed: 0", "cc_num", "trans_num",
    "first", "last", "street"
]
df = df.drop(columns=[c for c in drop_cols if c in df.columns])

if "trans_date_trans_time" in df.columns:
    df["trans_date_trans_time"] = pd.to_datetime(
        df["trans_date_trans_time"], errors="coerce"
    )
    df["transaction_hour"] = df["trans_date_trans_time"].dt.hour
    df["transaction_day"] = df["trans_date_trans_time"].dt.dayofweek

if "is_fraud" in df.columns:
    print("\nFraud Target Distribution:")
    print(df["is_fraud"].value_counts(normalize=True))

    sns.countplot(x="is_fraud", data=df)
    plt.title("Fraud vs Non-Fraud Distribution")
    plt.show()

num_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
num_cols = [c for c in num_cols if c != "is_fraud"]

for col in num_cols:
    sns.histplot(df[col].dropna(), kde=True)
    plt.title(f"Distribution: {col}")
    plt.show()

corr_cols = num_cols.copy()
if "is_fraud" in df.columns:
    corr_cols.append("is_fraud")

corr_matrix = df[corr_cols].corr()

sns.heatmap(
    corr_matrix,
    cmap="coolwarm",
    center=0
)
plt.title("Feature Correlation Matrix")
plt.show()

if "job" in df.columns:
    behavioral_df = df.groupby("job").agg({
        "amt": ["mean", "sum", "count"],
        "transaction_hour": "nunique"
    })

    behavioral_df.columns = [
        "avg_amount", "total_amount",
        "txn_count", "active_hours"
    ]

    print("\nBehavioral Aggregated Sample:")
    print(behavioral_df.head())

    behavioral_df.hist()
    plt.suptitle("Behavioral Feature Distributions")
    plt.show()

FINAL_FEATURES = {
    "logistic_regression": [
        "amt", "transaction_hour", "transaction_day",
        "category", "gender", "state"
    ],
    "linear_regression": [
        "amt", "transaction_hour", "transaction_day",
        "city_pop"
    ],
    "kmeans_clustering": [
        "avg_amount", "total_amount",
        "txn_count", "active_hours"
    ]
}

print("\nFinal Feature Contract:")
for model, features in FINAL_FEATURES.items():
    print(f"\n{model.upper()}:")
    for feature in features:
        print(" -", feature)


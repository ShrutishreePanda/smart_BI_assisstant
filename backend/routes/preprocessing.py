import pandas as pd
from sklearn.preprocessing import StandardScaler


def preprocess(df, target=None):
    df = df.copy()

    # Drop ID columns
    df = df.drop(columns=[c for c in df.columns if "id" in c.lower()], errors="ignore")

    # One-hot encoding
    df = pd.get_dummies(df, drop_first=True)

    if target and target in df.columns:
        X = df.drop(columns=[target])
        y = df[target]
    else:
        X = df
        y = None

    # Standardization
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y
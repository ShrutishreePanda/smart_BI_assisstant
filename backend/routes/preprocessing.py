# backend/routes/preprocessing.py

import pandas as pd
from sklearn.preprocessing import StandardScaler


def preprocess(df, target=None):
    df = df.copy()

    # Drop ID-like columns
    df = df.drop(columns=[c for c in df.columns if "id" in c.lower()], errors="ignore")

    if target:
        if target not in df.columns:
            return None, None, None
        y = df[target]
        X = df.drop(columns=[target])
    else:
        X = df
        y = None

    # 🔹 One-hot encode ONLY features
    X = pd.get_dummies(X, drop_first=True)

    # 🔹 Standardization
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, X.columns
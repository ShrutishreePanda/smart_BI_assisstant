import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def preprocess(df, target=None, features=None):
    df = df.copy()

    df = df.drop(columns=[c for c in df.columns if "id" in c.lower()], errors="ignore")

    if features:
        missing_features = [column for column in features if column not in df.columns]
        if missing_features:
            raise ValueError(f"Feature column(s) not found: {missing_features}")

        if target and target in features:
            raise ValueError("Target column cannot also be used as an input feature")

        selected_columns = list(features)
        if target:
            selected_columns.append(target)
        df = df[selected_columns]

    if target:
        if target not in df.columns:
            raise ValueError(f"Target '{target}' not found")
        df = df.dropna(subset=[target])
        y = df[target]
        X = df.drop(columns=[target])
    else:
        X = df
        y = None

    if X.empty or X.shape[1] == 0:
        raise ValueError("Select at least one input feature")

    X = X.replace([np.inf, -np.inf], np.nan)

    numeric_cols = X.select_dtypes(include="number").columns
    categorical_cols = X.columns.difference(numeric_cols)

    if len(numeric_cols) > 0:
        X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].median())
        X[numeric_cols] = X[numeric_cols].fillna(0)

    if len(categorical_cols) > 0:
        X[categorical_cols] = X[categorical_cols].fillna("Missing")

    X = pd.get_dummies(X, drop_first=True)
    X = X.fillna(0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, X.columns

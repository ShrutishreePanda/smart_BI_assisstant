from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from backend.routes.preprocessing import preprocess


def train_transaction_prediction(df: pd.DataFrame, target: str, features: list[str]) -> dict[str, Any]:
    X, y, feature_names = preprocess(df, target, features)
    if y is None or y.dtype.kind not in "biufc":
        raise ValueError("Prediction target must be numeric")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    errors = y_test.to_numpy() - predictions

    return {
        "task": "regression",
        "model": "Linear Regression",
        "result": {
            "mae": float(mean_absolute_error(y_test, predictions)),
            "mse": float(mean_squared_error(y_test, predictions)),
            "optional_r2": float(r2_score(y_test, predictions)),
        },
        "visualizations": {
            "actual_vs_predicted": actual_vs_predicted_points(y_test, predictions),
            "error_distribution": error_points(errors),
        },
        "insights": [
            "We use historical and behavioral features to predict transaction values.",
            "Mean absolute error is the average prediction miss in transaction-amount units.",
            "Mean squared error gives larger misses more weight, so it helps reveal unstable predictions.",
        ],
        "features": list(feature_names),
        "use_case": "Predictive Modeling",
    }


def actual_vs_predicted_points(y_true: pd.Series, predictions) -> list[dict[str, float]]:
    points = []
    for actual, predicted in zip(y_true.head(80), predictions[:80]):
        points.append(
            {
                "Actual Transaction Amount": float(actual),
                "Predicted Transaction Amount": float(predicted),
            }
        )
    return points


def error_points(errors) -> list[dict[str, float]]:
    return [{"Prediction Error": float(error)} for error in errors[:200]]

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from backend.routes.preprocessing import preprocess


def train_fraud_detection(df: pd.DataFrame, target: str, features: list[str]) -> dict[str, Any]:
    X, y, feature_names = preprocess(df, target, features)
    if y is None or y.nunique(dropna=True) < 2:
        raise ValueError("Fraud target must contain at least two classes")

    stratify = y if y.nunique(dropna=True) == 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=stratify,
    )

    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    actual_counts = fraud_counts(y_test)
    predicted_counts = fraud_counts(pd.Series(predictions))

    return {
        "task": "classification",
        "model": "Logistic Regression",
        "result": {
            "accuracy": float(accuracy_score(y_test, predictions)),
            "precision": float(precision_score(y_test, predictions, zero_division=0)),
            "recall": float(recall_score(y_test, predictions, zero_division=0)),
        },
        "visualizations": {
            "fraud_distribution": actual_counts,
            "prediction_summary": predicted_counts,
        },
        "insights": [
            "We use behavioral and transactional features to detect anomalies indicative of fraud.",
            "Accuracy shows overall correctness, precision shows how reliable fraud alerts are, and recall shows how many fraud cases were caught.",
        ],
        "features": list(feature_names),
        "use_case": "Fraud Detection",
    }


def fraud_counts(values: pd.Series) -> dict[str, int]:
    counts = values.value_counts().to_dict()
    return {
        "Normal Transactions": int(counts.get(0, 0)),
        "Fraud Transactions": int(counts.get(1, 0)),
    }

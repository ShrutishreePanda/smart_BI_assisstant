from __future__ import annotations

from typing import Any, Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend.models.kmeans_model import train_behavioral_clustering
from backend.models.linear_model import train_transaction_prediction
from backend.models.logistic_model import train_fraud_detection
from backend.state import get_df


router = APIRouter(prefix="/ml", tags=["ML"])


RequestedModel = Literal["Auto", "Logistic Regression", "Linear Regression", "K-Means"]
ModelName = Literal["Logistic Regression", "Linear Regression", "K-Means"]
TaskName = Literal["classification", "regression", "clustering"]


class TrainRequest(BaseModel):
    target: str | None = None
    features: list[str] | None = None
    model: RequestedModel = "Auto"
    k: int = 3


class TrainResponse(BaseModel):
    task: TaskName
    model: ModelName
    result: dict[str, Any]
    visualizations: dict[str, Any]
    insights: list[str]
    features: list[str]
    use_case: str


FRAUD_TARGET = "IsFraud"
PREDICTION_TARGET = "TransactionAmount"

FRAUD_FEATURES = [
    "TransactionAmount",
    "TransactionHour",
    "TransactionDayOfWeek",
    "IsWeekend",
    "TransactionDistanceKm",
    "CustomerAge",
    "CityPopulation",
    "TransactionCategoryEncoded",
]

PREDICTION_FEATURES = [
    "TransactionHour",
    "TransactionDayOfWeek",
    "IsWeekend",
    "CustomerAge",
    "CityPopulation",
    "TransactionCategoryEncoded",
]

CLUSTERING_FEATURES = [
    "TransactionAmount",
    "TransactionHour",
    "TransactionDistanceKm",
    "CustomerAge",
    "CityPopulation",
    "TransactionCategoryEncoded",
]


def available_columns(df, requested_columns: list[str]) -> list[str]:
    return [column for column in requested_columns if column in df.columns]


def resolve_request(df, req: TrainRequest) -> tuple[str | None, str, list[str], str]:
    if req.model == "K-Means" or (not req.target and not req.features):
        return (
            None,
            "K-Means",
            req.features or available_columns(df, CLUSTERING_FEATURES),
            "Behavioral Analysis",
        )

    if req.target == FRAUD_TARGET or req.model == "Logistic Regression":
        return (
            req.target or FRAUD_TARGET,
            "Logistic Regression",
            req.features or available_columns(df, FRAUD_FEATURES),
            "Fraud Detection",
        )

    if req.target == PREDICTION_TARGET or req.model == "Linear Regression":
        return (
            req.target or PREDICTION_TARGET,
            "Linear Regression",
            req.features or available_columns(df, PREDICTION_FEATURES),
            "Predictive Modeling",
        )

    if req.model == "Auto":
        return (
            FRAUD_TARGET,
            "Logistic Regression",
            req.features or available_columns(df, FRAUD_FEATURES),
            "Fraud Detection",
        )

    raise ValueError("Unsupported model and target combination")


def validate_request(df, target: str | None, model: str, features: list[str], k: int) -> None:
    if k < 2:
        raise ValueError("K-Means requires at least 2 clusters")
    if target and target not in df.columns:
        raise ValueError(f"Target '{target}' not found")
    if target and target in features:
        raise ValueError("Target column cannot also be used as an input feature")
    if not features:
        raise ValueError("No input features are available for this model")
    missing_features = [column for column in features if column not in df.columns]
    if missing_features:
        raise ValueError(f"Feature column(s) not found: {missing_features}")
    if model == "Logistic Regression" and target != FRAUD_TARGET:
        raise ValueError("Logistic Regression is mapped to Fraud Detection with target IsFraud")
    if model == "Linear Regression" and target != PREDICTION_TARGET:
        raise ValueError("Linear Regression is mapped to Predictive Modeling with target TransactionAmount")


@router.post("/train", response_model=TrainResponse)
def train(req: TrainRequest) -> dict[str, Any]:
    try:
        df = get_df()
        target, model, features, _ = resolve_request(df, req)
        validate_request(df, target, model, features, req.k)

        if model == "Logistic Regression":
            return train_fraud_detection(df, target or FRAUD_TARGET, features)
        if model == "Linear Regression":
            return train_transaction_prediction(df, target or PREDICTION_TARGET, features)
        return train_behavioral_clustering(df, features, req.k)

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

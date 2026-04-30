from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from state import get_df


router = APIRouter(prefix="/ml", tags=["Machine Learning"])


# These are the only ML models planned for this MVP:
# - Logistic Regression: classification, used for binary/categorical targets.
# - Linear Regression: regression, used for continuous numeric targets.
# - K-Means: clustering, used when no target column is selected.
ModelName = Literal["Logistic Regression", "Linear Regression", "K-Means"]
TaskName = Literal["classification", "regression", "clustering"]


class UploadResponse(BaseModel):
    columns: list[str]
    shape: list[int]
    dtypes: dict[str, str]
    preview: list[dict[str, Any]]


class EDAResponse(BaseModel):
    shape: list[int]
    columns: list[str]
    dtypes: dict[str, str]
    null_counts: dict[str, int]
    null_pct: dict[str, float]
    duplicate_rows: int
    cardinality: dict[str, int]
    summary: dict[str, dict[str, Any]]
    value_counts: dict[str, dict[str, int]]


class CleanResponse(BaseModel):
    rows_before: int
    rows_after: int
    rows_removed: int
    nulls_before: int
    nulls_after: int


class TrainRequest(BaseModel):
    target: str | None = Field(
        default=None,
        description="Selected target column. Leave null to run K-Means clustering.",
    )
    k: int = Field(
        default=3,
        ge=2,
        le=6,
        description="Number of clusters for K-Means when target is null.",
    )


class SupervisedResult(BaseModel):
    task: Literal["classification", "regression"]
    model: Literal["Logistic Regression", "Linear Regression"]
    target: str
    metrics: dict[str, float]
    coefficients: dict[str, float]


class ClusteringResult(BaseModel):
    task: Literal["clustering"]
    model: Literal["K-Means"]
    k: int
    inertia: float
    silhouette: float
    cluster_sizes: dict[str, int]
    cluster_summary: dict[str, dict[str, float]]


class TrainResponse(BaseModel):
    task: TaskName
    model: ModelName
    result: dict[str, Any]
    insights: list[str]


def _load_dataset() -> pd.DataFrame:
    try:
        return get_df()
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


def _feature_columns(df: pd.DataFrame, target: str | None = None) -> list[str]:
    excluded = {target} if target else set()
    return [column for column in df.columns if column not in excluded]


def _build_preprocessor(df: pd.DataFrame, feature_columns: list[str]) -> ColumnTransformer:
    numeric_columns = [
        column for column in feature_columns if pd.api.types.is_numeric_dtype(df[column])
    ]
    categorical_columns = [
        column for column in feature_columns if column not in numeric_columns
    ]

    transformers = []
    if numeric_columns:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_columns,
            )
        )
    if categorical_columns:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "encoder",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                categorical_columns,
            )
        )

    if not transformers:
        raise HTTPException(
            status_code=400,
            detail="Dataset does not contain usable feature columns.",
        )

    return ColumnTransformer(transformers=transformers)


def _is_classification_target(series: pd.Series) -> bool:
    unique_count = series.nunique(dropna=True)
    if not pd.api.types.is_numeric_dtype(series):
        return True

    return unique_count <= min(20, max(2, int(len(series) * 0.1)))


def _coefficient_map(model: Pipeline, feature_columns: list[str]) -> dict[str, float]:
    estimator = model.named_steps["model"]
    preprocessor = model.named_steps["preprocess"]

    try:
        names = preprocessor.get_feature_names_out(feature_columns)
    except ValueError:
        names = np.array(feature_columns)

    coefficients = getattr(estimator, "coef_", np.array([]))
    if coefficients.ndim == 2:
        coefficients = np.mean(np.abs(coefficients), axis=0)

    return {
        str(name): round(float(value), 4)
        for name, value in zip(names, np.ravel(coefficients), strict=False)
    }


def _round_metric(value: float) -> float:
    if not np.isfinite(value):
        return 0.0

    return round(float(value), 4)


def _train_classification(
    df: pd.DataFrame, target: str, feature_columns: list[str]
) -> TrainResponse:
    prepared = df[feature_columns + [target]].dropna(subset=[target])
    if len(prepared) < 4 or prepared[target].nunique() < 2:
        raise HTTPException(
            status_code=400,
            detail="Classification needs at least two target classes and four rows.",
        )

    x = prepared[feature_columns]
    y = prepared[target]
    class_counts = y.value_counts()
    test_count = max(1, int(np.ceil(len(prepared) * 0.25)))
    can_stratify = (
        class_counts.min() >= 2
        and test_count >= len(class_counts)
        and len(prepared) - test_count >= len(class_counts)
    )
    stratify = y if can_stratify else None

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=test_count,
        random_state=42,
        stratify=stratify,
    )

    model = Pipeline(
        steps=[
            ("preprocess", _build_preprocessor(prepared, feature_columns)),
            ("model", LogisticRegression(max_iter=1000)),
        ]
    )
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)

    result = SupervisedResult(
        task="classification",
        model="Logistic Regression",
        target=target,
        metrics={
            "accuracy": _round_metric(accuracy_score(y_test, predictions)),
            "f1": _round_metric(
                f1_score(y_test, predictions, average="weighted", zero_division=0)
            ),
        },
        coefficients=_coefficient_map(model, feature_columns),
    )

    return TrainResponse(
        task="classification",
        model="Logistic Regression",
        result=result.model_dump(),
        insights=[
            f"Trained Logistic Regression to predict {target}.",
            f"Model evaluated on {len(y_test)} holdout rows.",
        ],
    )


def _train_regression(
    df: pd.DataFrame, target: str, feature_columns: list[str]
) -> TrainResponse:
    prepared = df[feature_columns + [target]].dropna(subset=[target])
    if len(prepared) < 4:
        raise HTTPException(
            status_code=400,
            detail="Regression needs at least four rows with a target value.",
        )

    x = prepared[feature_columns]
    y = prepared[target]
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.25,
        random_state=42,
    )

    model = Pipeline(
        steps=[
            ("preprocess", _build_preprocessor(prepared, feature_columns)),
            ("model", LinearRegression()),
        ]
    )
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    r2 = r2_score(y_test, predictions) if len(y_test) > 1 else 0.0

    result = SupervisedResult(
        task="regression",
        model="Linear Regression",
        target=target,
        metrics={
            "r2": _round_metric(r2),
            "mae": _round_metric(mean_absolute_error(y_test, predictions)),
            "rmse": _round_metric(np.sqrt(mean_squared_error(y_test, predictions))),
        },
        coefficients=_coefficient_map(model, feature_columns),
    )

    return TrainResponse(
        task="regression",
        model="Linear Regression",
        result=result.model_dump(),
        insights=[
            f"Trained Linear Regression to predict {target}.",
            f"Model evaluated on {len(y_test)} holdout rows.",
        ],
    )


def _train_clustering(df: pd.DataFrame, k: int, feature_columns: list[str]) -> TrainResponse:
    prepared = df[feature_columns].dropna(how="all")
    if len(prepared) < k:
        raise HTTPException(
            status_code=400,
            detail=f"K-Means needs at least {k} usable rows.",
        )

    preprocessor = _build_preprocessor(prepared, feature_columns)
    x = preprocessor.fit_transform(prepared)
    model = KMeans(n_clusters=k, random_state=42, n_init="auto")
    labels = model.fit_predict(x)

    cluster_sizes = pd.Series(labels).value_counts().sort_index()
    numeric_summary = prepared.select_dtypes(include="number").copy()
    numeric_summary["cluster"] = labels
    cluster_summary = (
        numeric_summary.groupby("cluster").mean(numeric_only=True).round(4).to_dict()
        if numeric_summary.shape[1] > 1
        else {}
    )
    cluster_summary = {
        str(column): {str(cluster): float(value) for cluster, value in values.items()}
        for column, values in cluster_summary.items()
    }
    label_count = len(set(labels))
    silhouette = (
        silhouette_score(x, labels) if 1 < label_count < len(prepared) else 0.0
    )

    result = ClusteringResult(
        task="clustering",
        model="K-Means",
        k=k,
        inertia=_round_metric(model.inertia_),
        silhouette=_round_metric(silhouette),
        cluster_sizes={
            str(cluster): int(size) for cluster, size in cluster_sizes.to_dict().items()
        },
        cluster_summary=cluster_summary,
    )

    return TrainResponse(
        task="clustering",
        model="K-Means",
        result=result.model_dump(),
        insights=[
            f"Grouped the dataset into {k} clusters.",
            "Cluster summaries use numeric feature averages where available.",
        ],
    )


@router.post("/train", response_model=TrainResponse)
def train_model(request: TrainRequest) -> TrainResponse:
    """Train an ML model using the dataset stored in state.py."""
    df = _load_dataset()
    if df.empty:
        raise HTTPException(status_code=400, detail="Uploaded dataset is empty.")

    if request.target is not None and request.target not in df.columns:
        raise HTTPException(
            status_code=400,
            detail=f"Target column '{request.target}' does not exist in the dataset.",
        )

    feature_columns = _feature_columns(df, request.target)
    if request.target is None:
        return _train_clustering(df, request.k, feature_columns)

    if _is_classification_target(df[request.target]):
        return _train_classification(df, request.target, feature_columns)

    return _train_regression(df, request.target, feature_columns)

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


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

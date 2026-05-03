from __future__ import annotations

from typing import Any, Literal
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend.state import get_df
from backend.routes.preprocessing import preprocess

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, r2_score, silhouette_score


router = APIRouter(prefix="/ml", tags=["ML"])


# ------------------ SCHEMAS ------------------

ModelName = Literal["Logistic Regression", "Linear Regression", "K-Means"]
TaskName = Literal["classification", "regression", "clustering"]


class TrainRequest(BaseModel):
    target: str | None = None
    k: int = 3


class TrainResponse(BaseModel):
    task: TaskName
    model: ModelName
    result: dict[str, Any]
    insights: list[str]


# ------------------ ROUTE ------------------

@router.post("/train", response_model=TrainResponse)
def train(req: TrainRequest):
    try:
        df = get_df()

        # -------- SUPERVISED --------
        if req.target:

            # 🔥 Validate BEFORE preprocessing
            if req.target not in df.columns:
                raise ValueError(
                    f"Target '{req.target}' not found. Available: {list(df.columns)}"
                )

            X, y, feature_names = preprocess(df, req.target)

            if y is None:
                raise ValueError("Target extraction failed")

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            # Classification
            if y.nunique() <= 10:
                model = LogisticRegression(max_iter=1000)
                model.fit(X_train, y_train)

                pred = model.predict(X_test)
                acc = accuracy_score(y_test, pred)

                return {
                    "task": "classification",
                    "model": "Logistic Regression",
                    "result": {"accuracy": float(acc)},
                    "insights": [
                        "Model trained successfully",
                        "Binary classification (fraud detection)"
                    ]
                }

            # Regression
            else:
                model = LinearRegression()
                model.fit(X_train, y_train)

                pred = model.predict(X_test)
                r2 = r2_score(y_test, pred)

                return {
                    "task": "regression",
                    "model": "Linear Regression",
                    "result": {"r2": float(r2)},
                    "insights": [
                        "Model trained successfully",
                        "Regression task"
                    ]
                }

        # -------- UNSUPERVISED --------
        else:
            X, _, _ = preprocess(df, None)

            model = KMeans(n_clusters=req.k, random_state=42)
            labels = model.fit_predict(X)

            silhouette = silhouette_score(X, labels)

            return {
                "task": "clustering",
                "model": "K-Means",
                "result": {
                    "clusters": req.k,
                    "silhouette": float(silhouette)
                },
                "insights": [
                    "Clustering completed",
                    "Customer segmentation performed"
                ]
            }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
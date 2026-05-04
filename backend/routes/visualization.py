from __future__ import annotations

from typing import Any

import pandas as pd
from fastapi import APIRouter, HTTPException

from backend.state import get_df


router = APIRouter(prefix="/visualize", tags=["Visualization"])


def counts_for(series: pd.Series, limit: int = 10) -> dict[str, int]:
    counts = series.value_counts(dropna=False).head(limit)
    return {str(key): int(value) for key, value in counts.items()}


def first_distribution_column(df: pd.DataFrame) -> str | None:
    for column in df.columns:
        if df[column].nunique(dropna=True) <= 20:
            return column
    return None


def first_time_series(df: pd.DataFrame) -> dict[str, Any]:
    numeric_columns = list(df.select_dtypes(include="number").columns)
    if not numeric_columns:
        return {}

    date_candidates = [
        column for column in df.columns if "date" in column.lower() or "time" in column.lower()
    ]
    for column in date_candidates:
        parsed = pd.to_datetime(df[column], errors="coerce")
        if parsed.notna().sum() < 2:
            continue

        chart_df = pd.DataFrame({"date": parsed, "value": df[numeric_columns[0]]}).dropna()
        if chart_df.empty:
            continue

        grouped = chart_df.groupby(chart_df["date"].dt.date)["value"].mean().head(50)
        return {
            "column": column,
            "value_column": numeric_columns[0],
            "values": {str(key): float(value) for key, value in grouped.items()},
        }

    return {}


@router.get("")
def visualize():
    try:
        df = get_df()
        numeric = df.select_dtypes(include="number").fillna(0)
        if numeric.empty:
            raise ValueError("No numeric columns available for visualization")

        distribution_column = first_distribution_column(df)
        distribution = {}
        if distribution_column:
            distribution = {
                "column": distribution_column,
                "values": counts_for(df[distribution_column]),
            }

        comparison_column = distribution_column or numeric.columns[0]
        heatmap = numeric.corr().round(3).fillna(0).to_dict()

        return {
            "distribution": distribution,
            "comparison": {
                "column": comparison_column,
                "values": counts_for(df[comparison_column]),
            },
            "time_series": first_time_series(df),
            "heatmap": heatmap,
            "bar": counts_for(df[comparison_column]),
            "line": {},
        }

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

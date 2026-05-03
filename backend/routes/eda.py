from __future__ import annotations

import pandas as pd
from fastapi import APIRouter, HTTPException
from state import get_df


def compute_eda(df: pd.DataFrame) -> dict:
    """
    Compute JSON-friendly exploratory data analysis results.

    This file stays pandas-only. FastAPI can call this function from a route,
    and Streamlit can render the returned dictionaries directly.
    """
    null_counts = df.isna().sum()
    null_pct = (null_counts / len(df) * 100).round(2)
    cardinality = df.nunique(dropna=True)
    summary = df.describe(include="all").where(pd.notnull(df.describe(include="all")), None)

    value_counts = {}
    for column in df.columns:
        if cardinality[column] <= 20:
            counts = df[column].value_counts(dropna=False).head(10)
            value_counts[column] = {
                str(key): int(value) for key, value in counts.to_dict().items()
            }

    return {
        "shape": [int(df.shape[0]), int(df.shape[1])],
        "columns": list(df.columns),
        "dtypes": {column: str(dtype) for column, dtype in df.dtypes.items()},
        "null_counts": {
            column: int(value) for column, value in null_counts.to_dict().items()
        },
        "null_pct": {
            column: float(value) for column, value in null_pct.to_dict().items()
        },
        "duplicate_rows": int(df.duplicated().sum()),
        "cardinality": {
            column: int(value) for column, value in cardinality.to_dict().items()
        },
        "summary": summary.to_dict(),
        "value_counts": value_counts,
    }


def correlation_matrix(df: pd.DataFrame) -> dict:
    """Return correlation data for numeric columns."""
    numeric_df = df.select_dtypes(include="number")
    if numeric_df.shape[1] < 2:
        return {"columns": [], "matrix": []}

    corr = numeric_df.corr().round(3)
    return {
        "columns": list(corr.columns),
        "matrix": corr.values.tolist(),
    }


eda_router = APIRouter(prefix="/eda", tags=["EDA"])

DATA_PATH = "data/UnifiedTransactionFeatures.csv"


@eda_router.get("/summary")
def eda_summary():
    """
    Return full EDA summary for the unified dataset.
    """
    try:
        df = get_df()
        return compute_eda(df)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Feature dataset not found")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@eda_router.get("/correlation")
def eda_correlation():
    """
    Return correlation matrix for numeric features.
    """
    try:
        df = get_df()
        return correlation_matrix(df)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Feature dataset not found")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
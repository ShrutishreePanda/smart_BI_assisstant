# backend/routes/clean.py

from __future__ import annotations

import pandas as pd
from fastapi import APIRouter, HTTPException

from backend.state import get_df, update_df


router = APIRouter(prefix="/clean", tags=["Cleaning"])


@router.post("/")
def clean_data():
    try:
        df = get_df()

        rows_before = df.shape[0]
        nulls_before = int(df.isna().sum().sum())

        # 🔹 Remove duplicate rows
        df = df.drop_duplicates()

        # 🔹 Clean each column
        for col in df.columns:
            # Convert numeric-like strings to numbers
            converted = pd.to_numeric(df[col], errors="coerce")
            
            if converted.notna().sum() > 0:
                df[col] = converted
                
            if df[col].dropna().empty:
                continue
            
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna(df[col].mode()[0])

        rows_after = df.shape[0]
        nulls_after = int(df.isna().sum().sum())

        # 🔹 Save cleaned dataset
        update_df(df)

        return {
            "rows_before": rows_before,
            "rows_after": rows_after,
            "rows_removed": rows_before - rows_after,
            "nulls_before": nulls_before,
            "nulls_after": nulls_after,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
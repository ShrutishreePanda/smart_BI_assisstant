from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.data_loader import load_file, validate_dataframe
from backend.routes.eda import compute_eda
from backend.state import save_df
from feature_engineering import generate_unified_features, save_unified_features


router = APIRouter(tags=["Upload"])

class UploadResponse(BaseModel):
    message: str
    rows: int
    columns: int
    column_names: list[str]
    dtypes: dict[str, str]
    preview: list[dict[str, Any]]
    raw_rows: int
    raw_columns: int
    generated_file: str
    feature_mappings: dict[str, str]
    raw_eda: dict[str, Any]


@router.post("/upload", response_model=UploadResponse)
def upload(file: UploadFile = File(...)) -> dict[str, Any]:
    try:
        raw_df = load_file(file)
        validate_dataframe(raw_df)
        raw_eda = compute_eda(raw_df)
        feature_df, feature_mappings = generate_unified_features(raw_df)
        output_path = save_unified_features(feature_df)
        save_df(feature_df)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "message": "Raw dataset uploaded, analyzed, and converted into unified model features",
        "rows": int(feature_df.shape[0]),
        "columns": int(feature_df.shape[1]),
        "column_names": list(feature_df.columns),
        "dtypes": {column: str(dtype) for column, dtype in feature_df.dtypes.items()},
        "preview": feature_df.head(5).where(feature_df.notnull(), None).to_dict(orient="records"),
        "raw_rows": int(raw_df.shape[0]),
        "raw_columns": int(raw_df.shape[1]),
        "generated_file": str(output_path),
        "feature_mappings": feature_mappings,
        "raw_eda": raw_eda,
    }



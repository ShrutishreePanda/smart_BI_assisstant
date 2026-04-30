from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from fastapi import APIRouter, FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.services.data_loader import load_file, validate_dataframe
from backend.state import save_df


router = APIRouter(tags=["Upload"])

class UploadResponse(BaseModel):
    message: str
    rows: int
    columns: int
    column_names: list[str]
    dtypes: dict[str, str]
    preview: list[dict[str, Any]]


@router.post("/upload", response_model=UploadResponse)
def upload(file: UploadFile = File(...)) -> dict[str, Any]:
    try:
        df = load_file(file)
        validate_dataframe(df)
        save_df(df)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "message": "Dataset uploaded successfully",
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "column_names": list(df.columns),
        "dtypes": {column: str(dtype) for column, dtype in df.dtypes.items()},
        "preview": df.head(5).where(df.notnull(), None).to_dict(orient="records"),
    }


app = FastAPI(title="Smart BI Assistant Upload API")
app.include_router(router)

from __future__ import annotations

import pandas as pd
from fastapi import UploadFile


def load_csv(file: UploadFile) -> pd.DataFrame:
    """
    Load CSV file from FastAPI UploadFile object.
    """
    try:
        return pd.read_csv(file.file)
    except Exception as e:
        raise ValueError(f"Error reading CSV file: {str(e)}") from e


def load_file(file: UploadFile) -> pd.DataFrame:
    """
    Load file based on extension. Supports CSV and Excel files.
    """
    filename = (file.filename or "").lower()

    try:
        if filename.endswith(".csv"):
            return pd.read_csv(file.file)

        if filename.endswith((".xlsx", ".xls")):
            return pd.read_excel(file.file)

        raise ValueError("Unsupported file format. Please upload CSV or Excel.")
    except ValueError:
        raise
    except Exception as e:
        raise ValueError(f"File loading failed: {str(e)}") from e


def validate_dataframe(df: pd.DataFrame) -> bool:
    """
    Run basic validation checks for uploaded datasets.
    """
    if df is None or df.empty:
        raise ValueError("Uploaded file is empty")

    if df.shape[1] == 0:
        raise ValueError("No columns found in dataset")

    return True

from __future__ import annotations

from pathlib import Path
from typing import BinaryIO

import pandas as pd


def load_file(file: BinaryIO) -> pd.DataFrame:
    """Load an uploaded CSV file into a DataFrame."""
    filename = Path(getattr(file, "filename", "") or getattr(file, "name", "")).name
    if not filename.lower().endswith(".csv"):
        raise ValueError("Only CSV files are supported")

    try:
        return pd.read_csv(file.file if hasattr(file, "file") else file)
    except Exception as exc:
        raise ValueError(f"Could not read CSV file: {exc}") from exc


def validate_dataframe(df: pd.DataFrame) -> None:
    """Validate uploaded data before saving it in backend state."""
    if df is None or df.empty:
        raise ValueError("Uploaded file is empty")
    if len(df.columns) == 0:
        raise ValueError("Uploaded file has no columns")

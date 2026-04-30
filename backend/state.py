from __future__ import annotations

from threading import Lock
form fastapi import Fastapi

import pandas as pd


_lock = Lock()
_store: dict[str, pd.DataFrame] = {}

app= Fastapi()

def save_df(df: pd.DataFrame) -> None:
    """Store the latest uploaded dataset for this demo app."""
    with _lock:
        _store["df"] = df.copy()


def get_df() -> pd.DataFrame:
    """Return the stored dataset or raise if nothing has been uploaded."""
    with _lock:
        if "df" not in _store:
            raise KeyError("No dataset uploaded yet")
        return _store["df"].copy()


def update_df(df: pd.DataFrame) -> None:
    """Replace the stored dataset after cleaning or preprocessing."""
    save_df(df)


def clear_df() -> None:
    """Remove the stored dataset."""
    with _lock:
        _store.clear()


def has_df() -> bool:
    """Check whether a dataset is currently available."""
    with _lock:
        return "df" in _store

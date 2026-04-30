# backend/state.py

from __future__ import annotations
from threading import Lock
import pandas as pd


# Thread lock for safe concurrent access
_lock = Lock()

# In-memory store
_store: dict[str, pd.DataFrame] = {}


def save_df(df: pd.DataFrame) -> None:
    """
    Save uploaded dataset.
    Overwrites existing dataset (single-user demo design).
    """
    if df is None or df.empty:
        raise ValueError("Cannot save empty DataFrame")

    with _lock:
        _store["df"] = df.copy()


def get_df() -> pd.DataFrame:
    """
    Retrieve stored dataset.
    Raises error if not available.
    """
    with _lock:
        if "df" not in _store:
            raise KeyError("No dataset uploaded yet")

        return _store["df"].copy()


def update_df(df: pd.DataFrame) -> None:
    """
    Replace dataset after cleaning / preprocessing.
    """
    save_df(df)


def clear_df() -> None:
    """
    Clear stored dataset (useful for testing/debugging).
    """
    with _lock:
        _store.clear()


def has_df() -> bool:
    """
    Check if dataset exists.
    """
    with _lock:
        return "df" in _store
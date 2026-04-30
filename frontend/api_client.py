"""
api_client.py — Smart BI Assistant
====================================
Handles ALL backend API communication.
No Streamlit imports here. Pure requests logic only.

Separation of concerns:
  - app.py    → UI rendering, user interaction, chart display
  - api_client.py → HTTP calls, response parsing, error handling
"""

import requests
from typing import Optional

# ─────────────────────────────────────────
# 🔧 Configuration — change base URL here
# ─────────────────────────────────────────
BASE_URL = "http://localhost:8000"   # ← your backend base URL

UPLOAD_ENDPOINT      = f"{BASE_URL}/upload"
COLUMNS_ENDPOINT     = f"{BASE_URL}/columns"
VISUALIZE_ENDPOINT   = f"{BASE_URL}/visualize"

# Default timeout for all requests (seconds)
REQUEST_TIMEOUT = 30


# ─────────────────────────────────────────────────────────────────────────────
# 📤 upload_file
# Sends the uploaded CSV file to the backend.
# Returns a dict with keys:
#   {"success": bool, "message": str, "file_id": str | None}
# ─────────────────────────────────────────────────────────────────────────────
def upload_file(file) -> dict:
    """
    Upload a CSV file to the backend.

    Args:
        file: Streamlit UploadedFile object (file-like, with .name and .read())

    Returns:
        dict with:
          - success (bool)
          - message (str) — human-readable status or error
          - file_id (str | None) — backend identifier for the uploaded file
    """
    try:
        # Streamlit UploadedFile is file-like; send as multipart form-data
        files = {"file": (file.name, file.read(), "text/csv")}
        response = requests.post(UPLOAD_ENDPOINT, files=files, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()

        data = response.json()
        return {
            "success": True,
            "message": data.get("message", "File uploaded successfully."),
            "file_id": data.get("file_id"),   # backend may return an ID for the session
        }

    except requests.exceptions.ConnectionError:
        return {
            "success": False,
            "message": "Cannot connect to the backend. Is the server running?",
            "file_id": None,
        }
    except requests.exceptions.Timeout:
        return {
            "success": False,
            "message": "Request timed out. The server took too long to respond.",
            "file_id": None,
        }
    except requests.exceptions.HTTPError as e:
        return {
            "success": False,
            "message": f"Server error ({e.response.status_code}): {e.response.text}",
            "file_id": None,
        }
    except Exception as e:
        return {
            "success": False,
            "message": f"Unexpected error: {str(e)}",
            "file_id": None,
        }


# ─────────────────────────────────────────────────────────────────────────────
# 📋 get_columns
# Fetches column names for the currently uploaded file from the backend.
# Returns a dict with keys:
#   {"success": bool, "columns": list[str], "message": str}
# ─────────────────────────────────────────────────────────────────────────────
def get_columns(file_id: Optional[str] = None) -> dict:
    """
    Retrieve column names from the backend for the uploaded dataset.

    Args:
        file_id (str | None): Optional file identifier returned by upload_file().
                              Pass None if the backend uses session-based state.

    Returns:
        dict with:
          - success (bool)
          - columns (list[str]) — list of column names; empty on failure
          - message (str) — status or error description
    """
    try:
        params = {}
        if file_id:
            params["file_id"] = file_id

        response = requests.get(COLUMNS_ENDPOINT, params=params, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()

        data = response.json()
        columns = data.get("columns", [])

        if not columns:
            return {
                "success": False,
                "columns": [],
                "message": "No columns returned by the backend.",
            }

        return {
            "success": True,
            "columns": columns,
            "message": f"{len(columns)} column(s) loaded.",
        }

    except requests.exceptions.ConnectionError:
        return {
            "success": False,
            "columns": [],
            "message": "Cannot connect to the backend.",
        }
    except requests.exceptions.Timeout:
        return {
            "success": False,
            "columns": [],
            "message": "Request timed out while fetching columns.",
        }
    except requests.exceptions.HTTPError as e:
        return {
            "success": False,
            "columns": [],
            "message": f"Server error ({e.response.status_code}): {e.response.text}",
        }
    except Exception as e:
        return {
            "success": False,
            "columns": [],
            "message": f"Unexpected error: {str(e)}",
        }


# ─────────────────────────────────────────────────────────────────────────────
# 📊 get_visualizations
# Fetches chart data for a given target column from the backend.
# Returns a dict with keys:
#   {"success": bool, "charts": list[dict], "message": str}
#
# Each chart dict is expected to have:
#   {
#     "type":   "bar" | "pie" | "line",
#     "title":  str,
#     "labels": list,         # x-axis / pie slice labels
#     "values": list[float],  # y-axis / pie slice sizes
#   }
# ─────────────────────────────────────────────────────────────────────────────
def get_visualizations(target_column: str, file_id: Optional[str] = None) -> dict:
    """
    Fetch visualization data for the selected target column.

    Args:
        target_column (str): The column the user selected as target.
        file_id (str | None): Optional file identifier from upload step.

    Returns:
        dict with:
          - success (bool)
          - charts (list[dict]) — see schema above; empty on failure
          - message (str) — status or error description
    """
    if not target_column:
        return {
            "success": False,
            "charts": [],
            "message": "No target column specified.",
        }

    try:
        payload = {"target_column": target_column}
        if file_id:
            payload["file_id"] = file_id

        response = requests.post(VISUALIZE_ENDPOINT, json=payload, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()

        data = response.json()
        charts = data.get("charts", [])

        return {
            "success": True,
            "charts": charts,
            "message": f"{len(charts)} chart(s) received.",
        }

    except requests.exceptions.ConnectionError:
        return {
            "success": False,
            "charts": [],
            "message": "Cannot connect to the backend.",
        }
    except requests.exceptions.Timeout:
        return {
            "success": False,
            "charts": [],
            "message": "Request timed out while fetching visualizations.",
        }
    except requests.exceptions.HTTPError as e:
        return {
            "success": False,
            "charts": [],
            "message": f"Server error ({e.response.status_code}): {e.response.text}",
        }
    except Exception as e:
        return {
            "success": False,
            "charts": [],
            "message": f"Unexpected error: {str(e)}",
        }
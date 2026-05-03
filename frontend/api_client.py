from __future__ import annotations

from typing import Optional

import requests


BASE_URL = "http://localhost:8000"
REQUEST_TIMEOUT = 30


def _get_json(url: str, **kwargs) -> dict:
    try:
        response = requests.get(url, timeout=REQUEST_TIMEOUT, **kwargs)
        response.raise_for_status()
        return {"success": True, "data": response.json()}
    except requests.exceptions.ConnectionError:
        return {"success": False, "message": "Cannot connect to the backend. Is the server running?"}
    except requests.exceptions.Timeout:
        return {"success": False, "message": "Request timed out. The server took too long to respond."}
    except requests.exceptions.HTTPError as exc:
        return {"success": False, "message": _format_http_error(exc)}
    except Exception as exc:
        return {"success": False, "message": f"Unexpected error: {exc}"}


def _format_http_error(exc: requests.exceptions.HTTPError) -> str:
    response = exc.response
    if response is None:
        return str(exc)

    try:
        detail = response.json().get("detail", response.text)
    except ValueError:
        detail = response.text

    return f"Server error ({response.status_code}): {detail}"


def upload_file(file) -> dict:
    try:
        file.seek(0)
        files = {"file": (file.name, file.read(), "text/csv")}
        response = requests.post(f"{BASE_URL}/upload", files=files, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()

        data = response.json()
        return {
            "success": True,
            "message": data.get("message", "Dataset uploaded successfully."),
            "file_id": data.get("file_id"),
            "rows": data.get("rows", 0),
            "columns_count": data.get("columns", 0),
            "columns": data.get("column_names", []),
            "dtypes": data.get("dtypes", {}),
            "preview": data.get("preview", []),
        }
    except requests.exceptions.ConnectionError:
        return {"success": False, "message": "Cannot connect to the backend. Is the server running?"}
    except requests.exceptions.Timeout:
        return {"success": False, "message": "Request timed out. The server took too long to respond."}
    except requests.exceptions.HTTPError as exc:
        return {"success": False, "message": _format_http_error(exc)}
    except Exception as exc:
        return {"success": False, "message": f"Unexpected error: {exc}"}


def get_columns(file_id: Optional[str] = None) -> dict:
    params = {"file_id": file_id} if file_id else None
    result = _get_json(f"{BASE_URL}/eda/summary", params=params)
    if not result["success"]:
        return {"success": False, "columns": [], "message": result["message"]}

    columns = result["data"].get("columns", [])
    return {
        "success": bool(columns),
        "columns": columns,
        "message": f"{len(columns)} column(s) loaded." if columns else "No columns returned by the backend.",
    }


def get_eda_summary() -> dict:
    return _get_json(f"{BASE_URL}/eda/summary")


def get_eda_correlation() -> dict:
    return _get_json(f"{BASE_URL}/eda/correlation")


def get_visualizations(target_column: str, file_id: Optional[str] = None) -> dict:
    if not target_column:
        return {"success": False, "charts": [], "message": "No target column specified."}

    try:
        payload = {"target_column": target_column}
        if file_id:
            payload["file_id"] = file_id

        response = requests.post(f"{BASE_URL}/visualize", json=payload, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()

        data = response.json()
        charts = data.get("charts", [])
        return {"success": True, "charts": charts, "message": f"{len(charts)} chart(s) received."}
    except requests.exceptions.ConnectionError:
        return {"success": False, "charts": [], "message": "Cannot connect to the backend."}
    except requests.exceptions.Timeout:
        return {"success": False, "charts": [], "message": "Request timed out while fetching visualizations."}
    except requests.exceptions.HTTPError as exc:
        return {"success": False, "charts": [], "message": _format_http_error(exc)}
    except Exception as exc:
        return {"success": False, "charts": [], "message": f"Unexpected error: {exc}"}


def run_ml(target: str | None = None, k: int = 3) -> dict:
    try:
        payload = {"target": target, "k": k}
        response = requests.post(
            f"{BASE_URL}/ml/train",
            json=payload,
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        return {"success": True, "data": response.json()}
    except requests.exceptions.ConnectionError:
        return {"success": False, "message": "Cannot connect to the backend."}
    except requests.exceptions.Timeout:
        return {"success": False, "message": "Model training timed out."}
    except requests.exceptions.HTTPError as exc:
        return {"success": False, "message": _format_http_error(exc)}
    except Exception as exc:
        return {"success": False, "message": str(exc)}


def get_visualization() -> dict:
    return _get_json(f"{BASE_URL}/visualize/")

# exports/exporter.py
# Export analysis results as CSV/JSON for Streamlit downloads.
# Updated: safer serialization + flatten helpers for nested exports (scenarios, components, etc.)

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Dict, List

import pandas as pd


def _json_safe(obj: Any) -> Any:
    """
    Convert non-JSON-serializable objects into safe representations.
    Handles: pandas/numpy types, datetimes, sets, bytes, and fallback to str().
    """
    try:
        # Primitive fast-path
        if obj is None or isinstance(obj, (bool, int, float, str)):
            return obj

        # Datetime-like
        if isinstance(obj, (datetime,)):
            return obj.isoformat()

        # Pandas / numpy scalars
        if hasattr(obj, "item") and callable(getattr(obj, "item")):
            try:
                return obj.item()
            except Exception:
                pass

        # Dict
        if isinstance(obj, dict):
            return {str(k): _json_safe(v) for k, v in obj.items()}

        # List / tuple
        if isinstance(obj, (list, tuple)):
            return [_json_safe(x) for x in obj]

        # Set
        if isinstance(obj, set):
            return [_json_safe(x) for x in sorted(list(obj), key=lambda x: str(x))]

        # Bytes
        if isinstance(obj, (bytes, bytearray)):
            return obj.decode("utf-8", errors="ignore")

        # Pandas Timestamp
        if hasattr(obj, "to_pydatetime"):
            try:
                return obj.to_pydatetime().isoformat()
            except Exception:
                pass

        # Fallback string
        return str(obj)
    except Exception:
        return str(obj)


def trends_timeseries_to_csv_bytes(timeseries: List[Dict[str, Any]]) -> bytes:
    """
    Input timeseries format: [{"date": "...", "value": ...}, ...]
    """
    df = pd.DataFrame(timeseries or [])
    if df.empty:
        df = pd.DataFrame(columns=["date", "value"])
    # Normalize types a bit
    if "date" in df.columns:
        df["date"] = df["date"].astype(str)
    if "value" in df.columns:
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df.to_csv(index=False).encode("utf-8")


def headlines_to_csv_bytes(headlines: List[Dict[str, Any]]) -> bytes:
    """
    Input headline format: [{"title": "...", "date": "...", "source": "...", ...}, ...]
    """
    df = pd.DataFrame(headlines or [])
    if df.empty:
        df = pd.DataFrame(columns=["title", "date", "source", "url", "sentiment_0_100"])

    # Ensure consistent columns if present
    for col in ["title", "date", "source", "url"]:
        if col in df.columns:
            df[col] = df[col].astype(str)

    if "sentiment_0_100" in df.columns:
        df["sentiment_0_100"] = pd.to_numeric(df["sentiment_0_100"], errors="coerce")

    return df.to_csv(index=False).encode("utf-8")


def flatten_for_csv(data: Dict[str, Any], *, prefix: str = "") -> Dict[str, Any]:
    """
    Flatten nested dicts into a single-level dict for quick CSV-friendly exports.
    Lists/dicts that are too nested are stringified safely.

    Example:
      {"forecast":{"direction":"Up","scenarios":{"best":{...}}}}
      -> {"forecast.direction":"Up", "forecast.scenarios":"{...json...}"}
    """
    out: Dict[str, Any] = {}
    for k, v in (data or {}).items():
        key = f"{prefix}{k}" if not prefix else f"{prefix}.{k}"
        if isinstance(v, dict):
            # recurse but avoid exploding extremely deep trees by stringifying if needed
            if len(v) <= 25:
                out.update(flatten_for_csv(v, prefix=key))
            else:
                out[key] = json.dumps(_json_safe(v), ensure_ascii=False)
        elif isinstance(v, list):
            out[key] = json.dumps(_json_safe(v), ensure_ascii=False)
        else:
            out[key] = _json_safe(v)
    return out


def analysis_to_json_bytes(analysis: Dict[str, Any]) -> bytes:
    """
    Full analysis export for reproducibility.
    Uses JSON-safe conversion to avoid crashing on numpy/pandas/datetime objects.
    """
    safe = _json_safe(analysis or {})
    return json.dumps(safe, indent=2, ensure_ascii=False).encode("utf-8")

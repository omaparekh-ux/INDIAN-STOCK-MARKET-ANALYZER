# utils/vis.py
# Chart/visual helpers for Streamlit UI (presentation-only).
# Purpose: keep app.py clean by centralizing common dataframe prep for charts.
# NOTE: No business logic; only transformations for plotting.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pandas as pd


# -----------------------------
# Trend timeseries helpers
# -----------------------------
def prep_trend_df(ts: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Convert trends timeseries (list[{'date':..., 'value':...}]) into a clean DataFrame.
    Ensures: date parsed, sorted, numeric value.
    Returns columns: ['date', 'value']
    """
    if not ts:
        return pd.DataFrame(columns=["date", "value"])

    df = pd.DataFrame(ts).copy()
    if "date" not in df.columns or "value" not in df.columns:
        return pd.DataFrame(columns=["date", "value"])

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    df = df.dropna(subset=["date", "value"]).sort_values("date")
    return df[["date", "value"]].reset_index(drop=True)


def add_moving_averages(
    df: pd.DataFrame,
    windows: List[int] | None = None,
    value_col: str = "value",
) -> pd.DataFrame:
    """
    Adds moving average columns (MA_{window}) for cleaner trend visualization.
    """
    windows = windows or [7, 28]
    if df is None or df.empty or value_col not in df.columns:
        return (df.copy() if df is not None else pd.DataFrame())

    out = df.copy()
    for w in windows:
        col = f"MA_{w}"
        out[col] = out[value_col].rolling(window=w, min_periods=max(2, w // 4)).mean()
    return out


def trend_key_points(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Returns simple 'key points' summary for UI: latest, min, max.
    """
    if df is None or df.empty:
        return {
            "latest_date": None,
            "latest_value": None,
            "min_date": None,
            "min_value": None,
            "max_date": None,
            "max_value": None,
        }

    s = df.dropna(subset=["date", "value"]).copy()
    if s.empty:
        return {
            "latest_date": None,
            "latest_value": None,
            "min_date": None,
            "min_value": None,
            "max_date": None,
            "max_value": None,
        }

    latest = s.iloc[-1]
    min_row = s.loc[s["value"].idxmin()]
    max_row = s.loc[s["value"].idxmax()]

    return {
        "latest_date": latest["date"],
        "latest_value": float(latest["value"]),
        "min_date": min_row["date"],
        "min_value": float(min_row["value"]),
        "max_date": max_row["date"],
        "max_value": float(max_row["value"]),
    }


# -----------------------------
# Compare mode helpers
# -----------------------------
def merge_compare_series(series: Dict[str, List[Dict[str, Any]]]) -> pd.DataFrame:
    """
    Given: series = { 'kw1': ts1, 'kw2': ts2, ... }
    Returns a merged DF with date column + one column per keyword.
    """
    if not series:
        return pd.DataFrame()

    dfs: List[pd.DataFrame] = []
    for kw, ts in series.items():
        df = prep_trend_df(ts)
        if df.empty:
            continue
        df = df.rename(columns={"value": kw})
        dfs.append(df[["date", kw]])

    if not dfs:
        return pd.DataFrame()

    merged = dfs[0]
    for d in dfs[1:]:
        merged = merged.merge(d, on="date", how="inner")

    merged = merged.sort_values("date").reset_index(drop=True)
    return merged


# -----------------------------
# Forecast helpers
# -----------------------------
def prep_forecast_smoothed_df(forecast: Any) -> pd.DataFrame:
    """
    forecast.smoothed -> DataFrame(date, value) for charting.
    """
    sm = getattr(forecast, "smoothed", None)
    if not sm:
        return pd.DataFrame(columns=["date", "value"])

    df = pd.DataFrame(sm).copy()
    if "date" not in df.columns or "value" not in df.columns:
        return pd.DataFrame(columns=["date", "value"])

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["date", "value"]).sort_values("date")
    return df[["date", "value"]].reset_index(drop=True)


def prep_scenarios_df(forecast: Any) -> pd.DataFrame:
    """
    Converts forecast.scenarios (best/base/worst) into a simple DF for charts/tables.
    Expected scenario dict keys:
      - label
      - horizon_steps
      - projected_value
    Returns columns: scenario, label, horizon_steps, projected_value
    """
    scenarios = getattr(forecast, "scenarios", None) or {}
    if not scenarios:
        return pd.DataFrame(columns=["scenario", "label", "horizon_steps", "projected_value"])

    rows: List[Dict[str, Any]] = []
    for key in ["best", "base", "worst"]:
        if key not in scenarios:
            continue
        s = scenarios.get(key) or {}
        rows.append(
            {
                "scenario": key,
                "label": s.get("label", key.title()),
                "horizon_steps": s.get("horizon_steps", None),
                "projected_value": s.get("projected_value", None),
            }
        )
    return pd.DataFrame(rows)


# -----------------------------
# Sentiment helpers
# -----------------------------
def sentiment_counts_df(counts: Dict[str, Any]) -> pd.DataFrame:
    """
    counts -> tidy DF for bar charts.
    Expects keys: positive/neutral/negative
    """
    counts = counts or {}
    rows = [
        {"label": "positive", "count": int(counts.get("positive", 0))},
        {"label": "neutral", "count": int(counts.get("neutral", 0))},
        {"label": "negative", "count": int(counts.get("negative", 0))},
    ]
    return pd.DataFrame(rows)


def sentiment_compound_buckets_df(
    scored: List[Dict[str, Any]],
    *,
    column: str = "compound",
    bins: Optional[List[float]] = None,
) -> pd.DataFrame:
    """
    Creates bucket counts for the VADER 'compound' score (-1..+1).
    Default bins (finance-ish): [-1,-0.5,-0.1,0.1,0.5,1]
    Output columns: bucket, count
    """
    if not scored:
        return pd.DataFrame(columns=["bucket", "count"])

    df = pd.DataFrame(scored).copy()
    if column not in df.columns:
        return pd.DataFrame(columns=["bucket", "count"])

    df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df.dropna(subset=[column])
    if df.empty:
        return pd.DataFrame(columns=["bucket", "count"])

    bins = bins or [-1.0, -0.5, -0.1, 0.1, 0.5, 1.0]
    # Create labels like "(-0.5,-0.1]"
    labels = []
    for i in range(len(bins) - 1):
        left = bins[i]
        right = bins[i + 1]
        labels.append(f"({left:.1f},{right:.1f}]")

    df["bucket"] = pd.cut(df[column], bins=bins, labels=labels, include_lowest=True)
    out = df["bucket"].value_counts(dropna=False).sort_index().reset_index()
    out.columns = ["bucket", "count"]
    out["bucket"] = out["bucket"].astype(str)
    out["count"] = out["count"].astype(int)
    return out


# -----------------------------
# Coverage helpers
# -----------------------------
def coverage_ratio(matched: int, fetched: int) -> float:
    """
    Safe matched/fetched ratio for UI.
    """
    try:
        fetched_i = int(fetched)
        matched_i = int(matched)
        if fetched_i <= 0:
            return 0.0
        return max(0.0, min(1.0, matched_i / fetched_i))
    except Exception:
        return 0.0

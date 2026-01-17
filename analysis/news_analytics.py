# analysis/news_analytics.py
# News analytics: sentiment over time + coverage strength label.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Tuple
from collections import defaultdict

import pandas as pd

from utils.helpers import clamp


Bucket = Literal["daily", "weekly"]


@dataclass(frozen=True)
class CoverageStrength:
    label: str  # "Low" | "Medium" | "High"
    matched_headlines: int
    fetched_headlines: int
    match_rate: float
    notes: List[str]


@dataclass(frozen=True)
class SentimentTimeSeries:
    bucket: Bucket
    rows: List[Dict[str, Any]]  # [{period, avg_sentiment_0_100, count}, ...]


def coverage_strength(
    *,
    fetched_count: int,
    matched_count: int,
) -> CoverageStrength:
    """
    Simple, explainable label based on matched headline volume + match rate.
    """
    notes: List[str] = []
    fetched = int(max(0, fetched_count))
    matched = int(max(0, matched_count))
    match_rate = (matched / fetched) if fetched > 0 else 0.0

    if matched >= 25 and match_rate >= 0.35:
        label = "High"
    elif matched >= 10 and match_rate >= 0.20:
        label = "Medium"
    else:
        label = "Low"

    if fetched == 0:
        notes.append("No news records fetched (query may be too narrow or network issue).")
    elif matched == 0:
        notes.append("No matched headlines; try changing context or use a more specific keyword.")
    elif match_rate < 0.15:
        notes.append("Low match rate; keyword may be ambiguous.")

    return CoverageStrength(
        label=label,
        matched_headlines=matched,
        fetched_headlines=fetched,
        match_rate=round(float(match_rate * 100.0), 2),
        notes=notes,
    )


def _to_period(dt: pd.Timestamp, bucket: Bucket) -> str:
    if bucket == "daily":
        return dt.strftime("%Y-%m-%d")
    # weekly: ISO week
    iso_year, iso_week, _ = dt.isocalendar()
    return f"{iso_year}-W{int(iso_week):02d}"


def build_sentiment_timeseries(
    scored_headlines: List[Dict[str, Any]],
    *,
    bucket: Bucket = "weekly",
) -> SentimentTimeSeries:
    """
    Input: scored_headlines from nlp.sentiment.score_headlines_vader
      each item expected to include:
        - date (ISO string)
        - sentiment_0_100 (float) OR compound (float -1..1)
    Output: grouped series (daily/weekly) with avg sentiment and count.
    """
    # Collect by bucket
    groups: Dict[str, List[float]] = defaultdict(list)

    for h in scored_headlines or []:
        date_str = h.get("date")
        if not date_str:
            continue
        try:
            dt = pd.to_datetime(date_str, utc=True)
        except Exception:
            continue

        # Prefer sentiment_0_100 if present, else derive from compound
        if "sentiment_0_100" in h and h["sentiment_0_100"] is not None:
            try:
                s = float(h["sentiment_0_100"])
            except Exception:
                continue
        else:
            try:
                compound = float(h.get("compound", 0.0))
                s = (compound + 1.0) * 50.0  # map -1..1 → 0..100
            except Exception:
                continue

        s = float(clamp(s, 0.0, 100.0))
        period = _to_period(dt, bucket)
        groups[period].append(s)

    rows: List[Dict[str, Any]] = []
    for period, vals in groups.items():
        if not vals:
            continue
        rows.append(
            {
                "period": period,
                "avg_sentiment_0_100": round(float(sum(vals) / len(vals)), 2),
                "count": int(len(vals)),
            }
        )

    # Sort periods
    rows.sort(key=lambda r: r["period"])

    return SentimentTimeSeries(bucket=bucket, rows=rows)

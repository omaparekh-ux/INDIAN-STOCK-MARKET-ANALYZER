# analysis/metrics.py
# Compute trend growth, averages, volatility + minimum-data checks.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from config import DATA_QUALITY_CFG
from utils.helpers import safe_div


@dataclass(frozen=True)
class TrendMetrics:
    points: int
    past_avg: float
    recent_avg: float
    growth_pct: float
    volatility: float  # std dev of values
    min_data_ok: bool
    warnings: List[str]


def _extract_values(timeseries: List[Dict[str, Any]]) -> List[float]:
    vals: List[float] = []
    for p in timeseries or []:
        try:
            v = float(p.get("value", 0.0))
        except Exception:
            v = 0.0
        vals.append(v)
    return vals


def compute_trend_metrics(timeseries: List[Dict[str, Any]]) -> TrendMetrics:
    """
    Computes:
      - past_avg: avg of first half
      - recent_avg: avg of second half
      - growth_pct: ((recent - past) / past) * 100
      - volatility: std dev
      - min_data_ok + warnings (based on config)
    """
    values = _extract_values(timeseries)
    n = len(values)

    warnings: List[str] = []
    if n < DATA_QUALITY_CFG.min_trend_points:
        warnings.append(
            f"Not enough trend data points ({n}) for high confidence; "
            f"recommended >= {DATA_QUALITY_CFG.min_trend_points}."
        )

    if n == 0:
        return TrendMetrics(
            points=0,
            past_avg=0.0,
            recent_avg=0.0,
            growth_pct=0.0,
            volatility=0.0,
            min_data_ok=False,
            warnings=warnings + ["Trend data is empty."],
        )

    mid = n // 2
    past = values[:mid] if mid > 0 else values
    recent = values[mid:] if mid > 0 else values

    past_avg = float(np.mean(past)) if len(past) else 0.0
    recent_avg = float(np.mean(recent)) if len(recent) else 0.0

    growth_pct = safe_div((recent_avg - past_avg) * 100.0, past_avg, default=0.0)

    volatility = float(np.std(values)) if n > 1 else 0.0

    min_data_ok = n >= DATA_QUALITY_CFG.min_trend_points

    return TrendMetrics(
        points=n,
        past_avg=round(past_avg, 2),
        recent_avg=round(recent_avg, 2),
        growth_pct=round(float(growth_pct), 2),
        volatility=round(volatility, 2),
        min_data_ok=min_data_ok,
        warnings=warnings,
    )


@dataclass(frozen=True)
class NewsQuality:
    fetched_count: int
    matched_count: int
    match_quality: float
    min_data_ok: bool
    warnings: List[str]


def compute_news_quality(fetched_count: int, matched_count: int) -> NewsQuality:
    """
    Computes match quality % and minimum headline count warnings.
    """
    warnings: List[str] = []

    if fetched_count <= 0:
        return NewsQuality(
            fetched_count=0,
            matched_count=0,
            match_quality=0.0,
            min_data_ok=False,
            warnings=["No news headlines were fetched."],
        )

    match_quality = matched_count / fetched_count if fetched_count > 0 else 0.0

    if matched_count < DATA_QUALITY_CFG.min_matched_headlines:
        warnings.append(
            f"Low number of matched headlines ({matched_count}); "
            f"recommended >= {DATA_QUALITY_CFG.min_matched_headlines}. "
            f"Sentiment confidence may be low."
        )

    min_data_ok = matched_count >= DATA_QUALITY_CFG.min_matched_headlines

    return NewsQuality(
        fetched_count=int(fetched_count),
        matched_count=int(matched_count),
        match_quality=round(float(match_quality), 4),
        min_data_ok=min_data_ok,
        warnings=warnings,
    )

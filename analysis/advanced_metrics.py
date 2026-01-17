# analysis/advanced_metrics.py
# Advanced trend analytics: momentum, consistency, spikes, acceleration.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.linear_model import LinearRegression

from config import FORECAST_SLOPE_FLAT_EPS
from utils.helpers import clamp


@dataclass(frozen=True)
class AdvancedTrendMetrics:
    momentum_pct: float
    consistency_score: float
    spike_index: int
    spike_label: str
    recent_slope: float
    long_slope: float
    acceleration: float
    notes: List[str]


def _extract_values(timeseries: List[Dict[str, Any]]) -> List[float]:
    vals: List[float] = []
    for p in timeseries or []:
        try:
            vals.append(float(p.get("value", 0.0)))
        except Exception:
            vals.append(0.0)
    return vals


def _linear_slope(values: List[float]) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    X = np.arange(n).reshape(-1, 1)
    y = np.array(values).reshape(-1, 1)
    model = LinearRegression()
    model.fit(X, y)
    return float(model.coef_[0][0])


def _avg(values: List[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _chunk_for_momentum(values: List[float]) -> Tuple[List[float], List[float]]:
    """
    Momentum definition (explainable):
      - Recent = last 4 points
      - Past = previous 8 points
    If series is short, falls back to best-effort splits.
    """
    if len(values) >= 12:
        past = values[-12:-4]
        recent = values[-4:]
        return past, recent

    # Fallback: split 2/3 vs 1/3
    n = len(values)
    if n < 4:
        return values[:], values[:]
    cut = max(1, int(n * 0.66))
    past = values[:cut]
    recent = values[cut:]
    return past, recent


def _consistency_score(values: List[float]) -> float:
    """
    Consistency = % of steps that are upward (0..100).
    """
    if len(values) < 2:
        return 0.0
    ups = 0
    total = 0
    for i in range(1, len(values)):
        if values[i] > values[i - 1]:
            ups += 1
        total += 1
    return float((ups / total) * 100.0) if total else 0.0


def _spike_index(values: List[float]) -> Tuple[int, str]:
    """
    Spike if value > mean + 2*std.
    Returns (count, label).
    """
    if len(values) < 6:
        return 0, "Unknown (too little data)"
    mean = float(np.mean(values))
    std = float(np.std(values))
    if std == 0:
        return 0, "Normal"
    threshold = mean + (2.0 * std)
    spikes = sum(1 for v in values if v > threshold)

    if spikes >= 3:
        return spikes, "Very Spiky"
    if spikes >= 1:
        return spikes, "Spiky"
    return spikes, "Normal"


def compute_advanced_trend_metrics(timeseries: List[Dict[str, Any]]) -> AdvancedTrendMetrics:
    """
    Adds pro-level, explainable analytics on top of the basic trend chart:
      - Momentum % (recent vs past)
      - Consistency score (how often it rises)
      - Spike index + label
      - Acceleration (recent slope - long slope)
    """
    notes: List[str] = []
    values = _extract_values(timeseries)

    if len(values) < 2:
        return AdvancedTrendMetrics(
            momentum_pct=0.0,
            consistency_score=0.0,
            spike_index=0,
            spike_label="Unknown (too little data)",
            recent_slope=0.0,
            long_slope=0.0,
            acceleration=0.0,
            notes=["Not enough data points for advanced metrics."],
        )

    # Momentum
    past, recent = _chunk_for_momentum(values)
    past_avg = _avg(past)
    recent_avg = _avg(recent)

    if past_avg <= 0:
        momentum_pct = 0.0
        notes.append("Momentum set to 0 because past average is near zero.")
    else:
        momentum_pct = ((recent_avg - past_avg) / past_avg) * 100.0

    # Consistency
    consistency = _consistency_score(values)

    # Spikes
    spike_count, spike_label = _spike_index(values)

    # Acceleration via slopes
    long_slope = _linear_slope(values)
    recent_window = values[-8:] if len(values) >= 8 else values[:]
    recent_slope = _linear_slope(recent_window)
    acceleration = recent_slope - long_slope

    # Small helpful note
    if abs(acceleration) <= FORECAST_SLOPE_FLAT_EPS:
        notes.append("Acceleration is near-flat (trend speed not changing much).")
    elif acceleration > 0:
        notes.append("Acceleration is positive (trend is speeding up).")
    else:
        notes.append("Acceleration is negative (trend is slowing down).")

    return AdvancedTrendMetrics(
        momentum_pct=round(float(momentum_pct), 2),
        consistency_score=round(float(clamp(consistency, 0.0, 100.0)), 2),
        spike_index=int(spike_count),
        spike_label=spike_label,
        recent_slope=round(float(recent_slope), 4),
        long_slope=round(float(long_slope), 4),
        acceleration=round(float(acceleration), 4),
        notes=notes,
    )

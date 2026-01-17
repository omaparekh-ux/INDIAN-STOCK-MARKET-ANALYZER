# analysis/forecast.py
# Moving average smoothing + linear regression slope + confidence label.
# Updated: Adds best/base/worst scenario projections + plain-English explanation (backward compatible).

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np
from sklearn.linear_model import LinearRegression

from config import FORECAST_SLOPE_FLAT_EPS, DATA_QUALITY_CFG
from utils.helpers import clamp


@dataclass(frozen=True)
class ForecastResult:
    direction: str           # "Up" | "Flat" | "Down"
    slope: float             # regression slope
    confidence: str          # "High" | "Medium" | "Low"
    smoothed: List[Dict[str, Any]]  # [{date, value}, ...]
    notes: List[str]

    # New fields (safe defaults → backward compatible)
    scenarios: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    explanation: str = ""


def moving_average(values: List[float], window: int = 3) -> List[float]:
    """
    Simple moving average (explainable). Window defaults to 3.
    """
    if not values:
        return []
    if window <= 1:
        return values[:]
    out: List[float] = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        chunk = values[start : i + 1]
        out.append(float(sum(chunk) / len(chunk)))
    return out


def _extract(timeseries: List[Dict[str, Any]]) -> tuple[list[str], list[float]]:
    dates: List[str] = []
    vals: List[float] = []
    for p in timeseries or []:
        dates.append(str(p.get("date", "")))
        try:
            vals.append(float(p.get("value", 0.0)))
        except Exception:
            vals.append(0.0)
    return dates, vals


def regression_slope(values: List[float]) -> float:
    """
    Linear regression slope of value vs time index.
    """
    n = len(values)
    if n < 2:
        return 0.0
    X = np.arange(n).reshape(-1, 1)
    y = np.array(values).reshape(-1, 1)
    model = LinearRegression()
    model.fit(X, y)
    return float(model.coef_[0][0])


def slope_to_direction(slope: float) -> str:
    if slope > FORECAST_SLOPE_FLAT_EPS:
        return "Up"
    if slope < -FORECAST_SLOPE_FLAT_EPS:
        return "Down"
    return "Flat"


def confidence_label(
    *,
    volatility: float,
    match_quality: float,
    trend_points: int,
    matched_headlines: int,
) -> str:
    """
    Simple confidence heuristic (viva-friendly):
    - Needs enough trend points + enough headlines
    - Higher match quality helps
    - Lower volatility helps
    """
    if trend_points < DATA_QUALITY_CFG.min_trend_points or matched_headlines < DATA_QUALITY_CFG.min_matched_headlines:
        return "Low"

    if volatility <= 12 and match_quality >= DATA_QUALITY_CFG.match_quality_high:
        return "High"
    if volatility <= 20 and match_quality >= DATA_QUALITY_CFG.match_quality_med:
        return "Medium"
    return "Low"


def _scenario_projection(
    last_value: float,
    slope: float,
    steps: int,
    multiplier: float,
) -> float:
    """
    Very simple projection: last_value + (slope * steps * multiplier)
    Multiplier is used to create best/base/worst scenarios.

    NOTE: This is not a prediction of truth, it's an illustrative scenario.
    """
    projected = last_value + (slope * float(steps) * float(multiplier))
    return float(clamp(projected, 0.0, 100.0))


def _build_scenarios(
    last_value: float,
    slope: float,
    confidence: str,
    volatility: float,
) -> Dict[str, Dict[str, Any]]:
    """
    Creates best/base/worst scenario projections for a short horizon (next 12 steps).
    - Confidence controls the spread
    - Volatility increases spread slightly
    """
    steps = 12  # illustrative near-term window (e.g., ~12 weeks if weekly)
    base_mult = 1.0

    if confidence == "High":
        spread = 0.6
    elif confidence == "Medium":
        spread = 1.0
    else:
        spread = 1.4

    # If very volatile, widen the scenario band a bit
    if volatility >= 25:
        spread *= 1.2

    best_mult = base_mult + spread
    worst_mult = base_mult - spread

    base = _scenario_projection(last_value, slope, steps, base_mult)
    best = _scenario_projection(last_value, slope, steps, best_mult)
    worst = _scenario_projection(last_value, slope, steps, worst_mult)

    return {
        "base": {"label": "Base case", "horizon_steps": steps, "projected_value": round(base, 2)},
        "best": {"label": "Best case", "horizon_steps": steps, "projected_value": round(best, 2)},
        "worst": {"label": "Worst case", "horizon_steps": steps, "projected_value": round(worst, 2)},
    }


def _explanation(direction: str, confidence: str) -> str:
    if confidence == "Low":
        return "This forecast is indicative only; data is limited or unstable, so validate with a small test before acting."
    if direction == "Up":
        return "The smoothed trend is rising, suggesting near-term demand may increase if current conditions continue."
    if direction == "Down":
        return "The smoothed trend is falling, suggesting near-term demand may cool down if current conditions continue."
    return "The smoothed trend is mostly flat, suggesting stable demand in the near term."


def compute_forecast(
    timeseries: List[Dict[str, Any]],
    *,
    volatility: float,
    match_quality: float,
    trend_points: int,
    matched_headlines: int,
) -> ForecastResult:
    """
    Produces:
    - smoothed series (moving average)
    - slope (linear regression)
    - direction (Up/Flat/Down)
    - confidence label
    - scenarios (best/base/worst)
    - explanation
    """
    notes: List[str] = []
    dates, vals = _extract(timeseries)

    if not vals:
        return ForecastResult(
            direction="Flat",
            slope=0.0,
            confidence="Low",
            smoothed=[],
            notes=["No trend series available for forecasting."],
            scenarios={},
            explanation="No forecast could be generated because trend data is missing.",
        )

    smoothed_vals = moving_average(vals, window=3)

    slope = regression_slope(smoothed_vals)
    direction = slope_to_direction(slope)

    conf = confidence_label(
        volatility=volatility,
        match_quality=match_quality,
        trend_points=trend_points,
        matched_headlines=matched_headlines,
    )

    if conf == "Low":
        notes.append("Forecast confidence is low due to limited/unstable data.")
    if volatility > 25:
        notes.append("Trend is highly volatile (possible hype/spikes).")

    smoothed = [{"date": dates[i], "value": round(float(smoothed_vals[i]), 2)} for i in range(len(smoothed_vals))]

    last_value = float(smoothed_vals[-1]) if smoothed_vals else float(vals[-1])
    scenarios = _build_scenarios(last_value=last_value, slope=float(slope), confidence=conf, volatility=volatility)

    # Make sure we explicitly tell the user what scenarios are
    if scenarios:
        notes.append("Best/Base/Worst scenarios are illustrative projections, not guaranteed outcomes.")

    explanation = _explanation(direction, conf)

    return ForecastResult(
        direction=direction,
        slope=round(float(slope), 4),
        confidence=conf,
        smoothed=smoothed,
        notes=notes,
        scenarios=scenarios,
        explanation=explanation,
    )

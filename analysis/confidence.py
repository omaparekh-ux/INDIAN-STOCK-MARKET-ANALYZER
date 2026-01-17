# analysis/confidence.py
# Confidence scoring: how reliable is the trend decision?
# Updated: adds data_coverage + explanation (backward-compatible), improves plain-English reasons.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict

from utils.helpers import clamp


@dataclass(frozen=True)
class ConfidenceResult:
    confidence: float  # 0..100
    label: str         # Low | Medium | High
    reasons: List[str]

    # New fields (safe defaults → backward compatible)
    data_coverage: Dict[str, str] = field(default_factory=dict)  # e.g. {"trends": "strong", "news": "medium"}
    explanation: str = ""  # single plain-English sentence


def _coverage_bucket(value: float, strong_thr: float, med_thr: float) -> str:
    if value >= strong_thr:
        return "strong"
    if value >= med_thr:
        return "medium"
    return "weak"


def compute_confidence_score(
    *,
    trend_points: int,
    matched_headlines: int,
    match_quality: float,      # 0..1
    volatility: float,         # 0..100-ish
    growth_pct: float,
    momentum_pct: float,
    slope: float,
) -> ConfidenceResult:
    """
    Explainable confidence score:
      - More trend points and more matched headlines => higher confidence
      - Higher match_quality => higher confidence
      - Higher volatility => lower confidence
      - Agreement between growth, momentum and slope => higher confidence

    Adds:
      - data_coverage buckets (trends/news/match_quality/volatility)
      - a 1-sentence explanation suitable for non-technical users
    """
    reasons: List[str] = []

    # -----------------------------
    # Base from data quantity
    # -----------------------------
    points_score = clamp((trend_points / 52.0) * 30.0, 0.0, 30.0)  # up to 30
    headlines_score = clamp((matched_headlines / 30.0) * 25.0, 0.0, 25.0)  # up to 25

    # Match quality (0..1) -> up to 20
    mq_score = clamp(match_quality * 20.0, 0.0, 20.0)

    # Volatility penalty (0..25)
    vol_penalty = clamp((volatility / 25.0) * 25.0, 0.0, 25.0)

    # -----------------------------
    # Agreement bonus (0..25)
    # -----------------------------
    def sign(x: float) -> int:
        if x > 0:
            return 1
        if x < 0:
            return -1
        return 0

    sg = sign(growth_pct)
    sm = sign(momentum_pct)
    ss = sign(slope)

    if sg == sm == ss and sg != 0:
        agreement = 25
        reasons.append("Growth, momentum, and forecast slope all point in the same direction.")
    elif (sg == sm and sg != 0) or (sg == ss and sg != 0) or (sm == ss and sm != 0):
        agreement = 15
        reasons.append("Two signals agree on direction (partial agreement).")
    else:
        agreement = 5
        reasons.append("Signals disagree on direction (higher uncertainty).")

    # Final score
    score = points_score + headlines_score + mq_score + agreement - vol_penalty
    score = clamp(score, 0.0, 100.0)

    # -----------------------------
    # Human-readable reasons
    # -----------------------------
    # Trend history
    if trend_points < 20:
        reasons.append("Few trend data points; trend estimates may be unstable.")
    elif trend_points < 40:
        reasons.append("Moderate trend history; usable but not ideal for long horizons.")
    else:
        reasons.append("Strong trend history; trend estimates are more reliable.")

    # News / headline volume
    if matched_headlines < 5:
        reasons.append("Very few matched headlines; sentiment and context may be unreliable.")
    elif matched_headlines < 12:
        reasons.append("Moderate headline coverage; sentiment is usable but limited.")
    else:
        reasons.append("Good headline coverage; sentiment/context are more reliable.")

    # Match quality
    if match_quality < 0.2:
        reasons.append("Low match quality; keyword may be ambiguous (try a more specific term).")
    elif match_quality < 0.4:
        reasons.append("Medium match quality; results are okay but could be clearer with a refined keyword.")
    else:
        reasons.append("High match quality; news relevance is strong.")

    # Volatility
    if volatility >= 25:
        reasons.append("Trend is highly volatile; short-term spikes may distort conclusions.")
    elif volatility >= 12:
        reasons.append("Trend has moderate volatility; timing/seasonality may matter.")
    else:
        reasons.append("Trend is relatively stable (low volatility).")

    # -----------------------------
    # Label
    # -----------------------------
    if score >= 70:
        label = "High"
    elif score >= 45:
        label = "Medium"
    else:
        label = "Low"

    # -----------------------------
    # Data coverage buckets (for dashboard transparency)
    # -----------------------------
    data_coverage: Dict[str, str] = {
        "trends": _coverage_bucket(float(trend_points), strong_thr=40.0, med_thr=20.0),
        "news": _coverage_bucket(float(matched_headlines), strong_thr=12.0, med_thr=5.0),
        "match_quality": _coverage_bucket(float(match_quality), strong_thr=0.40, med_thr=0.20),
        "stability": "weak" if volatility >= 25 else ("medium" if volatility >= 12 else "strong"),
    }

    # -----------------------------
    # One-line explanation (non-technical)
    # -----------------------------
    if label == "High":
        explanation = "This conclusion is supported by strong trend history, relevant news signals, and consistent direction."
    elif label == "Medium":
        explanation = "This conclusion is reasonably supported, but some signals are limited or mixed, so treat it as a guide and validate with a small test."
    else:
        explanation = "This conclusion is low-confidence due to limited or unclear data; monitor longer and re-check before investing heavily."

    return ConfidenceResult(
        confidence=round(float(score), 2),
        label=label,
        reasons=reasons,
        data_coverage=data_coverage,
        explanation=explanation,
    )

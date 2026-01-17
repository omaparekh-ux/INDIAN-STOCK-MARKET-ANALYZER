# analysis/scoring.py
# Normalize growth + sentiment and compute Trend Strength Score (0-100).

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from config import TREND_SCORE_CFG, SCORE_MODERATE_MIN, SCORE_STRONG_MIN
from utils.helpers import clamp, normalize_to_0_100
from analysis.metrics import TrendMetrics


@dataclass(frozen=True)
class TrendScore:
    score: float
    classification: str
    components: Dict[str, float]
    warnings: List[str]


def classify_score(score: float) -> str:
    if score >= SCORE_STRONG_MIN:
        return "Strong Trend"
    if score >= SCORE_MODERATE_MIN:
        return "Moderate Trend"
    return "Weak / Declining Trend"


def compute_trend_score(
    metrics: TrendMetrics,
    sentiment_0_100: float,
) -> TrendScore:
    """
    Locked scoring:
      - Clamp growth_pct to [-50, +150]
      - Normalize clamped growth to [0, 100]
      - Sentiment is already [0, 100]
      - Final Score = 0.6*Growth + 0.4*Sentiment
    """

    warnings: List[str] = []
    warnings.extend(metrics.warnings)

    # 1) Clamp growth
    clamped_growth = clamp(metrics.growth_pct, TREND_SCORE_CFG.growth_clamp_min, TREND_SCORE_CFG.growth_clamp_max)

    # 2) Normalize growth to 0-100
    growth_0_100 = normalize_to_0_100(
        clamped_growth,
        TREND_SCORE_CFG.growth_clamp_min,
        TREND_SCORE_CFG.growth_clamp_max,
    )

    # 3) Clamp sentiment to 0-100 (safety)
    sentiment_0_100 = float(clamp(sentiment_0_100, 0.0, 100.0))

    # 4) Weighted score
    score = (growth_0_100 * TREND_SCORE_CFG.weight_growth) + (sentiment_0_100 * TREND_SCORE_CFG.weight_sentiment)
    score = round(float(score), 2)

    classification = classify_score(score)

    return TrendScore(
        score=score,
        classification=classification,
        components={
            "growth_pct": round(float(metrics.growth_pct), 2),
            "growth_clamped": round(float(clamped_growth), 2),
            "growth_0_100": round(float(growth_0_100), 2),
            "sentiment_0_100": round(float(sentiment_0_100), 2),
            "w_growth": float(TREND_SCORE_CFG.weight_growth),
            "w_sentiment": float(TREND_SCORE_CFG.weight_sentiment),
        },
        warnings=warnings,
    )

# analysis/pro_scoring.py
# Pro scoring engine: combines growth, momentum, sentiment, consistency + volatility penalty.

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

from utils.helpers import clamp


@dataclass(frozen=True)
class ProScoreResult:
    pro_score: float                 # 0..100
    classification: str              # Strong | Moderate | Weak
    components: Dict[str, float]     # each 0..100 (except penalty)
    notes: List[str]


def _scale_growth_pct_to_0_100(growth_pct: float) -> float:
    """
    Map growth% to 0..100 in an explainable way.
    - -50% or below => 0
    - 0% => 50
    - +50% or above => 100
    """
    g = float(growth_pct)
    g = clamp(g, -50.0, 50.0)
    return ((g + 50.0) / 100.0) * 100.0


def _scale_momentum_pct_to_0_100(momentum_pct: float) -> float:
    """
    Similar mapping for momentum.
    - -40% => 0
    - 0% => 50
    - +40% => 100
    """
    m = float(momentum_pct)
    m = clamp(m, -40.0, 40.0)
    return ((m + 40.0) / 80.0) * 100.0


def _scale_sentiment_to_0_100(sentiment_0_100: float) -> float:
    return float(clamp(sentiment_0_100, 0.0, 100.0))


def _scale_consistency_to_0_100(consistency_score: float) -> float:
    return float(clamp(consistency_score, 0.0, 100.0))


def _volatility_penalty(volatility: float) -> float:
    """
    Penalize very spiky trends. Volatility typically 0..30+
    Map:
      0..10 => 0..10 penalty small
      10..25 => 10..20 penalty
      25+ => up to 30 penalty (capped)
    Returned as 0..30 (to subtract).
    """
    v = float(max(0.0, volatility))
    if v <= 10:
        return clamp(v, 0.0, 10.0)
    if v <= 25:
        return 10.0 + clamp((v - 10.0) * (10.0 / 15.0), 0.0, 10.0)
    return clamp(20.0 + (v - 25.0), 0.0, 30.0)


def compute_pro_score(
    *,
    growth_pct: float,
    momentum_pct: float,
    sentiment_0_100: float,
    consistency_score: float,
    volatility: float,
) -> ProScoreResult:
    """
    Pro Score (explainable):
      score = 0.30*Growth + 0.20*Momentum + 0.20*Sentiment + 0.20*Consistency - 0.10*VolPenalty
    Returns score + components for UI breakdown.
    """
    notes: List[str] = []

    growth_score = _scale_growth_pct_to_0_100(growth_pct)
    momentum_score = _scale_momentum_pct_to_0_100(momentum_pct)
    sentiment_score = _scale_sentiment_to_0_100(sentiment_0_100)
    consistency_scaled = _scale_consistency_to_0_100(consistency_score)
    vol_pen = _volatility_penalty(volatility)

    raw = (
        0.30 * growth_score
        + 0.20 * momentum_score
        + 0.20 * sentiment_score
        + 0.20 * consistency_scaled
        - 0.10 * vol_pen
    )
    score = float(clamp(raw, 0.0, 100.0))

    # Classification
    if score >= 70:
        classification = "Strong"
    elif score >= 40:
        classification = "Moderate"
    else:
        classification = "Weak"

    # Notes for explainability
    if vol_pen >= 20:
        notes.append("High volatility penalty: trend may be hype-driven.")
    elif vol_pen >= 10:
        notes.append("Moderate volatility penalty: some spikiness present.")
    else:
        notes.append("Low volatility penalty: trend appears stable.")

    if sentiment_score >= 60:
        notes.append("News sentiment is supportive.")
    elif sentiment_score <= 40:
        notes.append("News sentiment is unfavorable.")
    else:
        notes.append("News sentiment is mixed/neutral.")

    if momentum_score >= 60:
        notes.append("Recent momentum is strong.")
    elif momentum_score <= 40:
        notes.append("Recent momentum is weak.")
    else:
        notes.append("Recent momentum is moderate.")

    return ProScoreResult(
        pro_score=round(score, 2),
        classification=classification,
        components={
            "growth_score": round(float(growth_score), 2),
            "momentum_score": round(float(momentum_score), 2),
            "sentiment_score": round(float(sentiment_score), 2),
            "consistency_score": round(float(consistency_scaled), 2),
            "volatility_penalty": round(float(vol_pen), 2),
        },
        notes=notes,
    )

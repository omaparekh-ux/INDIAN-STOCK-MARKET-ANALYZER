# analysis/reco.py
# Business-ready conclusions: hype vs sustainable, decision summary, recommendations.
# Updated: Adds non-technical explainability fields (reasons/risk/takeaway) in a backward-compatible way.
# Updated (this revision): fixes robustness + keeps fields stable + improves reason selection without changing UI/logic elsewhere.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Any

from analysis.metrics import TrendMetrics, NewsQuality
from analysis.forecast import ForecastResult
from config import DATA_QUALITY_CFG


@dataclass(frozen=True)
class RecommendationPack:
    # Existing fields (do not change / remove)
    market_status: str
    hype_label: str
    decision: str
    entry_window: str
    bullets: List[str]
    cautions: List[str]

    # New fields (safe defaults → backward compatible)
    key_reasons: List[str] = field(default_factory=list)          # 2–3 plain-English reasons
    biggest_risk: str = ""                                        # single biggest risk
    takeaway: str = ""                                            # “If you remember one thing…”
    who_should_not_enter: List[str] = field(default_factory=list) # self-filtering warnings
    how_to_use_wisely: str = ""                                   # short guidance for the user


# -----------------------------
# Backward-compatible score parsing
# -----------------------------
def _extract_score_value(score: Any) -> float:
    """
    Backward-compatible score extraction.

    Supports:
    - Old TrendScore object: score.score
    - New ProScoreResult object: score.pro_score
    - Plain float/int: treated as score
    """
    if score is None:
        return 50.0

    if isinstance(score, (int, float)):
        return float(score)

    if hasattr(score, "score"):
        try:
            return float(getattr(score, "score"))
        except Exception:
            pass

    if hasattr(score, "pro_score"):
        try:
            return float(getattr(score, "pro_score"))
        except Exception:
            pass

    return 50.0


def _extract_classification(score: Any) -> str:
    """
    Tries to read a classification label from score object.
    Fallbacks to a basic label if missing.
    """
    if score is None:
        return "Moderate / Uncertain Trend"

    if hasattr(score, "classification"):
        try:
            val = str(getattr(score, "classification"))
            return val if val else "Moderate / Uncertain Trend"
        except Exception:
            pass

    # If older object used a different attribute name
    if hasattr(score, "class_label"):
        try:
            val = str(getattr(score, "class_label"))
            return val if val else "Moderate / Uncertain Trend"
        except Exception:
            pass

    # Fallback bucket based on numeric score
    score_value = _extract_score_value(score)
    if score_value >= 70:
        return "Strong Emerging Trend"
    if score_value >= 40:
        return "Moderate / Uncertain Trend"
    return "Weak / Declining Trend"


# -----------------------------
# Core explainable logic
# -----------------------------
def hype_vs_sustainable(metrics: TrendMetrics, forecast: ForecastResult) -> str:
    """
    Simple, explainable rule:
    - High volatility + sharp growth often = hype/seasonality
    - Lower volatility + upward forecast = sustainable
    """
    if metrics.volatility >= 25 and metrics.growth_pct >= 30:
        return "Likely Hype / Spike-driven"
    if forecast.direction == "Up" and metrics.volatility <= 18:
        return "Likely Sustainable Growth"
    if forecast.direction == "Down":
        return "Cooling / Declining Interest"
    return "Unclear / Mixed Signals"


def _entry_window(score_value: float, forecast: ForecastResult) -> str:
    if score_value >= 70 and forecast.direction in ("Up", "Flat"):
        return "Next 3–6 months"
    if score_value >= 40 and forecast.direction == "Up":
        return "Test in next 1–3 months, scale later"
    if forecast.direction == "Down":
        return "Avoid new entry for now"
    return "Monitor for 4–8 weeks"


def _decision(score_value: float, forecast: ForecastResult, data_low: bool) -> str:
    if data_low:
        return "INSUFFICIENT DATA — Monitor and re-check"
    if score_value >= 70 and forecast.direction == "Up":
        return "ENTER / INVEST — High potential"
    if score_value >= 70 and forecast.direction == "Flat":
        return "ENTER CAREFULLY — Strong but stable"
    if score_value >= 40 and forecast.direction == "Up":
        return "TEST MARKET — Moderate growth, upside"
    if score_value >= 40 and forecast.direction == "Flat":
        return "HOLD — Stable but not accelerating"
    return "WAIT / AVOID — Weak signals"


# -----------------------------
# Non-technical explainability fields
# -----------------------------
def _build_key_reasons(
    *,
    score_value: float,
    metrics: TrendMetrics,
    news_q: NewsQuality,
    forecast: ForecastResult,
    data_low: bool,
) -> List[str]:
    """
    Picks 2–3 plain-English reasons.
    Kept short + deterministic + not overly repetitive.
    """
    reasons: List[str] = []

    if data_low:
        reasons.append("Not enough reliable data was found to make a confident call.")
        reasons.append("Try a more specific keyword or expand the news window and re-check.")
        reasons.append("Use this only as a direction—not as a final decision—until data improves.")
        return reasons[:3]

    # 1) Demand signal (growth)
    if metrics.growth_pct >= 12:
        reasons.append("Search interest is rising, suggesting growing buyer demand.")
    elif metrics.growth_pct <= -8:
        reasons.append("Search interest is falling, suggesting weaker buyer demand.")
    else:
        reasons.append("Search interest is relatively stable (no strong growth signal).")

    # 2) Direction signal (forecast)
    if forecast.direction == "Up":
        reasons.append("Near-term direction is upward, which supports testing or entry.")
    elif forecast.direction == "Down":
        reasons.append("Near-term direction is downward, which increases risk of losses.")
    else:
        reasons.append("Near-term direction is flat, so growth may not accelerate soon.")

    # 3) Data clarity signal (news match quality) OR volatility (whichever is more important)
    if news_q.match_quality < DATA_QUALITY_CFG.match_quality_med:
        reasons.append("News relevance is weak; the keyword may be ambiguous, reducing reliability.")
    elif metrics.volatility >= 25:
        reasons.append("Interest is very volatile, which can indicate hype/seasonality.")
    else:
        reasons.append("News relevance is decent and trend stability is acceptable.")

    # Keep 3 max
    return reasons[:3]


def _biggest_risk(
    *,
    metrics: TrendMetrics,
    news_q: NewsQuality,
    forecast: ForecastResult,
    data_low: bool,
) -> str:
    if data_low:
        return "Limited data can produce misleading conclusions."
    if forecast.direction == "Down":
        return "Demand may decline, creating inventory and marketing loss risk."
    if metrics.volatility >= 25:
        return "Demand may be spike-driven or seasonal, so forecasts may not hold."
    if news_q.match_quality < DATA_QUALITY_CFG.match_quality_med:
        return "Keyword ambiguity can distort results (you may be analyzing the wrong meaning)."
    return "Execution risk: branding, distribution, and trust will decide outcomes more than trends."


def _takeaway(decision: str, metrics: TrendMetrics) -> str:
    d = (decision or "").upper()
    if "INSUFFICIENT" in d:
        return "Don’t make a big move yet—collect more signals first."
    if "ENTER" in d and metrics.volatility >= 25:
        return "Enter, but keep inventory light until demand proves stable."
    if "ENTER" in d:
        return "Move early, but win with differentiation—not price wars."
    if "TEST" in d:
        return "Start small, validate fast, then scale what works."
    if "HOLD" in d or "WAIT" in d:
        return "Wait for clearer upward momentum before committing serious budget."
    return "Keep it niche and low-risk until signals improve."


def _who_should_not_enter(score_value: float, forecast: ForecastResult, metrics: TrendMetrics) -> List[str]:
    out: List[str] = []
    if score_value < 40 or forecast.direction == "Down":
        out.append("People expecting quick profits without testing and patience.")
    if metrics.volatility >= 25:
        out.append("Founders who cannot handle demand swings or inventory risk.")
    out.append("Anyone planning to copy existing products without differentiation.")
    return out[:3]


def _how_to_use_wisely_text() -> str:
    return (
        "Use this analysis to choose: ENTER vs TEST vs WAIT. "
        "Then validate using a small pilot (ads, samples, limited inventory) before scaling."
    )


# -----------------------------
# Main builder
# -----------------------------
def build_recommendations(
    *,
    keyword: str,
    metrics: TrendMetrics,
    news_q: NewsQuality,
    score: Any,  # accepts both TrendScore and ProScoreResult
    forecast: ForecastResult,
    region_label: str,
) -> RecommendationPack:
    """
    Produces:
    - market_status (classification)
    - hype label
    - decision summary
    - entry window
    - bullets + cautions
    - non-technical explainability fields
    """
    # More robust data_low check: handle missing attrs safely
    metrics_ok = bool(getattr(metrics, "min_data_ok", True))
    news_ok = bool(getattr(news_q, "min_data_ok", True))
    data_low = (not metrics_ok) or (not news_ok)

    score_value = _extract_score_value(score)
    market_status = _extract_classification(score)

    hype_label = hype_vs_sustainable(metrics, forecast)
    decision = _decision(score_value, forecast, data_low=data_low)
    entry_window = _entry_window(score_value, forecast)

    bullets: List[str] = []
    cautions: List[str] = []

    # Core recommendation bullets
    if score_value >= 70:
        bullets.append("High potential: prioritize product/market entry planning.")
        bullets.append("Use performance marketing + influencer content to capture demand early.")
        bullets.append("Validate pricing and positioning with quick A/B tests.")
    elif score_value >= 40:
        bullets.append("Moderate potential: run pilot campaigns and measure conversion.")
        bullets.append("Focus on differentiated messaging to stand out.")
        bullets.append("Use SEO/content to build steady demand.")
    else:
        bullets.append("Weak signals: avoid major investment until trend improves.")
        bullets.append("If necessary, target niche sub-segments only.")
        bullets.append("Re-check trend after 2–4 weeks for changes.")

    # Region hint
    if (region_label or "").strip().lower() == "india":
        bullets.append("India focus: prioritize metro + tier-1 cities for faster adoption.")
    else:
        bullets.append("Global focus: analyze top regions before scaling spend.")

    # Forecast-based adjustments
    if forecast.direction == "Up":
        bullets.append("Demand is trending upward: build inventory and marketing pipeline.")
    elif forecast.direction == "Down":
        cautions.append("Trend is declining: avoid long-term inventory risk.")
    else:
        bullets.append("Trend is stable: grow steadily, avoid overspending.")

    # Data-quality cautions
    if not metrics_ok:
        cautions.append("Limited trend points: growth estimate may be unstable.")
    if not news_ok:
        cautions.append("Few matched headlines: sentiment may be unreliable.")
    if float(getattr(news_q, "match_quality", 0.0) or 0.0) < DATA_QUALITY_CFG.match_quality_med:
        cautions.append("Low headline match quality: keyword may be ambiguous.")

    # Volatility caution
    if float(getattr(metrics, "volatility", 0.0) or 0.0) >= 25:
        cautions.append("High volatility: trend may be spike-driven or seasonal.")

    # Non-technical explainability fields
    key_reasons = _build_key_reasons(
        score_value=score_value,
        metrics=metrics,
        news_q=news_q,
        forecast=forecast,
        data_low=data_low,
    )
    biggest_risk = _biggest_risk(metrics=metrics, news_q=news_q, forecast=forecast, data_low=data_low)
    takeaway = _takeaway(decision, metrics)
    who_not = _who_should_not_enter(score_value, forecast, metrics)
    how_to_use_wisely = _how_to_use_wisely_text()

    return RecommendationPack(
        market_status=market_status,
        hype_label=hype_label,
        decision=decision,
        entry_window=entry_window,
        bullets=bullets,
        cautions=cautions,
        key_reasons=key_reasons,
        biggest_risk=biggest_risk,
        takeaway=takeaway,
        who_should_not_enter=who_not,
        how_to_use_wisely=how_to_use_wisely,
    )

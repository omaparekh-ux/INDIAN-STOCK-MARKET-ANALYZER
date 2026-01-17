"""
analysis/dashboard.py

Main orchestrator that converts:
- Google Trends + News + Sentiment + Forecast + Scores
into a single consumer-facing DashboardResponse.

This is the bridge between backend analytics and the "non-technical user" UI.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from services.trends import fetch_trends_single, fetch_related_queries
from services.news import fetch_news_gdelt
from nlp.sentiment import score_headlines_vader
from nlp.themes import extract_themes_from_headlines

from analysis.metrics import compute_trend_metrics, compute_news_quality
from analysis.forecast import compute_forecast
from analysis.advanced_metrics import compute_advanced_trend_metrics
from analysis.news_analytics import coverage_strength
from analysis.confidence import compute_confidence_score
from analysis.pro_scoring import compute_pro_score
from analysis.reco import build_recommendations

from analysis.assumptions import (
    get_core_assumptions,
    get_limitations,
    get_ethics_and_compliance_notes,
)
from analysis.user_guidance import (
    get_focus_areas,
    get_common_mistakes,
    get_who_should_not_enter,
    get_execution_difficulty,
    estimate_startup_budget,
)
from analysis.timeline_simulator import generate_timeline
from analysis.market_size import estimate_market_size
from analysis.explainability import (
    explain_verdict_drivers,
    explain_confidence,
    explain_biggest_risk,
)

from schemas.dashboard_schema import (
    DashboardResponse,
    Verdict,
    Confidence,
    DemandTrend,
    ConsumerInsights,
    Competition,
    Pricing,
    Innovation,
    Guidance,
    Transparency,
)


# -----------------------------
# Small helpers (simple + explainable)
# -----------------------------

def _sentiment_label(sent_0_100: float) -> str:
    if sent_0_100 >= 60:
        return "Positive"
    if sent_0_100 <= 40:
        return "Negative"
    return "Neutral"


def _growth_direction(growth_pct: float, forecast_direction: str) -> str:
    # Blend growth + forecast direction into a clean label
    if forecast_direction == "Down" or growth_pct <= -5:
        return "Falling"
    if forecast_direction == "Up" or growth_pct >= 5:
        return "Rising"
    return "Stable"


def _lifecycle_stage(growth_pct: float, volatility: float) -> str:
    # Very simple heuristics (good enough for consumer explanation)
    if growth_pct >= 15 and volatility <= 20:
        return "Growth"
    if growth_pct >= 5:
        return "Emerging"
    if growth_pct <= -5:
        return "Declining"
    return "Mature"


def _seasonality_summary(volatility: float, spike_label: str) -> str:
    if volatility >= 25 or "High" in (spike_label or ""):
        return "Seasonal/spike-driven pattern likely (watch timing)."
    if volatility >= 15:
        return "Some seasonality/spikes possible."
    return "Fairly stable year-round demand pattern."


def _detect_trust_risks(headlines: List[str]) -> List[str]:
    text = " ".join([h.lower() for h in headlines if isinstance(h, str)])
    risks: List[str] = []
    keywords = [
        ("fake", "Counterfeit / fake products"),
        ("adulter", "Adulteration risk"),
        ("scam", "Scams / misleading sellers"),
        ("complaint", "Customer complaints / quality issues"),
        ("side effect", "Safety or side-effect concerns"),
        ("recall", "Recalls / compliance concerns"),
    ]
    for k, label in keywords:
        if k in text:
            risks.append(label)
    return list(dict.fromkeys(risks))  # unique, preserve order


def _competition_from_signals(
    fetched_news: int,
    matched_news: int,
    related_rising_count: int,
    pro_score: float,
) -> Competition:
    """
    Heuristic competition assessment using *available* signals.
    (No paid market report assumptions.)
    """
    # intensity
    if fetched_news >= 45 or matched_news >= 25 or related_rising_count >= 8:
        intensity = "High"
    elif fetched_news >= 20 or matched_news >= 10:
        intensity = "Medium"
    else:
        intensity = "Low"

    # entry barrier (proxy)
    if intensity == "High" and pro_score < 55:
        entry_barrier = "High"
    elif intensity == "Medium":
        entry_barrier = "Medium"
    else:
        entry_barrier = "Low"

    # demand-supply gap (proxy)
    if pro_score >= 70 and intensity != "High":
        gap = "Underserved"
    elif intensity == "High" and pro_score < 60:
        gap = "Oversaturated"
    else:
        gap = "Balanced"

    notes = (
        "Competition is estimated from public interest signals (news volume + related queries), "
        "not from private sales/channel data."
    )

    return Competition(
        intensity=intensity,
        entry_barrier=entry_barrier,
        demand_supply_gap=gap,
        notes=notes,
    )


def _pricing_from_signals(competition_intensity: str, volatility: float) -> Pricing:
    if competition_intensity == "High":
        pressure = "Aggressive"
        margin = "Tight"
        band = "Varies widely (heavy discounting likely)"
    elif competition_intensity == "Medium":
        pressure = "Stable"
        margin = "Good"
        band = "Mid-range price bands are common"
    else:
        pressure = "Stable"
        margin = "Good"
        band = "Premium positioning is possible"

    if volatility >= 25 and competition_intensity != "Low":
        pressure = "Increasing"
        margin = "Risky"

    return Pricing(
        typical_price_band=band,
        margin_feasibility=margin,
        price_pressure=pressure,
    )


def _innovation_from_themes(themes: List[str], sentiment_0_100: float) -> Innovation:
    trending = themes[:6] if themes else []
    fading: List[str] = []
    fatigue = "Low"
    if sentiment_0_100 < 45 and len(trending) >= 4:
        fatigue = "Medium"
    if sentiment_0_100 < 40 and len(trending) >= 5:
        fatigue = "High"
    return Innovation(
        trending_features=trending,
        fading_features=fading,
        innovation_fatigue=fatigue,
    )


# -----------------------------
# Public API
# -----------------------------

def build_dashboard(
    product_type: str,
    *,
    timeframe_label: str,
    region_label: str,
    news_window_days: int,
    context_label: str,
    use_cache: bool = True,
    sentiment_bucket: str = "weekly",  # reserved for UI; dashboard currently uses overall sentiment
) -> DashboardResponse:
    """
    Orchestrates everything and returns a complete DashboardResponse.
    """

    # 1) Fetch data
    tdata = fetch_trends_single(
        product_type,
        timeframe_label=timeframe_label,
        region_label=region_label,
        use_cache=use_cache,
    )
    ts = tdata.get("timeseries", [])

    ndata = fetch_news_gdelt(
        product_type,
        context_label=context_label,
        days=news_window_days,
        max_records=60,
        use_cache=use_cache,
    )
    headlines = ndata.get("headlines", []) or []

    # Related queries (nice for consumer + differentiation ideas)
    rdata = fetch_related_queries(product_type, region_label=region_label, use_cache=use_cache)
    rising = rdata.get("rising", []) or []
    rising_count = len(rising)

    # Sentiment
    sdata = score_headlines_vader(product_type, headlines, use_cache=use_cache)
    sentiment_0_100 = float(sdata.get("sentiment_0_100", 50.0))

    # 2) Compute analytics
    metrics = compute_trend_metrics(ts)
    news_q = compute_news_quality(ndata.get("fetched_count", 0), ndata.get("matched_count", 0))
    adv = compute_advanced_trend_metrics(ts)

    cov = coverage_strength(
        fetched_count=int(ndata.get("fetched_count", 0)),
        matched_count=int(ndata.get("matched_count", 0)),
    )

    forecast = compute_forecast(
        ts,
        volatility=metrics.volatility,
        match_quality=news_q.match_quality,
        trend_points=metrics.points,
        matched_headlines=news_q.matched_count,
    )

    pro = compute_pro_score(
        growth_pct=metrics.growth_pct,
        momentum_pct=adv.momentum_pct,
        sentiment_0_100=sentiment_0_100,
        consistency_score=adv.consistency_score,
        volatility=metrics.volatility,
    )

    conf = compute_confidence_score(
        trend_points=metrics.points,
        matched_headlines=news_q.matched_count,
        match_quality=news_q.match_quality,
        volatility=metrics.volatility,
        growth_pct=metrics.growth_pct,
        momentum_pct=adv.momentum_pct,
        slope=float(getattr(forecast, "slope", 0.0) or 0.0),
    )

    rec = build_recommendations(
        keyword=product_type,
        metrics=metrics,
        news_q=news_q,
        score=pro,
        forecast=forecast,
        region_label=region_label,
    )

    # 3) Themes / consumer insights
    themes = extract_themes_from_headlines(headlines, max_themes=6)
    trust_risks = _detect_trust_risks(headlines)

    competition = _competition_from_signals(
        fetched_news=int(ndata.get("fetched_count", 0)),
        matched_news=int(ndata.get("matched_count", 0)),
        related_rising_count=rising_count,
        pro_score=float(pro.pro_score),
    )

    pricing = _pricing_from_signals(competition.intensity, metrics.volatility)
    innovation = _innovation_from_themes(themes, sentiment_0_100)

    demand = DemandTrend(
        direction=_growth_direction(metrics.growth_pct, forecast.direction),
        growth_rate_pct=float(metrics.growth_pct),
        lifecycle_stage=_lifecycle_stage(metrics.growth_pct, metrics.volatility),
        seasonality_summary=_seasonality_summary(metrics.volatility, adv.spike_label),
    )

    consumer = ConsumerInsights(
        sentiment_summary=_sentiment_label(sentiment_0_100),
        top_pain_points=themes[:5] if themes else ["Not enough clear themes from news data."],
        trust_risks=trust_risks if trust_risks else ["No major trust-risk keywords detected in recent headlines."],
    )

    # 4) Explainability (why this verdict)
    # Build a small metrics dict for explainability engine (keep normalized/simple)
    explain_metrics: Dict[str, float] = {
        "growth_rate": float(metrics.growth_pct),
        "sentiment_score": float(sentiment_0_100) / 100.0,  # 0..1
        "competition_intensity": 0.8 if competition.intensity == "High" else (0.55 if competition.intensity == "Medium" else 0.3),
        "price_pressure": 0.8 if pricing.price_pressure in {"Aggressive", "Increasing"} else (0.55 if pricing.price_pressure == "Stable" else 0.3),
        "trust_risk": 0.7 if (trust_risks and "No major" not in trust_risks[0]) else 0.2,
        # demand momentum: map momentum_pct to -1..1 rough
        "demand_momentum": max(-1.0, min(1.0, float(adv.momentum_pct) / 40.0)),
    }

    positives, negatives = explain_verdict_drivers(explain_metrics, rec.decision)

    data_coverage = {
        "demand": "strong" if metrics.min_data_ok else "weak",
        "news": "strong" if news_q.min_data_ok else "weak",
        "sentiment": "strong" if news_q.match_quality >= 0.35 else "weak",
    }

    conf_label = str(getattr(conf, "label", "Medium"))
    conf_value = float(getattr(conf, "confidence", 50.0))

    confidence = Confidence(
        level=conf_label,
        explanation=explain_confidence(data_coverage),
        data_coverage=data_coverage,
    )

    verdict = Verdict(
        decision=str(rec.decision),
        confidence_level=conf_label,
        key_reasons=(positives[:3] if positives else ["Signals are mixed; proceed carefully."]),
        biggest_risk=explain_biggest_risk(explain_metrics),
        takeaway=(
            "Start small and validate quickly before investing heavily."
            if "TEST" in rec.decision or "HOLD" in rec.decision or "WAIT" in rec.decision
            else "Execution speed + differentiation will decide your outcome."
        ),
    )

    # 5) Market size + guidance + timeline + transparency
    market_size = estimate_market_size(
        product_type=product_type,
        geo="IN" if region_label == "India" else "GLOBAL",
        demand_signal={"growth_rate_pct": float(metrics.growth_pct)},
        price_signal=None,
    )

    exec_difficulty = get_execution_difficulty(competition.intensity, trust_risks if trust_risks else [])
    guidance = Guidance(
        focus_areas=get_focus_areas(competition.intensity, pricing.price_pressure, trust_risks),
        common_mistakes=get_common_mistakes(product_type),
        who_should_not_enter=get_who_should_not_enter(competition.intensity, competition.entry_barrier),
        execution_difficulty=exec_difficulty,
        estimated_startup_budget=estimate_startup_budget(competition.intensity, exec_difficulty),
    )

    timeline = generate_timeline(competition.intensity, exec_difficulty)

    assumptions = get_core_assumptions(product_type) + get_ethics_and_compliance_notes(product_type)
    limitations = get_limitations(product_type)

    transparency = Transparency(
        what_changed_recently=(
            f"Trend direction: {forecast.direction}. Momentum: {adv.momentum_pct:.2f}%."
        ),
        assumptions=assumptions,
        limitations=limitations,
        how_to_use_wisely="Use this tool to decide: Enter vs Test vs Wait. Then validate with small experiments (ads, sampling, pilot sales).",
    )

    # 6) Raw metrics (for debugging + optional UI)
    raw_metrics: Dict[str, float] = {
        "pro_score": float(pro.pro_score),
        "confidence": float(conf_value),
        "growth_pct": float(metrics.growth_pct),
        "momentum_pct": float(adv.momentum_pct),
        "sentiment_0_100": float(sentiment_0_100),
        "volatility": float(metrics.volatility),
        "consistency_score": float(adv.consistency_score),
        "news_match_quality": float(news_q.match_quality),
        "news_fetched": float(ndata.get("fetched_count", 0) or 0),
        "news_matched": float(ndata.get("matched_count", 0) or 0),
        "related_rising_count": float(rising_count),
    }

    return DashboardResponse(
        product_type=product_type,
        verdict=verdict,
        confidence=confidence,
        market_size=market_size,
        demand_trend=demand,
        consumer_insights=consumer,
        competition=competition,
        pricing=pricing,
        innovation=innovation,
        guidance=guidance,
        timeline=timeline,
        transparency=transparency,
        raw_metrics=raw_metrics,
    )

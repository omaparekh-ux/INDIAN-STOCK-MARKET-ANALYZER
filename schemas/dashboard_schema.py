"""
Unified dashboard output schema for the AI Market Trend Analyzer.

This schema defines the complete, non-technical, consumer-facing response
produced by the system after a user enters a product type.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional


# -----------------------------
# Core verdict & confidence
# -----------------------------

@dataclass
class Verdict:
    decision: str                 # ENTER / TEST / WAIT / AVOID
    confidence_level: str          # High / Medium / Low
    key_reasons: List[str]         # 2–3 plain-English reasons
    biggest_risk: str              # Single most important risk
    takeaway: str                  # “If you remember one thing…”


@dataclass
class Confidence:
    level: str                     # High / Medium / Low
    explanation: str               # Why this confidence level
    data_coverage: Dict[str, str]  # e.g. {"demand": "strong", "sentiment": "medium"}


# -----------------------------
# Market size & demand
# -----------------------------

@dataclass
class MarketSize:
    estimated_annual_value: str    # e.g. "₹12,000–15,000 Cr"
    realistic_new_brand_range: str # e.g. "₹2–5 Cr/year"
    explanation: str               # Plain-language explanation
    assumptions: List[str]


@dataclass
class DemandTrend:
    direction: str                 # Rising / Stable / Falling
    growth_rate_pct: Optional[float]
    lifecycle_stage: str           # Emerging / Growth / Mature / Declining
    seasonality_summary: str       # e.g. "Peaks in summer months"


# -----------------------------
# Consumer & competition
# -----------------------------

@dataclass
class ConsumerInsights:
    sentiment_summary: str         # Positive / Neutral / Negative
    top_pain_points: List[str]
    trust_risks: List[str]         # e.g. fake products, overclaiming


@dataclass
class Competition:
    intensity: str                 # Low / Medium / High
    entry_barrier: str             # Low / Medium / High
    demand_supply_gap: str         # Underserved / Balanced / Oversaturated
    notes: str


# -----------------------------
# Pricing & innovation
# -----------------------------

@dataclass
class Pricing:
    typical_price_band: str        # e.g. "₹199–₹499"
    margin_feasibility: str        # Good / Tight / Risky
    price_pressure: str            # Increasing / Stable / Aggressive


@dataclass
class Innovation:
    trending_features: List[str]
    fading_features: List[str]
    innovation_fatigue: str        # Low / Medium / High


# -----------------------------
# Guidance & execution
# -----------------------------

@dataclass
class Guidance:
    focus_areas: List[str]          # What the user should focus on
    common_mistakes: List[str]
    who_should_not_enter: List[str]
    execution_difficulty: str       # Easy / Moderate / Hard
    estimated_startup_budget: str   # e.g. "₹10–25 Lakhs"


@dataclass
class TimelinePhase:
    phase: str                      # Setup / Traction / Scale
    duration: str                   # e.g. "0–2 months"
    expectations: str


@dataclass
class Timeline:
    phases: List[TimelinePhase]


# -----------------------------
# Transparency
# -----------------------------

@dataclass
class Transparency:
    what_changed_recently: str
    assumptions: List[str]
    limitations: List[str]
    how_to_use_wisely: str


# -----------------------------
# Final dashboard response
# -----------------------------

@dataclass
class DashboardResponse:
    product_type: str
    verdict: Verdict
    confidence: Confidence
    market_size: MarketSize
    demand_trend: DemandTrend
    consumer_insights: ConsumerInsights
    competition: Competition
    pricing: Pricing
    innovation: Innovation
    guidance: Guidance
    timeline: Timeline
    transparency: Transparency
    raw_metrics: Dict[str, float] = field(default_factory=dict)  # optional, for debugging

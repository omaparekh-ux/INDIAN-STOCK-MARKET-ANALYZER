# config.py
# Central configuration for the AI Market Trend Analyzer (locked v1.0 + Phase 2 stability upgrades)
# Updated: adds cache namespace TTLs + global throttle + aligns timeframe mapping with services/trends.py logic.

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


# -----------------------------
# App-wide constants / defaults
# -----------------------------

APP_TITLE: str = "AI Market Trend Analysis"
APP_VERSION: str = "v1.0"

# Only 2 regions (locked)
REGION_OPTIONS: List[str] = ["Worldwide", "India"]
REGION_TO_GEO: Dict[str, str] = {
    "Worldwide": "",   # PyTrends uses empty geo for global
    "India": "IN",
}

# Timeframes we will support (locked)
TIMEFRAME_OPTIONS: List[str] = ["6 months", "12 months"]

# NOTE:
# services/trends.py approximates "6 months" by fetching "today 12-m" and keeping last ~26 weeks.
# So we keep this mapping for display/compatibility, but it is not relied upon by trends.py.
TIMEFRAME_TO_TRENDS_TF: Dict[str, str] = {
    "6 months": "today 12-m",   # fetched then truncated to ~26 points inside services/trends.py
    "12 months": "today 12-m",
}

# News lookback window options (locked)
NEWS_WINDOW_OPTIONS: List[str] = ["30 days", "60 days"]
NEWS_WINDOW_TO_DAYS: Dict[str, int] = {"30 days": 30, "60 days": 60}

# Category/Context options (locked)
CONTEXT_OPTIONS: List[str] = ["General", "Consumer", "Tech", "Health", "Finance"]


# -----------------------------
# Context expansion for news
# -----------------------------
# This reduces irrelevant results (e.g., "soap opera" for "soap")
# We build a query like: keyword AND (term1 OR term2 OR term3)

CONTEXT_TERMS: Dict[str, List[str]] = {
    "General": ["market", "trend", "demand", "industry", "growth"],
    "Consumer": ["product", "brand", "retail", "consumer", "personal care", "hygiene"],
    "Tech": ["AI", "startup", "software", "innovation", "platform", "technology"],
    "Health": ["wellness", "health", "dermatology", "clinic", "safety", "ingredients"],
    "Finance": ["investment", "revenue", "profit", "funding", "valuation", "stocks"],
}

# Extra keyword-specific hints (very small, but high impact)
# These are used ONLY when keyword is very broad/ambiguous.
KEYWORD_SUGGESTIONS: Dict[str, List[str]] = {
    "soap": ["hand wash soap", "bath soap", "body wash", "handwash"],
    "shampoo": ["anti dandruff shampoo", "herbal shampoo", "sulfate free shampoo"],
    "brush": ["hair brush", "toothbrush", "paint brush"],
}


# -----------------------------
# Scoring / classification rules
# -----------------------------

# Weighting (locked)
WEIGHT_GROWTH: float = 0.6
WEIGHT_SENTIMENT: float = 0.4

# Growth clamping to stabilize scoring (locked)
GROWTH_CLAMP_MIN: float = -50.0
GROWTH_CLAMP_MAX: float = 150.0

# Trend Strength Score bands (locked)
SCORE_STRONG_MIN: float = 70.0
SCORE_MODERATE_MIN: float = 40.0

# Forecast direction thresholds using regression slope (simple, explainable)
FORECAST_SLOPE_FLAT_EPS: float = 0.03


# -----------------------------
# Minimum data rules (locked)
# -----------------------------

MIN_TREND_POINTS: int = 20          # If fewer, warn about trend confidence
MIN_MATCHED_HEADLINES: int = 8      # If fewer, warn about sentiment confidence

# Headline match quality thresholds (matched/fetched)
MATCH_QUALITY_HIGH: float = 0.60
MATCH_QUALITY_MED: float = 0.35


# -----------------------------
# Caching rules (Phase 2 stability upgrades)
# -----------------------------

CACHE_DIR: str = "data/cache"

# Default TTL: 30 minutes (kept)
CACHE_TTL_SECONDS: int = 30 * 60

# Safety delay between network calls (used by cooldown_sleep)
NETWORK_COOLDOWN_SECONDS: float = 1.0

# NEW: cross-process throttle to reduce 429s (used by utils.cache.global_throttle)
GLOBAL_THROTTLE_SECONDS: float = 1.2

# NEW: per-namespace TTL overrides (used by utils.cache._ttl_for_namespace)
# These are conservative and help reduce repeated calls while keeping data reasonably fresh.
CACHE_TTL_BY_NAMESPACE: Dict[str, int] = {
    # Google Trends is rate-limited; keep longer TTL
    "trends_single": 6 * 60 * 60,           # 6 hours
    "trends_compare": 6 * 60 * 60,          # 6 hours
    "trends_related_queries": 12 * 60 * 60, # 12 hours

    # News & sentiment change faster; keep shorter TTL
    "news": 30 * 60,                        # 30 minutes
    "sentiment": 30 * 60,                   # 30 minutes

    # Optional: any other namespaces default to CACHE_TTL_SECONDS
}


# -----------------------------
# Small dataclasses for typed config
# -----------------------------

@dataclass(frozen=True)
class TrendScoreConfig:
    weight_growth: float = WEIGHT_GROWTH
    weight_sentiment: float = WEIGHT_SENTIMENT
    growth_clamp_min: float = GROWTH_CLAMP_MIN
    growth_clamp_max: float = GROWTH_CLAMP_MAX
    score_strong_min: float = SCORE_STRONG_MIN
    score_moderate_min: float = SCORE_MODERATE_MIN


@dataclass(frozen=True)
class DataQualityConfig:
    min_trend_points: int = MIN_TREND_POINTS
    min_matched_headlines: int = MIN_MATCHED_HEADLINES
    match_quality_high: float = MATCH_QUALITY_HIGH
    match_quality_med: float = MATCH_QUALITY_MED


@dataclass(frozen=True)
class CacheConfig:
    cache_dir: str = CACHE_DIR
    ttl_seconds: int = CACHE_TTL_SECONDS
    cooldown_seconds: float = NETWORK_COOLDOWN_SECONDS

    # NEW fields used by updated utils/cache.py
    global_throttle_seconds: float = GLOBAL_THROTTLE_SECONDS
    ttl_by_namespace: Dict[str, int] = None  # set in __post_init__-like style below


# Convenient bundles used by later modules
TREND_SCORE_CFG = TrendScoreConfig()
DATA_QUALITY_CFG = DataQualityConfig()

# Build CACHE_CFG with namespace TTLs
CACHE_CFG = CacheConfig(ttl_by_namespace=CACHE_TTL_BY_NAMESPACE)

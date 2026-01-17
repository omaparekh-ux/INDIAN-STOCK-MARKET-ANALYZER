# utils/helpers.py
# Common helper utilities: cleaning, validation, mapping, safe math.

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import List, Optional, Tuple

from config import (
    CONTEXT_TERMS,
    KEYWORD_SUGGESTIONS,
    NEWS_WINDOW_TO_DAYS,
    REGION_TO_GEO,
    TIMEFRAME_TO_TRENDS_TF,
)


# -----------------------------
# Text cleaning / validation
# -----------------------------

_WS_RE = re.compile(r"\s+")
_NONPRINT_RE = re.compile(r"[^\x20-\x7E]+")


def clean_keyword(raw: str) -> str:
    """
    Cleans a user keyword for safer querying:
    - trims
    - collapses whitespace
    - removes non-printable characters
    """
    if raw is None:
        return ""
    s = raw.strip()
    s = _NONPRINT_RE.sub("", s)
    s = _WS_RE.sub(" ", s)
    return s


def validate_keywords(keywords: List[str], max_keywords: int = 3) -> Tuple[List[str], List[str]]:
    """
    Returns (valid_keywords, errors).
    - Cleans all inputs
    - Removes empties
    - Enforces max count
    """
    cleaned = [clean_keyword(k) for k in keywords]
    cleaned = [k for k in cleaned if k]

    errors: List[str] = []
    if len(cleaned) == 0:
        errors.append("Please enter at least one keyword.")
        return [], errors

    if len(cleaned) > max_keywords:
        errors.append(f"Please enter at most {max_keywords} keywords.")
        cleaned = cleaned[:max_keywords]

    # Deduplicate while preserving order
    seen = set()
    deduped: List[str] = []
    for k in cleaned:
        kl = k.lower()
        if kl not in seen:
            seen.add(kl)
            deduped.append(k)

    return deduped, errors


def keyword_suggestions(keyword: str) -> List[str]:
    """
    Returns suggestion strings for broad/ambiguous keywords.
    This is intentionally small & curated (viva-friendly).
    """
    k = clean_keyword(keyword).lower()
    return KEYWORD_SUGGESTIONS.get(k, [])


# -----------------------------
# UI option mapping helpers
# -----------------------------

def region_to_geo(region_label: str) -> str:
    """
    Maps UI region label -> PyTrends geo code.
    Worldwide -> ""
    India -> "IN"
    """
    return REGION_TO_GEO.get(region_label, "")


def timeframe_to_trends_tf(timeframe_label: str) -> str:
    """
    Maps UI timeframe label -> PyTrends timeframe string.
    """
    return TIMEFRAME_TO_TRENDS_TF.get(timeframe_label, "today 12-m")


def news_window_to_days(news_window_label: str) -> int:
    """
    Maps UI news window label -> integer days.
    """
    return NEWS_WINDOW_TO_DAYS.get(news_window_label, 30)


def context_terms(context_label: str) -> List[str]:
    """
    Returns context terms for query expansion.
    """
    return CONTEXT_TERMS.get(context_label, CONTEXT_TERMS["General"])


# -----------------------------
# Safe math utilities
# -----------------------------

def safe_div(n: float, d: float, default: float = 0.0) -> float:
    """
    Safe division avoiding ZeroDivisionError.
    """
    try:
        if d == 0:
            return default
        return n / d
    except Exception:
        return default


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def normalize_to_0_100(x: float, lo: float, hi: float) -> float:
    """
    Linearly maps x in [lo, hi] to [0, 100], clamping outside range.
    """
    if hi == lo:
        return 50.0
    x2 = clamp(x, lo, hi)
    return (x2 - lo) * 100.0 / (hi - lo)


# -----------------------------
# Date/time utilities
# -----------------------------

def utc_now_iso() -> str:
    """
    ISO string timestamp for display/logging.
    """
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


# -----------------------------
# Simple keyword matching helpers
# -----------------------------

def normalize_for_match(text: str) -> str:
    """
    Lowercase + collapse whitespace for matching.
    """
    t = (text or "").lower()
    t = _WS_RE.sub(" ", t)
    return t.strip()


def is_keyword_match(headline: str, keyword: str) -> bool:
    """
    Simple, explainable match rule:
    headline contains keyword as a substring (case-insensitive),
    after normalization.
    """
    h = normalize_for_match(headline)
    k = normalize_for_match(keyword)
    if not h or not k:
        return False
    return k in h

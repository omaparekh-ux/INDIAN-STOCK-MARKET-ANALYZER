# nlp/sentiment.py
# VADER-based sentiment scoring for headlines (no dataset required).
# Updated: deterministic cache key, robust empty handling, stable output schema, aligned cache namespace.

from __future__ import annotations

import hashlib
from typing import Any, Dict, List

import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

from utils.cache import make_cache_key, read_cache, write_cache
from utils.helpers import clean_keyword, utc_now_iso


def _ensure_vader() -> None:
    """
    Ensure VADER lexicon is available.
    If not present, download once.
    """
    try:
        nltk.data.find("sentiment/vader_lexicon.zip")
    except LookupError:
        nltk.download("vader_lexicon", quiet=True)


def _analyzer() -> SentimentIntensityAnalyzer:
    _ensure_vader()
    return SentimentIntensityAnalyzer()


def _stable_titles_hash(headlines: List[Dict[str, Any]]) -> str:
    """
    Deterministic hash of input titles.
    Avoids Python's built-in hash() randomness across processes.
    """
    titles = [str(h.get("title", "")).strip() for h in (headlines or []) if str(h.get("title", "")).strip()]
    joined = "\n".join(titles).encode("utf-8")
    return hashlib.sha256(joined).hexdigest()


def score_headlines_vader(
    keyword: str,
    headlines: List[Dict[str, Any]],
    *,
    use_cache: bool = True,
) -> Dict[str, Any]:
    """
    Score each headline title using VADER.
    Input `headlines` is the matched headlines list returned by services/news.py.

    Output:
      {
        meta: {...},
        counts: {positive, neutral, negative, total},
        avg_compound: float,
        sentiment_0_100: float,
        scored: [{title, compound, label, url, source, date}, ...]
      }
    """
    kw = clean_keyword(keyword)

    # Empty-input deterministic return (prevents downstream crashes)
    if not headlines:
        return {
            "meta": {
                "keyword": kw,
                "method": "VADER",
                "fetched_at": utc_now_iso(),
                "cached": False,
                "error": None,
            },
            "counts": {"positive": 0, "neutral": 0, "negative": 0, "total": 0},
            "avg_compound": 0.0,
            "sentiment_0_100": 50.0,
            "scored": [],
        }

    titles_hash = _stable_titles_hash(headlines)

    cache_key = make_cache_key(
        "sentiment",  # IMPORTANT: aligns with config CACHE_TTL_BY_NAMESPACE
        keyword=kw,
        extra={"method": "vader", "titles_sha256": titles_hash},
    )

    if use_cache:
        cached = read_cache(cache_key)
        if cached is not None:
            cached_meta = cached.get("meta", {}) or {}
            cached_meta["cached"] = True
            cached["meta"] = cached_meta
            return cached

    sia = _analyzer()

    positive = 0
    neutral = 0
    negative = 0
    compounds: List[float] = []
    scored: List[Dict[str, Any]] = []

    for h in headlines or []:
        title = str(h.get("title", "")).strip()
        if not title:
            continue

        scores = sia.polarity_scores(title)
        compound = float(scores.get("compound", 0.0))
        compounds.append(compound)

        # Simple, explainable labeling
        if compound >= 0.05:
            label = "positive"
            positive += 1
        elif compound <= -0.05:
            label = "negative"
            negative += 1
        else:
            label = "neutral"
            neutral += 1

        scored.append(
            {
                "title": title,
                "compound": round(compound, 4),
                "label": label,
                "url": str(h.get("url", "")),
                "source": str(h.get("source", "")),
                "date": str(h.get("date", "")),
            }
        )

    n = len(compounds)
    avg_compound = (sum(compounds) / n) if n > 0 else 0.0

    # Convert avg_compound (-1..+1) to 0..100 (simple linear map)
    sentiment_0_100 = (avg_compound + 1.0) * 50.0

    # Clamp for safety
    if sentiment_0_100 < 0.0:
        sentiment_0_100 = 0.0
    elif sentiment_0_100 > 100.0:
        sentiment_0_100 = 100.0

    result = {
        "meta": {
            "keyword": kw,
            "method": "VADER",
            "fetched_at": utc_now_iso(),
            "cached": False,
            "error": None,
        },
        "counts": {
            "positive": int(positive),
            "neutral": int(neutral),
            "negative": int(negative),
            "total": int(positive + neutral + negative),
        },
        "avg_compound": round(float(avg_compound), 4),
        "sentiment_0_100": round(float(sentiment_0_100), 2),
        "scored": scored,
    }

    if use_cache:
        write_cache(cache_key, result)

    return result

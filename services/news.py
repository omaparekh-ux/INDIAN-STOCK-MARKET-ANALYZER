# services/news.py
# GDELT news fetching + context query expansion + filtering + match quality.
# Updated: retries/backoff + safe cached fallback + global throttle to reduce rate-limit failures.

from __future__ import annotations

import random
import time
from typing import Any, Dict, List, Optional

import requests

from utils.cache import make_cache_key, read_cache, write_cache, global_throttle
from utils.helpers import clean_keyword, context_terms, is_keyword_match, utc_now_iso


GDELT_DOC_API = "https://api.gdeltproject.org/api/v2/doc/doc"

DEFAULT_TIMEOUT = 20
DEFAULT_RETRIES = 4
DEFAULT_MAX_SLEEP = 12.0


def build_news_query(keyword: str, context_label: str) -> str:
    """
    Builds an explainable GDELT query:
      keyword AND (term1 OR term2 OR term3)

    We keep it simple and viva-friendly.
    """
    kw = clean_keyword(keyword)
    terms = context_terms(context_label)

    # Quote the keyword to reduce weird matches
    kw_part = f'"{kw}"'

    # Limit terms to avoid overly-long queries
    terms = (terms or [])[:6]
    or_block = " OR ".join([f'"{t}"' for t in terms]) if terms else ""

    # Example: "soap" AND ("product" OR "brand" OR "hygiene")
    if or_block:
        return f"{kw_part} AND ({or_block})"
    return kw_part


def fetch_news_gdelt(
    keyword: str,
    *,
    context_label: str,
    days: int,
    max_records: int = 50,
    use_cache: bool = True,
) -> Dict[str, Any]:
    """
    Fetch news headlines from GDELT Doc API.

    Returns:
      {
        meta: {...},
        fetched_count: Y,
        matched_count: X,
        match_quality: float,
        headlines: [ {title, url, source, date, snippet} ... ],  # matched only
      }
    """
    kw = clean_keyword(keyword)
    query = build_news_query(kw, context_label)

    cache_key = make_cache_key(
        "news",  # IMPORTANT: aligns with config CACHE_TTL_BY_NAMESPACE
        keyword=kw,
        context=context_label,
        news_days=int(days),
        extra={"query": query, "max_records": int(max_records)},
    )

    if use_cache:
        cached = read_cache(cache_key)
        if cached is not None:
            cached_meta = cached.get("meta", {}) or {}
            cached_meta["cached"] = True
            cached["meta"] = cached_meta
            return cached

    meta = {
        "keyword": kw,
        "context": context_label,
        "days": int(days),
        "query": query,
        "fetched_at": utc_now_iso(),
        "cached": False,
        "error": None,
    }

    params = {
        "query": query,
        "mode": "ArtList",
        "format": "json",
        "maxrecords": int(max_records),
        "sort": "HybridRel",        # relevance + recency
        "timelinesmooth": 0,
        "formatdatetime": "true",
        "querymode": "ArtList",
        "startdatetime": _startdatetime_utc(int(days)),
    }

    def _do_request() -> Dict[str, Any]:
        # Global throttle reduces burst calls (especially compare mode)
        global_throttle()

        resp = requests.get(GDELT_DOC_API, params=params, timeout=DEFAULT_TIMEOUT)
        resp.raise_for_status()
        return resp.json()

    data: Optional[Dict[str, Any]] = None
    last_err: Optional[Exception] = None

    for attempt in range(DEFAULT_RETRIES):
        try:
            data = _do_request()
            break
        except Exception as e:
            last_err = e
            # backoff + jitter
            base = min(DEFAULT_MAX_SLEEP, (1.0 * (2 ** attempt)))
            jitter = random.random() * 0.8
            time.sleep(min(DEFAULT_MAX_SLEEP, base + jitter))

    # If request failed, serve stale cache fallback if possible
    if data is None:
        if use_cache:
            cached = read_cache(cache_key)
            if cached is not None:
                cached_meta = cached.get("meta", {}) or {}
                cached_meta["cached"] = True
                cached_meta["stale_fallback"] = True
                cached_meta["error"] = f"GDELT fetch failed (served cached): {type(last_err).__name__}"
                cached["meta"] = cached_meta
                return cached

        meta["error"] = f"GDELT fetch failed: {type(last_err).__name__}"
        return {
            "meta": meta,
            "fetched_count": 0,
            "matched_count": 0,
            "match_quality": 0.0,
            "headlines": [],
        }

    articles = (data.get("articles") or []) if isinstance(data, dict) else []
    fetched = _parse_articles(articles)

    # Filter for keyword match (title-based, simple and explainable)
    matched = [a for a in fetched if is_keyword_match(a.get("title", ""), kw)]

    fetched_count = len(fetched)
    matched_count = len(matched)
    match_quality = (matched_count / fetched_count) if fetched_count > 0 else 0.0

    result = {
        "meta": meta,
        "fetched_count": fetched_count,
        "matched_count": matched_count,
        "match_quality": round(float(match_quality), 4),
        "headlines": matched[: int(max_records)],
    }

    if use_cache:
        write_cache(cache_key, result)

    return result


def _startdatetime_utc(days: int) -> str:
    """
    GDELT expects UTC datetime in YYYYMMDDHHMMSS.
    We set startdatetime to now - days.
    """
    from datetime import datetime, timedelta, timezone

    dt = datetime.now(timezone.utc) - timedelta(days=int(days))
    return dt.strftime("%Y%m%d%H%M%S")


def _parse_articles(articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Extract minimal fields from GDELT articles list.
    Keep small + explainable.
    """
    out: List[Dict[str, Any]] = []
    for a in articles or []:
        title = a.get("title") or ""
        url = a.get("url") or ""
        source = a.get("sourceCountry") or a.get("sourceCollection") or a.get("source") or ""
        dt = a.get("seendate") or a.get("datetime") or ""
        snippet = a.get("snippet") or a.get("description") or ""

        out.append(
            {
                "title": str(title),
                "url": str(url),
                "source": str(source),
                "date": str(dt),
                "snippet": str(snippet),
            }
        )
    return out

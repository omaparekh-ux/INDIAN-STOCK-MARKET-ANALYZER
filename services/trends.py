# services/trends.py
# Google Trends (PyTrends) service: single + compare + related queries.
# Robust anti-429 + fallback protections:
# - Single shared TrendReq session
# - Global throttling (max requests/min) + jitter pacing
# - Circuit breaker: on 429, stop hitting Trends for cooldown window
# - Safe requests_args: ONLY headers/proxies (NO timeout duplication)
# - Fallback order when blocked/fails: cache -> demo dataset (data/demo_trends.json)
#
# Additional hardening (2026-01):
# - cache freshness-aware reads (avoid repeated Trends calls on reruns)
# - compare cache keys stabilized (order-insensitive, normalized keywords)
# - stronger safe-mode behavior via env TRENDS_FORCE_SAFE_MODE=1
# - circuit breaker tracks fail_count and can open after repeated failures
# - meta flags standardized for UI pills: cached/demo/rate_limited/circuit_open/stale_fallback

from __future__ import annotations

import json
import os
import random
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd
from pytrends.request import TrendReq

from utils.cache import make_cache_key, read_cache, write_cache, cooldown_sleep
from utils.helpers import clean_keyword, region_to_geo, utc_now_iso


# -----------------------------
# Tunables (safe defaults)
# -----------------------------
DEFAULT_RETRIES = 3  # reduce repeated hits (safer for 429)
DEFAULT_BASE_SLEEP = 1.0
DEFAULT_MAX_SLEEP = 12.0

# Cache TTL (used by cache freshness-aware reads; actual persistence is in utils/cache.py)
TRENDS_CACHE_TTL_S = int(os.getenv("TRENDS_CACHE_TTL_S", "21600"))  # 6 hours default

# Throttling / pacing
MAX_REQ_PER_MIN = int(os.getenv("TRENDS_MAX_REQ_PER_MIN", "20"))  # global cap
THROTTLE_BASE_S = float(os.getenv("TRENDS_THROTTLE_BASE_S", "0.35"))
THROTTLE_JITTER_S = float(os.getenv("TRENDS_THROTTLE_JITTER_S", "0.45"))

# Circuit breaker
CIRCUIT_COOLDOWN_S = int(os.getenv("TRENDS_CIRCUIT_COOLDOWN_S", "900"))  # 15 min
CIRCUIT_FAIL_THRESHOLD = int(os.getenv("TRENDS_CIRCUIT_FAIL_THRESHOLD", "2"))  # open after N fails
CIRCUIT_FILE = Path(".cache") / "trends_circuit.json"

# Demo fallback dataset
DEMO_PATH = Path(os.getenv("TRENDS_DEMO_PATH", "data/demo_trends.json"))

# Optional proxy support
ENV_PROXY = os.getenv("TRENDS_PROXY")

# Hard safe-mode: NEVER call pytrends
FORCE_SAFE_MODE = os.getenv("TRENDS_FORCE_SAFE_MODE", "").strip().lower() in {"1", "true", "yes"}


# -----------------------------
# Global shared client + guards
# -----------------------------
_client_lock = threading.Lock()
_shared_client: Optional[TrendReq] = None

_throttle_lock = threading.Lock()
_window_start_ts = 0.0
_window_count = 0


# -----------------------------
# Utilities
# -----------------------------
def _proxy_dict() -> Optional[Dict[str, str]]:
    if ENV_PROXY:
        return {"http": ENV_PROXY, "https": ENV_PROXY}

    http_p = os.getenv("HTTP_PROXY") or os.getenv("http_proxy")
    https_p = os.getenv("HTTPS_PROXY") or os.getenv("https_proxy")
    if http_p or https_p:
        out: Dict[str, str] = {}
        if http_p:
            out["http"] = http_p
        if https_p:
            out["https"] = https_p
        return out or None
    return None


def _safe_requests_args() -> Dict[str, Any]:
    """
    CRITICAL:
    Do NOT include 'timeout' in requests_args because pytrends internally
    passes timeout in some places -> duplicate timeout TypeError.
    Only pass headers/proxies here.
    """
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Connection": "keep-alive",
    }
    args: Dict[str, Any] = {"headers": headers}
    proxies = _proxy_dict()
    if proxies:
        args["proxies"] = proxies
    return args


def _get_shared_client() -> TrendReq:
    """
    Single shared TrendReq to reduce cookie churn & reduce repeated init calls.
    If something goes weird, we can recreate by setting _shared_client=None.
    """
    global _shared_client
    with _client_lock:
        if _shared_client is None:
            _shared_client = TrendReq(hl="en-US", tz=330, requests_args=_safe_requests_args())
        return _shared_client


def _throttle_acquire() -> None:
    """
    Global throttling: max N requests per minute + steady pacing with jitter.
    Prevents bursts that trigger 429.
    """
    global _window_start_ts, _window_count
    # steady pacing first (cheap, reduces bursts)
    time.sleep(max(0.0, THROTTLE_BASE_S + random.uniform(0.0, THROTTLE_JITTER_S)))

    # max-per-minute cap
    while True:
        with _throttle_lock:
            now = time.time()
            if _window_start_ts <= 0:
                _window_start_ts = now
                _window_count = 0

            # reset window
            if now - _window_start_ts >= 60.0:
                _window_start_ts = now
                _window_count = 0

            if _window_count < max(1, MAX_REQ_PER_MIN):
                _window_count += 1
                return

            wait_s = 60.0 - (now - _window_start_ts)

        # wait until the window rolls over (with tiny jitter)
        time.sleep(max(0.05, wait_s + random.uniform(0.0, 0.35)))


def _timeframe_to_pytrends_tf(timeframe_label: str) -> Tuple[str, Optional[int]]:
    """
    PyTrends does NOT have a native '6 months' timeframe string.
    So we approximate:
      - "6 months"  -> fetch "today 12-m" and keep last ~26 weeks
      - "12 months" -> fetch "today 12-m" and keep all
    Returns (pytrends_tf, keep_last_n_points or None)
    """
    if timeframe_label == "6 months":
        return "today 12-m", 26
    return "today 12-m", None


def _df_to_timeseries(df: pd.DataFrame, keyword: str, keep_last: Optional[int] = None) -> List[Dict[str, Any]]:
    if df is None or df.empty or keyword not in df.columns:
        return []

    series = df[keyword].copy()
    if keep_last is not None and keep_last > 0:
        series = series.tail(keep_last)

    out: List[Dict[str, Any]] = []
    for idx, val in series.items():
        out.append({"date": pd.to_datetime(idx).strftime("%Y-%m-%d"), "value": float(val) if val is not None else 0.0})
    return out


def _sleep_backoff(attempt: int) -> None:
    base = DEFAULT_BASE_SLEEP * (2 ** attempt)
    jitter = random.random() * 0.8
    s = min(DEFAULT_MAX_SLEEP, base + jitter)
    time.sleep(s)


def _is_429_error(err: Exception) -> bool:
    name = type(err).__name__.lower()
    msg = str(err).lower()
    return ("toomanyrequests" in name) or ("429" in msg) or ("too many requests" in msg)


def _normalize_keywords_for_cache(kws: List[str]) -> List[str]:
    """
    Make compare cache stable:
    - clean each keyword
    - lowercase for key normalization (display remains original elsewhere)
    - de-dupe preserving order
    - then sort for order-insensitive cache identity
    """
    cleaned = [clean_keyword(k).strip() for k in kws if clean_keyword(k).strip()]
    dedup: List[str] = []
    seen = set()
    for k in cleaned:
        kl = k.lower()
        if kl in seen:
            continue
        seen.add(kl)
        dedup.append(k)
    # order-insensitive cache key:
    return sorted(dedup, key=lambda x: x.lower())


# -----------------------------
# Circuit breaker (file-based)
# -----------------------------
def _load_circuit() -> Dict[str, Any]:
    try:
        CIRCUIT_FILE.parent.mkdir(exist_ok=True)
        if not CIRCUIT_FILE.exists():
            return {"open_until_ts": 0.0, "last_status": None, "last_error": None, "fail_count": 0}
        with CIRCUIT_FILE.open("r", encoding="utf-8") as f:
            d = json.load(f)
        if not isinstance(d, dict):
            return {"open_until_ts": 0.0, "last_status": None, "last_error": None, "fail_count": 0}
        d.setdefault("open_until_ts", 0.0)
        d.setdefault("last_status", None)
        d.setdefault("last_error", None)
        d.setdefault("fail_count", 0)
        return d
    except Exception:
        return {"open_until_ts": 0.0, "last_status": None, "last_error": None, "fail_count": 0}


def _save_circuit(state: Dict[str, Any]) -> None:
    try:
        CIRCUIT_FILE.parent.mkdir(exist_ok=True)
        tmp = CIRCUIT_FILE.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False)
        tmp.replace(CIRCUIT_FILE)
    except Exception:
        pass


def _circuit_is_open() -> Tuple[bool, float]:
    st = _load_circuit()
    open_until = float(st.get("open_until_ts", 0.0) or 0.0)
    now = time.time()
    return (now < open_until), max(0.0, open_until - now)


def _open_circuit(reason: str, *, is_rate_limit: bool) -> None:
    st = _load_circuit()
    st["open_until_ts"] = time.time() + max(60, int(CIRCUIT_COOLDOWN_S))
    st["last_status"] = "rate_limited" if is_rate_limit else "error"
    st["last_error"] = reason
    st["fail_count"] = int(st.get("fail_count", 0) or 0)
    _save_circuit(st)


def _record_failure(reason: str, *, is_rate_limit: bool) -> None:
    st = _load_circuit()
    st["fail_count"] = int(st.get("fail_count", 0) or 0) + 1
    st["last_error"] = reason
    st["last_status"] = "rate_limited" if is_rate_limit else "error"
    # Open circuit after threshold failures (or immediately on 429)
    if is_rate_limit or st["fail_count"] >= max(1, CIRCUIT_FAIL_THRESHOLD):
        st["open_until_ts"] = time.time() + max(60, int(CIRCUIT_COOLDOWN_S))
    _save_circuit(st)


def _record_success() -> None:
    st = _load_circuit()
    # only reset fail_count when circuit is not open anymore
    open_until = float(st.get("open_until_ts", 0.0) or 0.0)
    if time.time() >= open_until:
        st["open_until_ts"] = 0.0
        st["last_status"] = "ok"
        st["last_error"] = None
        st["fail_count"] = 0
        _save_circuit(st)


# -----------------------------
# Demo dataset loader
# -----------------------------
_demo_lock = threading.Lock()
_demo_cache: Optional[Dict[str, Any]] = None


def _load_demo_data() -> Optional[Dict[str, Any]]:
    global _demo_cache
    with _demo_lock:
        if _demo_cache is not None:
            return _demo_cache
        try:
            if not DEMO_PATH.exists():
                return None
            with DEMO_PATH.open("r", encoding="utf-8") as f:
                _demo_cache = json.load(f)
            return _demo_cache
        except Exception:
            return None


def _demo_timeseries(keyword: str, timeframe_label: str) -> List[Dict[str, Any]]:
    d = _load_demo_data()
    if not d:
        return []
    kw = keyword.lower().strip()
    tf_key = "6m" if timeframe_label == "6 months" else "12m"
    try:
        return list(d.get("keywords", {}).get(kw, {}).get(tf_key, []) or [])
    except Exception:
        return []


# -----------------------------
# Core: run with retries + circuit + throttle
# -----------------------------
def _run_trends_call(fn: Callable[[], Any], *, retries: int = DEFAULT_RETRIES) -> Any:
    if FORCE_SAFE_MODE:
        raise RuntimeError("Safe mode forced: skipping Trends calls.")

    last_err: Optional[Exception] = None

    for attempt in range(max(1, retries)):
        try:
            _throttle_acquire()
            out = fn()
            _record_success()
            return out
        except Exception as e:
            last_err = e
            is_429 = _is_429_error(e)
            _record_failure(f"{type(e).__name__}: {str(e)[:180]}", is_rate_limit=is_429)

            if is_429:
                # DO NOT hammer on 429; break early
                break
            # other errors: small backoff then retry
            _sleep_backoff(attempt)

    if last_err:
        raise last_err
    raise RuntimeError("Unknown Trends error.")


# -----------------------------
# Fallback helpers
# -----------------------------
def _fallback_cached_or_demo(
    *,
    cache_key: str,
    base_meta: Dict[str, Any],
    keyword: Optional[str] = None,
    keywords: Optional[List[str]] = None,
    timeframe_label: Optional[str] = None,
    err: Optional[Exception] = None,
    forced_demo: bool = False,
    circuit_open: bool = False,
    rate_limited: bool = False,
) -> Dict[str, Any]:
    """
    Fallback order:
      1) fresh cache (if available)
      2) any cache (stale) if circuit/rate-limited
      3) demo dataset (if available)
      4) empty result
    Adds meta flags for UI.
    """
    meta = dict(base_meta)
    meta["cached"] = False
    meta["demo_mode"] = False
    meta["stale_fallback"] = False
    meta["rate_limited"] = bool(rate_limited)
    meta["circuit_open"] = bool(circuit_open)

    if err is not None:
        meta["error"] = f"{type(err).__name__}"
    else:
        meta["error"] = None

    # If not forcing demo, try fresh cache first
    if not forced_demo:
        try:
            cached_fresh = read_cache(cache_key, max_age_s=TRENDS_CACHE_TTL_S)
        except TypeError:
            cached_fresh = read_cache(cache_key)

        if cached_fresh is not None:
            cached_meta = dict(cached_fresh.get("meta", {}))
            cached_meta["cached"] = True
            cached_meta["demo_mode"] = False
            cached_meta["stale_fallback"] = False
            cached_meta["rate_limited"] = bool(rate_limited)
            cached_meta["circuit_open"] = bool(circuit_open)
            cached_fresh["meta"] = cached_meta
            return cached_fresh

        # If circuit open / rate limited, allow stale cache too
        if circuit_open or rate_limited:
            cached_any = read_cache(cache_key)
            if cached_any is not None:
                cached_meta = dict(cached_any.get("meta", {}))
                cached_meta["cached"] = True
                cached_meta["demo_mode"] = False
                cached_meta["stale_fallback"] = True
                cached_meta["rate_limited"] = bool(rate_limited)
                cached_meta["circuit_open"] = bool(circuit_open)
                if err is not None:
                    cached_meta["error"] = f"served cached due to {type(err).__name__}"
                cached_any["meta"] = cached_meta
                return cached_any

    # Demo fallback
    if keyword and timeframe_label:
        ts = _demo_timeseries(keyword, timeframe_label)
        if ts:
            meta["demo_mode"] = True
            meta["stale_fallback"] = True
            meta["error"] = meta["error"] or "served demo fallback"
            return {"meta": meta, "timeseries": ts, "points": len(ts)}

    if keywords and timeframe_label:
        series: Dict[str, List[Dict[str, Any]]] = {}
        any_points = 0
        for kw in keywords:
            ts = _demo_timeseries(kw, timeframe_label)
            series[kw] = ts
            any_points = max(any_points, len(ts))
        if any_points > 0:
            meta["demo_mode"] = True
            meta["stale_fallback"] = True
            meta["error"] = meta["error"] or "served demo fallback"
            return {"meta": meta, "series": series, "points": any_points}

    # Nothing available
    meta["stale_fallback"] = True if (rate_limited or circuit_open) else False
    return {
        "meta": meta,
        "timeseries": [] if keyword else None,
        "series": {k: [] for k in (keywords or [])} if keywords else None,
        "points": 0,
    }


# -----------------------------
# Public API
# -----------------------------
def fetch_trends_single(
    keyword: str,
    *,
    timeframe_label: str,
    region_label: str,
    use_cache: bool = True,
    force_demo: bool = False,
) -> Dict[str, Any]:
    kw = clean_keyword(keyword)
    geo = region_to_geo(region_label)
    tf, keep_last = _timeframe_to_pytrends_tf(timeframe_label)

    cache_key = make_cache_key(
        "trends_single",
        keyword=kw,
        timeframe=timeframe_label,
        region=region_label,
        extra={"geo": geo, "tf": tf, "keep_last": keep_last},
    )

    base_meta: Dict[str, Any] = {
        "keyword": kw,
        "region": region_label,
        "geo": geo,
        "timeframe": timeframe_label,
        "pytrends_timeframe": tf,
        "fetched_at": utc_now_iso(),
        "cached": False,
        "demo_mode": False,
        "stale_fallback": False,
        "rate_limited": False,
        "circuit_open": False,
        "error": None,
    }

    # Force demo immediately
    if force_demo:
        return _fallback_cached_or_demo(
            cache_key=cache_key,
            base_meta=base_meta,
            keyword=kw,
            timeframe_label=timeframe_label,
            forced_demo=True,
        )

    # Circuit open => do not hit Trends
    is_open, _remaining = _circuit_is_open()
    if is_open or FORCE_SAFE_MODE:
        return _fallback_cached_or_demo(
            cache_key=cache_key,
            base_meta=base_meta,
            keyword=kw,
            timeframe_label=timeframe_label,
            forced_demo=False,
            circuit_open=True,
            rate_limited=True,
        )

    # Cache-first (fresh) if enabled
    if use_cache:
        try:
            cached = read_cache(cache_key, max_age_s=TRENDS_CACHE_TTL_S)
        except TypeError:
            cached = read_cache(cache_key)
        if cached is not None:
            cached_meta = dict(cached.get("meta", {}))
            cached_meta["cached"] = True
            cached_meta.setdefault("demo_mode", False)
            cached_meta.setdefault("stale_fallback", False)
            cached_meta.setdefault("rate_limited", False)
            cached_meta.setdefault("circuit_open", False)
            cached["meta"] = cached_meta
            return cached

    def _do():
        pytr = _get_shared_client()
        pytr.build_payload([kw], timeframe=tf, geo=geo)
        return pytr.interest_over_time()

    try:
        df = _run_trends_call(_do, retries=DEFAULT_RETRIES)
    except Exception as e:
        rate_limited = _is_429_error(e)
        return _fallback_cached_or_demo(
            cache_key=cache_key,
            base_meta=base_meta,
            keyword=kw,
            timeframe_label=timeframe_label,
            err=e,
            forced_demo=False,
            circuit_open=rate_limited or FORCE_SAFE_MODE,
            rate_limited=rate_limited or FORCE_SAFE_MODE,
        )

    timeseries = _df_to_timeseries(df, kw, keep_last=keep_last)
    result = {"meta": base_meta, "timeseries": timeseries, "points": len(timeseries)}

    if use_cache:
        write_cache(cache_key, result)

    cooldown_sleep()
    return result


def fetch_trends_compare(
    keywords: List[str],
    *,
    timeframe_label: str,
    region_label: str,
    use_cache: bool = True,
    force_demo: bool = False,
) -> Dict[str, Any]:
    # Keep display list (up to 3) but compute stable cache identity
    kws_display = [clean_keyword(k) for k in keywords if clean_keyword(k)]
    kws_display = kws_display[:3]
    kws_key = _normalize_keywords_for_cache(kws_display)

    geo = region_to_geo(region_label)
    tf, keep_last = _timeframe_to_pytrends_tf(timeframe_label)

    cache_key = make_cache_key(
        "trends_compare",
        keywords=kws_key,  # stable order-insensitive identity
        timeframe=timeframe_label,
        region=region_label,
        extra={"geo": geo, "tf": tf, "keep_last": keep_last},
    )

    base_meta: Dict[str, Any] = {
        "keywords": kws_display,
        "keywords_key": kws_key,
        "region": region_label,
        "geo": geo,
        "timeframe": timeframe_label,
        "pytrends_timeframe": tf,
        "fetched_at": utc_now_iso(),
        "cached": False,
        "demo_mode": False,
        "stale_fallback": False,
        "rate_limited": False,
        "circuit_open": False,
        "error": None,
    }

    if force_demo:
        return _fallback_cached_or_demo(
            cache_key=cache_key,
            base_meta=base_meta,
            keywords=kws_display,
            timeframe_label=timeframe_label,
            forced_demo=True,
        )

    is_open, _remaining = _circuit_is_open()
    if is_open or FORCE_SAFE_MODE:
        return _fallback_cached_or_demo(
            cache_key=cache_key,
            base_meta=base_meta,
            keywords=kws_display,
            timeframe_label=timeframe_label,
            circuit_open=True,
            rate_limited=True,
        )

    if use_cache:
        try:
            cached = read_cache(cache_key, max_age_s=TRENDS_CACHE_TTL_S)
        except TypeError:
            cached = read_cache(cache_key)
        if cached is not None:
            cached_meta = dict(cached.get("meta", {}))
            cached_meta["cached"] = True
            cached_meta.setdefault("demo_mode", False)
            cached_meta.setdefault("stale_fallback", False)
            cached_meta.setdefault("rate_limited", False)
            cached_meta.setdefault("circuit_open", False)
            cached["meta"] = cached_meta
            return cached

    def _do():
        pytr = _get_shared_client()
        pytr.build_payload(kws_display, timeframe=tf, geo=geo)
        return pytr.interest_over_time()

    try:
        df = _run_trends_call(_do, retries=DEFAULT_RETRIES)
    except Exception as e:
        rate_limited = _is_429_error(e)
        return _fallback_cached_or_demo(
            cache_key=cache_key,
            base_meta=base_meta,
            keywords=kws_display,
            timeframe_label=timeframe_label,
            err=e,
            circuit_open=rate_limited or FORCE_SAFE_MODE,
            rate_limited=rate_limited or FORCE_SAFE_MODE,
        )

    series_dict: Dict[str, List[Dict[str, Any]]] = {}
    for kw in kws_display:
        series_dict[kw] = _df_to_timeseries(df, kw, keep_last=keep_last)

    points = len(series_dict[kws_display[0]]) if kws_display else 0
    result = {"meta": base_meta, "series": series_dict, "points": points}

    if use_cache:
        write_cache(cache_key, result)

    cooldown_sleep()
    return result


def fetch_related_queries(
    keyword: str,
    *,
    region_label: str,
    use_cache: bool = True,
    force_demo: bool = False,
) -> Dict[str, Any]:
    """
    Related queries are optional. If rate-limited/circuit open:
      - serve cached if present
      - else return empty with meta flags (no demo for related)
    """
    kw = clean_keyword(keyword)
    geo = region_to_geo(region_label)

    cache_key = make_cache_key(
        "trends_related_queries",
        keyword=kw,
        region=region_label,
        extra={"geo": geo},
    )

    base_meta: Dict[str, Any] = {
        "keyword": kw,
        "region": region_label,
        "geo": geo,
        "fetched_at": utc_now_iso(),
        "cached": False,
        "demo_mode": bool(force_demo),  # demo doesn't apply; kept for UI consistency
        "stale_fallback": False,
        "rate_limited": False,
        "circuit_open": False,
        "error": None,
    }

    # If forcing demo, just return empty (we don't demo related queries)
    if force_demo:
        base_meta["demo_mode"] = True
        base_meta["stale_fallback"] = True
        base_meta["error"] = "demo mode: related queries unavailable"
        return {"meta": base_meta, "top": [], "rising": []}

    is_open, _remaining = _circuit_is_open()
    if is_open or FORCE_SAFE_MODE:
        cached = read_cache(cache_key) if use_cache else None
        if cached is not None:
            cached_meta = dict(cached.get("meta", {}))
            cached_meta["cached"] = True
            cached_meta["stale_fallback"] = True
            cached_meta["rate_limited"] = True
            cached_meta["circuit_open"] = True
            cached_meta["error"] = "served cached: circuit open"
            cached["meta"] = cached_meta
            return cached

        base_meta["rate_limited"] = True
        base_meta["circuit_open"] = True
        base_meta["stale_fallback"] = True
        base_meta["error"] = "circuit open: related queries skipped"
        return {"meta": base_meta, "top": [], "rising": []}

    if use_cache:
        try:
            cached = read_cache(cache_key, max_age_s=TRENDS_CACHE_TTL_S)
        except TypeError:
            cached = read_cache(cache_key)
        if cached is not None:
            cached_meta = dict(cached.get("meta", {}))
            cached_meta["cached"] = True
            cached_meta.setdefault("rate_limited", False)
            cached_meta.setdefault("circuit_open", False)
            cached_meta.setdefault("stale_fallback", False)
            cached["meta"] = cached_meta
            return cached

    def _do():
        pytr = _get_shared_client()
        pytr.build_payload([kw], timeframe="today 12-m", geo=geo)
        return pytr.related_queries()

    try:
        rel = _run_trends_call(_do, retries=max(2, DEFAULT_RETRIES))
    except Exception as e:
        cached = read_cache(cache_key) if use_cache else None
        if cached is not None:
            cached_meta = dict(cached.get("meta", {}))
            cached_meta["cached"] = True
            cached_meta["stale_fallback"] = True
            cached_meta["rate_limited"] = bool(_is_429_error(e))
            cached_meta["circuit_open"] = bool(_is_429_error(e))
            cached_meta["error"] = f"served cached due to {type(e).__name__}"
            cached["meta"] = cached_meta
            return cached

        base_meta["error"] = f"{type(e).__name__}"
        if _is_429_error(e):
            base_meta["rate_limited"] = True
            base_meta["circuit_open"] = True
            base_meta["stale_fallback"] = True
        return {"meta": base_meta, "top": [], "rising": []}

    top_df = None
    rising_df = None
    if isinstance(rel, dict) and kw in rel:
        top_df = rel[kw].get("top")
        rising_df = rel[kw].get("rising")

    def _df_to_list(d: Optional[pd.DataFrame], limit: int = 10) -> List[Dict[str, Any]]:
        if d is None or d.empty:
            return []
        d2 = d.head(limit).copy()
        out: List[Dict[str, Any]] = []
        for _, row in d2.iterrows():
            out.append({"query": str(row.get("query", "")), "value": float(row.get("value", 0))})
        return out

    result = {"meta": base_meta, "top": _df_to_list(top_df, limit=10), "rising": _df_to_list(rising_df, limit=10)}

    if use_cache:
        write_cache(cache_key, result)

    cooldown_sleep()
    return result

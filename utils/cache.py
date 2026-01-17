# utils/cache.py
# Lightweight file-based caching with TTL + refresh support.
# Updated (2026-01):
# - Adds read_cache(..., max_age_s=...) freshness-aware reads (critical for Trends safe-mode)
# - Keeps backward compatibility: allow_stale, namespace TTL, read_cache_fresh, read_cache_with_meta
# - Improves atomic write safety + lock staleness handling
# - Adds optional stampede/in-flight locking helpers (best-effort)
# - Standardizes meta injection: meta["_age_seconds"], meta["stale"]

from __future__ import annotations

import hashlib
import json
import os
import time
from typing import Any, Dict, Optional

from config import CACHE_CFG


# -----------------------------
# Internal helpers
# -----------------------------
def _ensure_cache_dir() -> None:
    os.makedirs(CACHE_CFG.cache_dir, exist_ok=True)


def _hash_key(parts: Dict[str, Any]) -> str:
    """
    Builds a stable hash key from a dict of parameters.
    Order-independent and safe for filenames.
    """
    payload = json.dumps(parts, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _cache_path(key_hash: str) -> str:
    return os.path.join(CACHE_CFG.cache_dir, f"{key_hash}.json")


def _throttle_path() -> str:
    # Single file used for simple cross-process throttling
    return os.path.join(CACHE_CFG.cache_dir, "_throttle.json")


def _lock_path(key_hash: Optional[str] = None) -> str:
    # Best-effort lock file (per-key if key_hash given, else global)
    if key_hash:
        return os.path.join(CACHE_CFG.cache_dir, f"_{key_hash}.lock")
    return os.path.join(CACHE_CFG.cache_dir, "_cache_write.lock")


def _inflight_path(key_hash: str) -> str:
    # Optional “in-flight” lock (helps prevent stampede recompute on reruns)
    return os.path.join(CACHE_CFG.cache_dir, f"_{key_hash}.inflight")


def _is_expired(ts: float, ttl: int) -> bool:
    return (time.time() - ts) > float(ttl)


def _atomic_write_json(path: str, blob: Dict[str, Any]) -> None:
    """
    Atomic-ish write: write to temp then replace.
    Helps prevent corrupted cache files on sudden exits.
    """
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(blob, f, ensure_ascii=False)
    os.replace(tmp, path)


def _ttl_for_namespace(namespace: Optional[str]) -> int:
    """
    Support optional per-namespace TTL overrides from CACHE_CFG.
    Backward compatible: if not provided, uses CACHE_CFG.ttl_seconds.
    """
    base = int(getattr(CACHE_CFG, "ttl_seconds", 3600))

    # Optional: CACHE_CFG.ttl_by_namespace = {"trends_single": 21600, ...}
    ttl_map = getattr(CACHE_CFG, "ttl_by_namespace", None)
    if isinstance(ttl_map, dict) and namespace:
        try:
            v = ttl_map.get(namespace)
            if v is not None:
                return int(v)
        except Exception:
            pass
    return base


def _try_acquire_lock(path: str, *, stale_after_s: float = 15.0) -> bool:
    """
    Best-effort lock: create lock file exclusively.
    If lock exists but is stale, remove it.
    """
    _ensure_cache_dir()
    now = time.time()

    # If stale lock, remove it
    try:
        if os.path.exists(path):
            age = now - os.path.getmtime(path)
            if age > float(stale_after_s):
                try:
                    os.remove(path)
                except OSError:
                    pass
    except Exception:
        pass

    try:
        # O_EXCL makes this atomic across processes
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        try:
            os.write(fd, str(now).encode("utf-8"))
        finally:
            os.close(fd)
        return True
    except FileExistsError:
        return False
    except Exception:
        return False


def _release_lock(path: str) -> None:
    try:
        if os.path.exists(path):
            os.remove(path)
    except OSError:
        pass


def _inject_cache_meta(data: Any, *, age_seconds: float, stale: bool) -> Any:
    """
    Injects meta fields into dict payloads without breaking existing structure.
    """
    if isinstance(data, dict):
        m = dict(data.get("meta", {}))
        m["_age_seconds"] = float(age_seconds)
        m["stale"] = bool(stale)
        data["meta"] = m
    return data


# -----------------------------
# Public cache API
# -----------------------------
def make_cache_key(
    namespace: str,
    *,
    keyword: Optional[str] = None,
    keywords: Optional[list] = None,
    timeframe: Optional[str] = None,
    region: Optional[str] = None,
    context: Optional[str] = None,
    news_days: Optional[int] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Build a cache key hash for any request.

    `namespace` distinguishes data types (e.g., "trends_single",
    "trends_compare", "news", "sentiment").
    """
    parts: Dict[str, Any] = {
        "ns": namespace,
        "keyword": keyword,
        "keywords": keywords,
        "timeframe": timeframe,
        "region": region,
        "context": context,
        "news_days": news_days,
        "extra": extra or {},
    }
    return _hash_key(parts)


def read_cache(
    key_hash: str,
    *,
    namespace: Optional[str] = None,
    allow_stale: bool = False,
    max_age_s: Optional[float] = None,
) -> Optional[Dict[str, Any]]:
    """
    Read cache entry.

    Modes:
      A) max_age_s provided:
         - returns only if age_seconds <= max_age_s (freshness-based)
         - ignores namespace TTL unless allow_stale=True fallback is desired by caller
      B) max_age_s is None:
         - uses namespace TTL (or default) as before:
             allow_stale=False -> only fresh
             allow_stale=True  -> return even if expired (meta.stale=True)

    Meta injected into payload dict (if payload is a dict):
      meta["_age_seconds"] = float
      meta["stale"] = bool
    """
    meta = read_cache_with_meta(key_hash)
    if meta is None:
        return None

    ts = float(meta["ts"])
    age = float(meta["age_seconds"])

    # Freshness-based read
    if max_age_s is not None:
        if age > float(max_age_s):
            if not allow_stale:
                return None
            # allow stale return
            data = meta["data"]
            return _inject_cache_meta(data, age_seconds=age, stale=True)
        data = meta["data"]
        return _inject_cache_meta(data, age_seconds=age, stale=False)

    # TTL-based read (backward compatible)
    ttl = _ttl_for_namespace(namespace)
    expired = _is_expired(ts, ttl)
    if expired and not allow_stale:
        return None

    data = meta["data"]
    return _inject_cache_meta(data, age_seconds=age, stale=bool(expired))


def read_cache_with_meta(key_hash: str) -> Optional[Dict[str, Any]]:
    """
    Returns:
      {"data": <payload>, "ts": <float>, "age_seconds": <float>}
    or None if not present / unreadable.

    NOTE: This does not enforce TTL by itself.
    """
    _ensure_cache_dir()
    path = _cache_path(key_hash)
    if not os.path.exists(path):
        return None

    try:
        with open(path, "r", encoding="utf-8") as f:
            blob = json.load(f)
        ts = float(blob.get("_ts", 0.0) or 0.0)
        if ts <= 0:
            return None
        age = time.time() - ts
        return {"data": blob.get("data"), "ts": ts, "age_seconds": float(age)}
    except Exception:
        return None


def read_cache_fresh(key_hash: str, *, namespace: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    Strict TTL read: returns data only if not expired for this namespace.
    Deletes expired cache file (kept for compatibility with old callers who relied on deletion).
    Prefer `read_cache(..., allow_stale=True)` for stale-while-revalidate behavior.
    """
    _ensure_cache_dir()
    path = _cache_path(key_hash)
    if not os.path.exists(path):
        return None

    meta = read_cache_with_meta(key_hash)
    if meta is None:
        return None

    ttl = _ttl_for_namespace(namespace)
    if _is_expired(float(meta["ts"]), ttl):
        try:
            os.remove(path)
        except OSError:
            pass
        return None

    data = meta["data"]
    return _inject_cache_meta(data, age_seconds=float(meta["age_seconds"]), stale=False)


def write_cache(key_hash: str, data: Dict[str, Any]) -> None:
    """
    Writes payload to cache with timestamp.
    Uses a best-effort lock to avoid simultaneous writes across Streamlit reruns/processes.
    """
    _ensure_cache_dir()
    path = _cache_path(key_hash)
    blob = {"_ts": time.time(), "data": data}

    lock = _lock_path(key_hash)
    acquired = _try_acquire_lock(lock, stale_after_s=20.0)
    try:
        # Always atomic write if possible
        _atomic_write_json(path, blob)
    except Exception:
        # last-resort non-atomic write
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(blob, f, ensure_ascii=False)
        except Exception:
            pass
    finally:
        if acquired:
            _release_lock(lock)


def clear_cache(key_hash: Optional[str] = None) -> None:
    """
    Clears cache:
    - if key_hash is provided → clears only that entry
    - else → clears entire cache directory (including throttle/lock files)
    """
    _ensure_cache_dir()
    if key_hash:
        path = _cache_path(key_hash)
        try:
            if os.path.exists(path):
                os.remove(path)
        except OSError:
            pass
        # also remove per-key lock if present
        try:
            lp = _lock_path(key_hash)
            if os.path.exists(lp):
                os.remove(lp)
        except OSError:
            pass
        # inflight
        try:
            ip = _inflight_path(key_hash)
            if os.path.exists(ip):
                os.remove(ip)
        except OSError:
            pass
        return

    for fname in os.listdir(CACHE_CFG.cache_dir):
        if fname.endswith(".json") or fname.endswith(".lock") or fname.endswith(".inflight") or fname.startswith("_"):
            try:
                os.remove(os.path.join(CACHE_CFG.cache_dir, fname))
            except OSError:
                pass


# -----------------------------
# Optional stampede protection
# -----------------------------
def try_begin_inflight(key_hash: str, *, stale_after_s: float = 30.0) -> bool:
    """
    Best-effort 'in-flight' marker to prevent multiple reruns from recomputing
    the same expensive key simultaneously.

    Returns True if you acquired inflight marker, else False.
    """
    return _try_acquire_lock(_inflight_path(key_hash), stale_after_s=stale_after_s)


def end_inflight(key_hash: str) -> None:
    """
    Release in-flight marker.
    """
    _release_lock(_inflight_path(key_hash))


# -----------------------------
# Global throttle / cooldown
# -----------------------------
def cooldown_sleep() -> None:
    """
    Small cooldown to avoid rate limits when making multiple requests.
    Backward compatible: uses CACHE_CFG.cooldown_seconds.
    """
    time.sleep(float(getattr(CACHE_CFG, "cooldown_seconds", 0.8)))


def global_throttle(min_interval_s: Optional[float] = None) -> None:
    """
    Cross-process throttle: ensures at least `min_interval_s` seconds between
    external calls (Google Trends, etc.) even if multiple Streamlit reruns happen.

    Uses a single file in cache_dir to store last request timestamp.

    If min_interval_s is None, uses CACHE_CFG.global_throttle_seconds if present,
    else falls back to CACHE_CFG.cooldown_seconds.
    """
    _ensure_cache_dir()
    interval = float(
        min_interval_s
        if min_interval_s is not None
        else getattr(CACHE_CFG, "global_throttle_seconds", getattr(CACHE_CFG, "cooldown_seconds", 0.8))
    )
    if interval <= 0:
        return

    path = _throttle_path()
    lock = _lock_path("throttle")

    acquired = _try_acquire_lock(lock, stale_after_s=10.0)
    try:
        now = time.time()
        last = 0.0

        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    blob = json.load(f)
                last = float(blob.get("last_ts", 0.0) or 0.0)
            except Exception:
                last = 0.0

        elapsed = now - last
        if elapsed < interval:
            time.sleep(max(0.0, interval - elapsed))

        # Update last_ts
        try:
            _atomic_write_json(path, {"last_ts": time.time()})
        except Exception:
            pass
    finally:
        if acquired:
            _release_lock(lock)

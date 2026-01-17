# storage/watchlist.py
# Watchlist storage using SQLite (recommended, no API keys).

from __future__ import annotations

import os
import sqlite3
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone


@dataclass(frozen=True)
class WatchItem:
    keyword: str
    region: str                 # e.g., "WORLD" or "INDIA"
    last_score: Optional[float]
    last_confidence: Optional[float]
    last_analyzed_utc: Optional[str]


class WatchlistStore:
    def __init__(self, db_path: str = "data/watchlist.db") -> None:
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS watchlist (
                    keyword TEXT NOT NULL,
                    region TEXT NOT NULL,
                    last_score REAL,
                    last_confidence REAL,
                    last_analyzed_utc TEXT,
                    PRIMARY KEY(keyword, region)
                )
                """
            )
            conn.commit()

    @staticmethod
    def _now_iso() -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    def upsert(
        self,
        *,
        keyword: str,
        region: str,
        score: Optional[float],
        confidence: Optional[float],
    ) -> None:
        keyword = (keyword or "").strip()
        region = (region or "").strip().upper()
        if not keyword or not region:
            return

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO watchlist(keyword, region, last_score, last_confidence, last_analyzed_utc)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(keyword, region) DO UPDATE SET
                    last_score=excluded.last_score,
                    last_confidence=excluded.last_confidence,
                    last_analyzed_utc=excluded.last_analyzed_utc
                """,
                (keyword, region, score, confidence, self._now_iso()),
            )
            conn.commit()

    def get(self, *, keyword: str, region: str) -> Optional[WatchItem]:
        keyword = (keyword or "").strip()
        region = (region or "").strip().upper()
        if not keyword or not region:
            return None

        with self._connect() as conn:
            cur = conn.execute(
                "SELECT * FROM watchlist WHERE keyword=? AND region=?",
                (keyword, region),
            )
            row = cur.fetchone()
            if not row:
                return None
            return WatchItem(
                keyword=row["keyword"],
                region=row["region"],
                last_score=row["last_score"],
                last_confidence=row["last_confidence"],
                last_analyzed_utc=row["last_analyzed_utc"],
            )

    def list_all(self) -> List[WatchItem]:
        with self._connect() as conn:
            cur = conn.execute(
                "SELECT * FROM watchlist ORDER BY last_analyzed_utc DESC NULLS LAST"
            )
            rows = cur.fetchall()

        items: List[WatchItem] = []
        for r in rows:
            items.append(
                WatchItem(
                    keyword=r["keyword"],
                    region=r["region"],
                    last_score=r["last_score"],
                    last_confidence=r["last_confidence"],
                    last_analyzed_utc=r["last_analyzed_utc"],
                )
            )
        return items

    def remove(self, *, keyword: str, region: str) -> None:
        keyword = (keyword or "").strip()
        region = (region or "").strip().upper()
        if not keyword or not region:
            return

        with self._connect() as conn:
            conn.execute(
                "DELETE FROM watchlist WHERE keyword=? AND region=?",
                (keyword, region),
            )
            conn.commit()

    def compute_delta(
        self,
        *,
        keyword: str,
        region: str,
        new_score: Optional[float],
    ) -> Optional[float]:
        """
        Returns new_score - old_score if old exists.
        """
        if new_score is None:
            return None
        old = self.get(keyword=keyword, region=region)
        if not old or old.last_score is None:
            return None
        try:
            return float(new_score) - float(old.last_score)
        except Exception:
            return None

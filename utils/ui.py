# utils/ui.py
# Premium UI helpers for Streamlit (no business-logic changes).
# Drop-in utilities: theme CSS, section headers, status pills, KPI cards, safe markdown.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import streamlit as st


# -----------------------------
# Core styling (CSS)
# -----------------------------
_DEFAULT_CSS = """
<style>
/* --- Global spacing tweaks --- */
.block-container { padding-top: 1.6rem; padding-bottom: 2rem; }
div[data-testid="stVerticalBlock"] > div:has(> div[data-testid="stMetric"]) { margin-bottom: 0.25rem; }

/* --- Cards --- */
.ui-card {
  background: #ffffff;
  border: 1px solid rgba(17,24,39,0.10);
  box-shadow: 0 8px 30px rgba(17,24,39,0.06);
  border-radius: 16px;
  padding: 14px 16px;
}
.ui-card.compact { padding: 12px 14px; border-radius: 14px; }
.ui-card .muted { color: rgba(17,24,39,0.60); font-size: 0.85rem; }
.ui-card .title { font-weight: 700; font-size: 0.95rem; margin-bottom: 0.35rem; }
.ui-card .big { font-size: 1.35rem; font-weight: 800; line-height: 1.2; }
.ui-card .delta { font-size: 0.9rem; font-weight: 600; margin-top: 0.15rem; }
.ui-card .delta.pos { color: #0F766E; }
.ui-card .delta.neg { color: #B42318; }
.ui-card .delta.neu { color: rgba(17,24,39,0.65); }

/* --- Section header --- */
.ui-section {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  gap: 10px;
  margin: 8px 0 10px 0;
}
.ui-section h2 {
  font-size: 1.15rem;
  font-weight: 800;
  margin: 0;
}
.ui-section .hint {
  color: rgba(17,24,39,0.60);
  font-size: 0.88rem;
}

/* --- Pill / status chips --- */
.ui-pill {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  padding: 6px 10px;
  border-radius: 999px;
  font-weight: 700;
  font-size: 0.85rem;
  border: 1px solid rgba(17,24,39,0.10);
  background: #ffffff;
}
.ui-dot { width: 8px; height: 8px; border-radius: 999px; display: inline-block; }
.ui-pill.ok    { border-color: rgba(15,118,110,0.35); }
.ui-pill.info  { border-color: rgba(49,41,148,0.25); }
.ui-pill.warn  { border-color: rgba(245,158,11,0.35); }
.ui-pill.bad   { border-color: rgba(180,35,24,0.35); }

.ui-dot.ok   { background: #0F766E; }
.ui-dot.info { background: #312994; }
.ui-dot.warn { background: #F59E0B; }
.ui-dot.bad  { background: #B42318; }

/* --- Caption text --- */
.ui-caption {
  color: rgba(17,24,39,0.60);
  font-size: 0.9rem;
}

/* --- Sidebar polish --- */
section[data-testid="stSidebar"] .block-container { padding-top: 1.25rem; }
section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h3 {
  letter-spacing: -0.01em;
}

/* --- Reduce default table harsh borders --- */
div[data-testid="stDataFrame"] { border-radius: 14px; overflow: hidden; }
</style>
"""


def inject_global_css(css: str = _DEFAULT_CSS) -> None:
    """Inject global CSS once per page render."""
    st.markdown(css, unsafe_allow_html=True)


# -----------------------------
# UI Components
# -----------------------------
def section_header(title: str, hint: str = "") -> None:
    """Nice section header with optional right-side hint."""
    right = f'<div class="hint">{hint}</div>' if hint else '<div class="hint"></div>'
    st.markdown(
        f'<div class="ui-section"><h2>{_escape(title)}</h2>{right}</div>',
        unsafe_allow_html=True,
    )


def caption(text: str) -> None:
    st.markdown(f'<div class="ui-caption">{_escape(text)}</div>', unsafe_allow_html=True)


@dataclass(frozen=True)
class Pill:
    label: str
    tone: str = "info"  # ok/info/warn/bad


def status_pill(label: str, tone: str = "info") -> None:
    """Inline status pill with colored dot."""
    tone = (tone or "info").strip().lower()
    if tone not in ("ok", "info", "warn", "bad"):
        tone = "info"
    st.markdown(
        f'<span class="ui-pill {tone}"><span class="ui-dot {tone}"></span>{_escape(label)}</span>',
        unsafe_allow_html=True,
    )


def trends_status_to_pill(meta: Dict[str, Any]) -> Pill:
    """
    Convert trends meta flags to a (label,tone) pill.
    Expected meta flags from your trends service:
      demo_mode, cached, stale, stale_fallback, rate_limited, circuit_open
    """
    if not isinstance(meta, dict):
        return Pill("Trends: Unavailable", "bad")

    if meta.get("demo_mode"):
        return Pill("Trends: Demo (offline)", "warn")

    if meta.get("rate_limited") or meta.get("circuit_open"):
        # Even if cached served, we still show rate-limited
        return Pill("Trends: Rate-limited", "warn")

    if meta.get("cached") or meta.get("stale") or meta.get("stale_fallback"):
        return Pill("Trends: Using cache", "info")

    # If we have points, assume OK
    return Pill("Trends: OK", "ok")


def kpi_card(
    title: str,
    value: str,
    *,
    delta: Optional[str] = None,
    delta_tone: Optional[str] = None,  # pos/neg/neu (auto if None)
    help_text: str = "",
    compact: bool = True,
) -> None:
    """
    KPI card with optional delta.
    - delta_tone: "pos" | "neg" | "neu". If None, tries to infer from leading +/-.
    """
    klass = "ui-card compact" if compact else "ui-card"
    dhtml = ""
    if delta is not None and str(delta).strip() != "":
        tone = (delta_tone or "").strip().lower()
        if tone not in ("pos", "neg", "neu"):
            # infer
            ds = str(delta).strip()
            if ds.startswith("+"):
                tone = "pos"
            elif ds.startswith("-"):
                tone = "neg"
            else:
                tone = "neu"
        dhtml = f'<div class="delta {tone}">{_escape(str(delta))}</div>'

    help_html = f'<div class="muted">{_escape(help_text)}</div>' if help_text else ""
    st.markdown(
        f"""
        <div class="{klass}">
          <div class="title">{_escape(title)}</div>
          <div class="big">{_escape(value)}</div>
          {dhtml}
          {help_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def callout(message: str, tone: str = "info") -> None:
    """
    Premium-looking callout bar using Streamlit native messages (safer),
    but you can keep it consistent with your app.
    """
    tone = (tone or "info").strip().lower()
    if tone == "ok":
        st.success(message)
    elif tone == "warn":
        st.warning(message)
    elif tone == "bad":
        st.error(message)
    else:
        st.info(message)


# -----------------------------
# Utilities
# -----------------------------
def _escape(s: str) -> str:
    """Minimal HTML escaping for safe string interpolation."""
    if s is None:
        return ""
    return (
        str(s)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )

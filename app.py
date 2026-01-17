# app.py
# Streamlit UI for Real-Time Market Trend Analysis (final integration + Phase 2A upgrades)
# Updated (safe-mode + compare correctness + anti-rerun hammering):
# - Prevent repeated Trends calls on Streamlit reruns (result memoization in st.session_state)
# - Compare flow: normalize keys + ensure series built from validated keywords
# - Stronger safe-mode UX: cache-first, demo fallback, no double-fetch per rerun
# - Leaves existing UI/UX intact unless necessary for correctness
# NOTE: No changes to business logic calculations; only orchestration and resilience.

from __future__ import annotations

import os
import pandas as pd
import streamlit as st

from services.trends import fetch_trends_single, fetch_trends_compare, fetch_related_queries
from services.news import fetch_news_gdelt
from nlp.sentiment import score_headlines_vader

from analysis.metrics import compute_trend_metrics, compute_news_quality
from analysis.forecast import compute_forecast
from analysis.reco import build_recommendations

# Phase 2A (pro upgrades)
from analysis.advanced_metrics import compute_advanced_trend_metrics
from analysis.news_analytics import build_sentiment_timeseries, coverage_strength
from nlp.themes import extract_themes_from_headlines
from analysis.confidence import compute_confidence_score
from analysis.pro_scoring import compute_pro_score

from storage.watchlist import WatchlistStore
from exports.exporter import trends_timeseries_to_csv_bytes, headlines_to_csv_bytes, analysis_to_json_bytes

from config import (
    APP_TITLE,
    APP_VERSION,
    REGION_OPTIONS,
    TIMEFRAME_OPTIONS,
    NEWS_WINDOW_OPTIONS,
    CONTEXT_OPTIONS,
)
from utils.helpers import validate_keywords, news_window_to_days, utc_now_iso


# -----------------------------
# Optional premium UI helpers (safe if file missing)
# -----------------------------
try:
    from utils.ui import (
        inject_global_css,
        load_css_file,
        status_pill,
        section_title,
        info_callout,
        warn_callout,
        exec_summary_card,
    )
except Exception:  # pragma: no cover
    inject_global_css = None
    load_css_file = None
    status_pill = None
    section_title = None
    info_callout = None
    warn_callout = None
    exec_summary_card = None


# -----------------------------
# Optional visualization helpers (safe if file missing)
# -----------------------------
try:
    from utils.vis import (
        trend_line_chart,
        compare_trends_chart,
        sentiment_donut_or_bar,
        score_components_chart,
        scenario_table,
        headline_source_bar,
    )
except Exception:  # pragma: no cover
    trend_line_chart = None
    compare_trends_chart = None
    sentiment_donut_or_bar = None
    score_components_chart = None
    scenario_table = None
    headline_source_bar = None


# -----------------------------
# App config
# -----------------------------
st.set_page_config(page_title=APP_TITLE, layout="wide")


# -----------------------------
# Premium UI: shared CSS + safe fallback
# -----------------------------
def _inject_fallback_css() -> None:
    """Fallback CSS in case assets/styles.css isn't present."""
    st.markdown(
        """
        <style>
          .block-container { padding-top: 1.2rem; }
          [data-testid="stMetricLabel"] { opacity: 0.85; }

          .om-card {
            border: 1px solid rgba(49, 41, 148, 0.12);
            border-radius: 16px;
            padding: 14px 16px;
            background: rgba(255,255,255,0.65);
          }

          .om-pill {
            display:inline-flex;
            align-items:center;
            gap:8px;
            border-radius:999px;
            padding:6px 10px;
            font-size: 0.88rem;
            border:1px solid rgba(0,0,0,0.08);
            background: rgba(255,255,255,0.6);
            margin-right: 8px;
            margin-bottom: 6px;
            white-space: nowrap;
          }

          .dot { width:10px; height:10px; border-radius: 999px; display:inline-block; }

          .om-hero h1 { margin: 0 0 0.25rem 0; font-size: 1.65rem; line-height: 1.15; }
          .om-hero p { margin: 0.25rem 0 0 0; opacity: 0.82; }

          section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 { margin-top: 0.6rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )


# Try loading assets/styles.css first; if not available, use fallback.
if load_css_file:
    load_css_file("assets/styles.css")
else:
    css_path = os.path.join("assets", "styles.css")
    if os.path.exists(css_path):
        try:
            with open(css_path, "r", encoding="utf-8") as f:
                st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
        except Exception:
            _inject_fallback_css()
    else:
        _inject_fallback_css()

if inject_global_css:
    try:
        inject_global_css()
    except Exception:
        pass


# -----------------------------
# Header (hero)
# -----------------------------
st.markdown(
    f"""
    <div class="om-hero om-card ui-hero">
      <h1>{APP_TITLE} <span style="opacity:.65; font-size:1.05rem;">• {APP_VERSION}</span></h1>
      <p>Live signals: Google Trends + GDELT News + VADER Sentiment • Updated: <b>{utc_now_iso()}</b></p>
    </div>
    """,
    unsafe_allow_html=True,
)
st.write("")

# Watchlist store (SQLite)
watch_store = WatchlistStore(db_path="data/watchlist.db")


# -----------------------------
# Sidebar Controls (premium grouping)
# -----------------------------
st.sidebar.markdown("## Controls")

with st.sidebar.expander("Mode & Scope", expanded=True):
    mode = st.radio("Mode", ["Single Keyword", "Compare (up to 3)"], index=0)
    region_label = st.selectbox("Region", REGION_OPTIONS, index=0)
    timeframe_label = st.selectbox("Trend timeframe", TIMEFRAME_OPTIONS, index=1)

with st.sidebar.expander("News Filters", expanded=True):
    news_window_label = st.selectbox("News window", NEWS_WINDOW_OPTIONS, index=0)
    context_label = st.selectbox("News context", CONTEXT_OPTIONS, index=0)

with st.sidebar.expander("Safety & Performance", expanded=True):
    use_cache = st.checkbox("Use cache (faster, safer)", value=True)
    demo_mode = st.checkbox(
        "Demo mode (offline)",
        value=False,
        help="Forces demo trends fallback so the app works even if Trends is blocked.",
    )
    sentiment_bucket = st.selectbox("Sentiment bucket", ["weekly", "daily"], index=0)

with st.sidebar.expander("Visualization", expanded=False):
    chart_height = st.slider("Chart height", min_value=220, max_value=520, value=320, step=20)
    show_raw_tables = st.checkbox("Show raw tables (advanced)", value=False)
    show_source_breakdown = st.checkbox("Show headline source breakdown", value=True)

st.sidebar.caption(
    "Tip: Cache reduces repeated requests and lowers ban risk. "
    "Demo mode forces offline trends data so the UI never breaks."
)

st.sidebar.markdown("---")

# Watchlist display
st.sidebar.markdown("### Watchlist")
watch_items = watch_store.list_all()
if not watch_items:
    st.sidebar.caption("No items yet. Run an analysis and add.")
else:
    wdf = pd.DataFrame(
        [
            {
                "keyword": w.keyword,
                "region": w.region,
                "last_score": w.last_score,
                "last_conf": w.last_confidence,
                "last_run_utc": w.last_analyzed_utc,
            }
            for w in watch_items[:10]
        ]
    )
    st.sidebar.dataframe(wdf, use_container_width=True, hide_index=True)


# -----------------------------
# Input Area
# -----------------------------
st.markdown("### Input")
if mode == "Single Keyword":
    kw1 = st.text_input(
        "Enter a product / market keyword",
        value="",
        placeholder="e.g., shampoo, electric scooter, AI skincare",
    )
    keywords, errors = validate_keywords([kw1], max_keywords=1)
else:
    c1, c2, c3 = st.columns(3)
    with c1:
        k1_in = st.text_input("Keyword 1", value="", placeholder="e.g., shampoo")
    with c2:
        k2_in = st.text_input("Keyword 2", value="", placeholder="e.g., soap")
    with c3:
        k3_in = st.text_input("Keyword 3", value="", placeholder="e.g., toothbrush")
    keywords, errors = validate_keywords([k1_in, k2_in, k3_in], max_keywords=3)

for e in errors:
    st.error(e)

run = st.button("Run Trend Analysis", type="primary", disabled=(len(errors) > 0))


# -----------------------------
# Helpers (UI)
# -----------------------------
def _status_badge(label: str) -> None:
    s = (label or "").lower()
    if "strong" in s or "high" in s:
        st.success(label)
    elif "moderate" in s or "medium" in s:
        st.warning(label)
    else:
        st.error(label)


def _confidence_badge(conf_label: str, conf_value: float) -> None:
    txt = f"{conf_label} ({conf_value:.0f}/100)"
    if conf_label == "High":
        st.success(txt)
    elif conf_label == "Medium":
        st.warning(txt)
    else:
        st.error(txt)


def _derive_trends_status(tmeta: dict) -> str:
    """
    Returns one of: OK / Rate-limited / Using cache / Demo / Unavailable
    """
    if not isinstance(tmeta, dict):
        return "Unavailable"
    if tmeta.get("demo_mode"):
        return "Demo"
    if tmeta.get("rate_limited") or tmeta.get("circuit_open"):
        return "Rate-limited"
    if tmeta.get("cached") or tmeta.get("stale") or tmeta.get("stale_fallback"):
        return "Using cache"
    return "OK"


def _pill_html(text: str, *, dot_color: str) -> str:
    return f"""
    <span class="om-pill ui-pill">
      <span class="dot" style="background:{dot_color};"></span>
      <span>{text}</span>
    </span>
    """


def _render_trends_status_pills(tmeta: dict) -> None:
    """
    Premium, compact status pills shown near the top of the page.
    Uses utils/ui.py if available; otherwise uses HTML pills.
    """
    status = _derive_trends_status(tmeta)

    if status_pill:
        try:
            st.markdown(
                " ".join(
                    [
                        status_pill(
                            "Trends",
                            status,
                            tone=(
                                "success"
                                if status == "OK"
                                else "info"
                                if status == "Using cache"
                                else "warning"
                                if status == "Demo"
                                else "danger"
                            ),
                        ),
                        status_pill(
                            "Cache",
                            "HIT" if (isinstance(tmeta, dict) and tmeta.get("cached")) else ("ON" if use_cache else "OFF"),
                            tone=("info" if use_cache else "muted"),
                        ),
                        status_pill("Demo", "ON" if demo_mode else "OFF", tone=("warning" if demo_mode else "muted")),
                    ]
                ),
                unsafe_allow_html=True,
            )
            return
        except Exception:
            pass

    parts = []
    if status == "OK":
        parts.append(_pill_html("Trends: OK", dot_color="#16a34a"))
    elif status == "Using cache":
        parts.append(_pill_html("Trends: Using cache", dot_color="#2563eb"))
    elif status == "Demo":
        parts.append(_pill_html("Trends: Demo", dot_color="#f59e0b"))
    elif status == "Rate-limited":
        parts.append(_pill_html("Trends: Rate-limited", dot_color="#ef4444"))
    else:
        parts.append(_pill_html("Trends: Unavailable", dot_color="#ef4444"))

    if isinstance(tmeta, dict) and tmeta.get("cached"):
        parts.append(_pill_html("Cache: HIT", dot_color="#2563eb"))
    elif use_cache:
        parts.append(_pill_html("Cache: ON", dot_color="#93c5fd"))
    else:
        parts.append(_pill_html("Cache: OFF", dot_color="#9ca3af"))

    if demo_mode:
        parts.append(_pill_html("Demo mode: ON", dot_color="#f59e0b"))

    st.markdown("".join(parts), unsafe_allow_html=True)


def _show_trends_meta_warnings(tmeta: dict) -> None:
    """
    Surface Trends failures (like 429) clearly for non-technical users.
    """
    if not isinstance(tmeta, dict):
        return

    status = _derive_trends_status(tmeta)
    if status == "Demo":
        (info_callout or st.info)("Using demo (offline) Trends dataset so the app continues to work.")
    elif status == "Using cache":
        (info_callout or st.info)("Using cached Trends data to avoid repeated requests.")
    elif status == "Rate-limited":
        (warn_callout or st.warning)(
            "Google Trends temporarily rate-limited requests (HTTP 429). "
            "We are using cache/demo fallback so the analysis continues."
        )

    err = (tmeta.get("error") or "").strip()
    if err and status not in ("Rate-limited", "Demo"):
        (warn_callout or st.warning)(f"Google Trends issue: {err}")

    if tmeta.get("stale_fallback"):
        st.caption("Fallback used (cached/demo) due to temporary Trends unavailability.")


def _render_scenarios_block(forecast) -> None:
    scenarios = getattr(forecast, "scenarios", None) or {}
    if not scenarios:
        st.caption("No scenario projections available.")
        return

    if scenario_table:
        try:
            scenario_table(scenarios)
            st.caption("Scenarios are illustrative projections, not guaranteed outcomes.")
            return
        except Exception:
            pass

    rows = []
    for key in ["best", "base", "worst"]:
        if key not in scenarios:
            continue
        s = scenarios[key]
        rows.append(
            {
                "Scenario": s.get("label", key.title()),
                "Horizon (steps)": s.get("horizon_steps", ""),
                "Projected (0-100)": s.get("projected_value", ""),
            }
        )
    if rows:
        sdf = pd.DataFrame(rows)
        st.dataframe(sdf, use_container_width=True, hide_index=True)
        st.caption("Scenarios are illustrative projections, not guaranteed outcomes.")


def _render_exec_summary_card(
    *,
    decision: str,
    pro_score: float,
    conf_label: str,
    conf_value: float,
    forecast_text: str,
) -> None:
    if exec_summary_card:
        try:
            exec_summary_card(
                decision=decision,
                pro_score=pro_score,
                confidence_label=conf_label,
                confidence_value=conf_value,
                forecast_text=forecast_text,
            )
            return
        except Exception:
            pass

    st.markdown(
        f"""
        <div class="om-card ui-card">
          <div style="display:flex; align-items:baseline; justify-content:space-between; gap:12px;">
            <div>
              <div style="opacity:.75; font-size:.9rem;">Executive Summary</div>
              <div style="font-size:1.15rem; margin-top:.15rem;"><b>Decision:</b> {decision}</div>
            </div>
            <div style="text-align:right; opacity:.85; font-size:.92rem;">
              <div><b>Pro Score:</b> {pro_score:.2f}/100</div>
              <div><b>Confidence:</b> {conf_label} ({conf_value:.0f}/100)</div>
              <div><b>Forecast:</b> {forecast_text}</div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _section(title: str, hint: str | None = None) -> None:
    if section_title:
        try:
            section_title(title, hint=hint)
            return
        except Exception:
            pass
    st.markdown(f"### {title}")
    if hint:
        st.caption(hint)


def _norm_kw(s: str) -> str:
    # Normalize to match trends service clean_keyword() behavior (case-insensitive, trimmed)
    return " ".join((s or "").strip().split())


def _run_key(mode_label: str, kws: list[str]) -> str:
    # Stable key used to prevent re-fetch loops across reruns
    return "|".join(
        [
            "v2",
            mode_label,
            region_label,
            timeframe_label,
            news_window_label,
            context_label,
            "cache=1" if use_cache else "cache=0",
            "demo=1" if demo_mode else "demo=0",
            "kws=" + ",".join([_norm_kw(k).lower() for k in kws]),
        ]
    )


# -----------------------------
# Core runners
# -----------------------------
def render_single(keyword: str) -> None:
    keyword = _norm_kw(keyword)
    days = news_window_to_days(news_window_label)

    run_key = _run_key("single", [keyword])

    # Safe-mode: prevent repeated Trends calls on reruns by memoizing the result
    # We only refresh when the user clicks Run, or when inputs change (run_key changes).
    if "single_cache" not in st.session_state:
        st.session_state["single_cache"] = {}
    if run_key in st.session_state["single_cache"]:
        tdata = st.session_state["single_cache"][run_key]
    else:
        with st.spinner("Fetching Google Trends..."):
            tdata = fetch_trends_single(
                keyword,
                timeframe_label=timeframe_label,
                region_label=region_label,
                use_cache=use_cache,
                force_demo=demo_mode,
            )
        st.session_state["single_cache"][run_key] = tdata

    _render_trends_status_pills(tdata.get("meta", {}))
    _show_trends_meta_warnings(tdata.get("meta", {}))

    with st.spinner("Fetching News (GDELT)..."):
        ndata = fetch_news_gdelt(
            keyword,
            context_label=context_label,
            days=days,
            max_records=60,
            use_cache=use_cache,
        )

    with st.spinner("Computing Sentiment (VADER)..."):
        sdata = score_headlines_vader(keyword, ndata.get("headlines", []), use_cache=use_cache)

    ts = tdata.get("timeseries", []) or []
    metrics = compute_trend_metrics(ts)
    news_q = compute_news_quality(ndata.get("fetched_count", 0), ndata.get("matched_count", 0))
    adv = compute_advanced_trend_metrics(ts)

    cov = coverage_strength(
        fetched_count=int(ndata.get("fetched_count", 0)),
        matched_count=int(ndata.get("matched_count", 0)),
    )

    st_series = build_sentiment_timeseries(sdata.get("scored", []), bucket=sentiment_bucket)
    themes = extract_themes_from_headlines(ndata.get("headlines", []), max_themes=5)

    forecast = compute_forecast(
        ts,
        volatility=metrics.volatility,
        match_quality=news_q.match_quality,
        trend_points=metrics.points,
        matched_headlines=news_q.matched_count,
    )

    sentiment_0_100 = float(sdata.get("sentiment_0_100", 50.0))
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
        keyword=keyword,
        metrics=metrics,
        news_q=news_q,
        score=pro,
        forecast=forecast,
        region_label=region_label,
    )

    prev = watch_store.get(keyword=keyword, region=region_label.upper())
    delta = watch_store.compute_delta(keyword=keyword, region=region_label.upper(), new_score=pro.pro_score)

    _render_exec_summary_card(
        decision=rec.decision,
        pro_score=float(pro.pro_score),
        conf_label=str(conf.label),
        conf_value=float(conf.confidence),
        forecast_text=f"{forecast.direction} ({forecast.confidence})",
    )

    st.write("")

    add_watch = st.button("⭐ Add/Update Watchlist", key=f"watch_{keyword}")
    if add_watch:
        watch_store.upsert(
            keyword=keyword,
            region=region_label.upper(),
            score=float(pro.pro_score),
            confidence=float(conf.confidence),
        )
        st.success("Saved to watchlist.")

    st.subheader(f"Results for: **{keyword}**")

    # KPI rows
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Pro Trend Score", f"{pro.pro_score:.2f}/100", delta=(f"{delta:+.2f}" if delta is not None else None))
    k2.metric("Confidence", f"{conf.confidence:.0f}/100", help="How reliable this decision is")
    k3.metric("Search Growth %", f"{metrics.growth_pct:.2f}%")
    k4.metric("Momentum %", f"{adv.momentum_pct:.2f}%", help="Recent vs past interest")
    k5.metric("Sentiment (0–100)", f"{sentiment_0_100:.2f}")
    k6.metric("Forecast", f"{forecast.direction} ({forecast.confidence})")

    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Consistency", f"{adv.consistency_score:.0f}/100", help="How often it rises week-to-week")
    s2.metric("Volatility", f"{metrics.volatility:.2f}", help="Spikiness in the trend series")
    s3.metric("Spikes", f"{adv.spike_index} ({adv.spike_label})")
    s4.metric("Coverage Strength", cov.label, help=f"Matched {cov.matched_headlines}/{cov.fetched_headlines}")

    with st.expander("Data trust & notes", expanded=False):
        _confidence_badge(conf.label, conf.confidence)
        if getattr(conf, "explanation", ""):
            st.write(f"**Confidence summary:** {conf.explanation}")
        if getattr(forecast, "explanation", ""):
            st.write(f"**Forecast summary:** {forecast.explanation}")

        dc = getattr(conf, "data_coverage", None) or {}
        if dc:
            st.write("**Data coverage:** " + ", ".join([f"{k}={v}" for k, v in dc.items()]))

        for r in conf.reasons:
            st.write(f"• {r}")
        for n in (adv.notes or []):
            st.write(f"• {n}")
        for w in (cov.notes or []):
            st.write(f"• {w}")
        for w in (news_q.warnings or []):
            st.warning(w)
        for n in (getattr(forecast, "notes", None) or []):
            st.warning(n)

    tab_overview, tab_trends, tab_news, tab_forecast, tab_score, tab_export = st.tabs(
        ["Overview", "Demand Trend", "Market Sentiment", "Forecast", "Why This Score?", "Export"]
    )

    with tab_overview:
        c1, c2 = st.columns([2, 1], gap="large")
        with c1:
            _section("Decision Summary")
            st.markdown(f"**Classification:** `{pro.classification}`")
            st.markdown("**Why:**")
            for n in pro.notes:
                st.write(f"• {n}")

            _section("Business Recommendation")
            st.markdown(f"**Decision:** **{rec.decision}**")
            st.markdown(f"**Hype vs Sustainable:** {rec.hype_label}")
            st.markdown(f"**Ideal Entry Window:** {rec.entry_window}")

            st.markdown("#### Recommendations")
            for b in rec.bullets:
                st.write(f"• {b}")
            if rec.cautions:
                st.markdown("#### Cautions")
                for c in rec.cautions:
                    st.write(f"• {c}")

        with c2:
            _section("Quick Badges")
            _status_badge(f"Trend: {pro.classification}")
            _status_badge(f"Coverage: {cov.label}")
            _status_badge(f"Confidence: {conf.label}")

            if prev and prev.last_score is not None and delta is not None:
                _section("Watchlist Delta")
                st.write(f"Previous score: `{prev.last_score:.2f}`")
                st.write(f"Current score: `{pro.pro_score:.2f}`")
                st.write(f"Change: `{delta:+.2f}`")

    with tab_trends:
        _section("Google Trends demand curve", hint="Interest over time (0–100).")

        if not ts:
            st.error("No Google Trends data returned for this keyword.")
        else:
            df = pd.DataFrame(ts)
            df["date"] = pd.to_datetime(df["date"])

            if trend_line_chart:
                try:
                    trend_line_chart(df, height=chart_height, y_col="value", title=None)
                except Exception:
                    st.line_chart(df.set_index("date")["value"], height=chart_height)
            else:
                st.line_chart(df.set_index("date")["value"], height=chart_height)

            if show_raw_tables:
                with st.expander("Raw Trend Data"):
                    st.dataframe(df, use_container_width=True)

        # Related queries are optional; do NOT re-fetch repeatedly on reruns.
        rq_key = _run_key("related", [keyword])
        if "related_cache" not in st.session_state:
            st.session_state["related_cache"] = {}
        if rq_key in st.session_state["related_cache"]:
            rq = st.session_state["related_cache"][rq_key]
        else:
            rq = fetch_related_queries(keyword, region_label=region_label, use_cache=use_cache, force_demo=demo_mode)
            st.session_state["related_cache"][rq_key] = rq

        _show_trends_meta_warnings(rq.get("meta", {}))

        colA, colB = st.columns(2)
        with colA:
            st.markdown("**Top related queries**")
            st.table(pd.DataFrame(rq.get("top", [])))
        with colB:
            st.markdown("**Rising related queries**")
            st.table(pd.DataFrame(rq.get("rising", [])))

    with tab_news:
        _section("Sentiment distribution", hint="How positive/neutral/negative the matched headlines are.")

        counts = sdata.get("counts", {})
        cdf = pd.DataFrame(
            [
                {"label": "positive", "count": counts.get("positive", 0)},
                {"label": "neutral", "count": counts.get("neutral", 0)},
                {"label": "negative", "count": counts.get("negative", 0)},
            ]
        )

        if sentiment_donut_or_bar:
            try:
                sentiment_donut_or_bar(cdf, height=260)
            except Exception:
                st.bar_chart(cdf.set_index("label")["count"], height=240)
        else:
            st.bar_chart(cdf.set_index("label")["count"], height=240)

        if st_series.rows:
            _section(f"Sentiment over time ({st_series.bucket})", hint="Average sentiment (0–100) over each period.")
            sdf = pd.DataFrame(st_series.rows)
            st.line_chart(sdf.set_index("period")["avg_sentiment_0_100"], height=min(chart_height, 360))
        else:
            st.info("Not enough scored headlines to plot sentiment over time.")

        st.markdown("---")
        _section("Top Themes (from headlines)")
        if themes.get("themes"):
            for t in themes["themes"]:
                with st.expander(f"Theme: {t.get('theme','')}", expanded=False):
                    st.write("**Keywords:** " + ", ".join(t.get("keywords", [])))
                    for s in t.get("sample_titles", []):
                        st.write(f"• {s}")
        else:
            st.caption(themes.get("note", "No themes available."))

        if show_source_breakdown and headline_source_bar:
            try:
                hdf_for_sources = pd.DataFrame(ndata.get("headlines", []))
                if not hdf_for_sources.empty:
                    st.markdown("---")
                    _section("Headline sources", hint="Where matched coverage is coming from.")
                    headline_source_bar(hdf_for_sources, height=260)
            except Exception:
                pass

        st.markdown("### Matched headlines used for sentiment")
        hdf = pd.DataFrame(ndata.get("headlines", []))
        if hdf.empty:
            st.info("No matched headlines found. Try changing context or using a more specific keyword.")
        else:
            cols = [c for c in ["title", "source", "date", "url"] if c in hdf.columns]
            st.dataframe(hdf[cols], use_container_width=True)

        st.markdown("### Scored headlines")
        scdf = pd.DataFrame(sdata.get("scored", []))
        if scdf.empty:
            st.info("No headlines were scored.")
        else:
            cols = [c for c in ["label", "compound", "title", "source", "date", "url"] if c in scdf.columns]
            st.dataframe(scdf[cols], use_container_width=True)

    with tab_forecast:
        _section("Forecast curve", hint="Smoothed trend line and directional outlook.")

        if not getattr(forecast, "smoothed", None):
            st.info("Not enough data to show forecast smoothing.")
        else:
            fdf = pd.DataFrame(forecast.smoothed)
            fdf["date"] = pd.to_datetime(fdf["date"])

            if trend_line_chart:
                try:
                    trend_line_chart(fdf, height=chart_height, y_col="value", title=None)
                except Exception:
                    st.line_chart(fdf.set_index("date")["value"], height=chart_height)
            else:
                st.line_chart(fdf.set_index("date")["value"], height=chart_height)

        st.markdown(
            f"**Slope:** `{forecast.slope}`  \n"
            f"**Direction:** `{forecast.direction}`  \n"
            f"**Confidence:** `{forecast.confidence}`"
        )

        if getattr(forecast, "explanation", ""):
            st.info(forecast.explanation)

        st.markdown("### Best / Base / Worst Scenarios")
        _render_scenarios_block(forecast)

    with tab_score:
        _section("Pro Score components", hint="How the 0–100 score is composed.")
        comp = pro.components.copy()
        comp_vis = {
            "growth": comp.get("growth_score", 0.0),
            "momentum": comp.get("momentum_score", 0.0),
            "sentiment": comp.get("sentiment_score", 0.0),
            "consistency": comp.get("consistency_score", 0.0),
            "volatility_penalty (subtract)": -comp.get("volatility_penalty", 0.0),
        }
        comp_df = pd.DataFrame({"component": list(comp_vis.keys()), "value": list(comp_vis.values())})

        if score_components_chart:
            try:
                score_components_chart(comp_df, height=320)
            except Exception:
                st.bar_chart(comp_df.set_index("component")["value"], height=280)
        else:
            st.bar_chart(comp_df.set_index("component")["value"], height=280)

        st.markdown("### Notes")
        for n in pro.notes:
            st.write(f"• {n}")

    with tab_export:
        _section("Download analysis outputs", hint="Export CSVs + full JSON payload for pipelines.")
        analysis_blob = {
            "keyword": keyword,
            "region": region_label,
            "timeframe": timeframe_label,
            "news_window_days": days,
            "context": context_label,
            "trend_timeseries": ts,
            "trend_meta": tdata.get("meta", {}),
            "trend_metrics": {
                "growth_pct": metrics.growth_pct,
                "volatility": metrics.volatility,
                "points": metrics.points,
            },
            "advanced_metrics": {
                "momentum_pct": adv.momentum_pct,
                "consistency_score": adv.consistency_score,
                "spike_index": adv.spike_index,
                "spike_label": adv.spike_label,
                "recent_slope": adv.recent_slope,
                "long_slope": adv.long_slope,
                "acceleration": adv.acceleration,
            },
            "news_quality": {
                "matched_count": news_q.matched_count,
                "match_quality": news_q.match_quality,
                "coverage_label": cov.label,
            },
            "sentiment": {
                "sentiment_0_100": sentiment_0_100,
                "counts": sdata.get("counts", {}),
                "timeseries": st_series.rows,
            },
            "themes": themes,
            "pro_score": {
                "score": pro.pro_score,
                "classification": pro.classification,
                "components": pro.components,
                "notes": pro.notes,
            },
            "confidence": {
                "confidence": conf.confidence,
                "label": conf.label,
                "reasons": conf.reasons,
                "data_coverage": getattr(conf, "data_coverage", {}),
                "explanation": getattr(conf, "explanation", ""),
            },
            "forecast": {
                "slope": forecast.slope,
                "direction": forecast.direction,
                "confidence": forecast.confidence,
                "scenarios": getattr(forecast, "scenarios", {}),
                "explanation": getattr(forecast, "explanation", ""),
            },
            "recommendation": {
                "decision": rec.decision,
                "hype_label": rec.hype_label,
                "entry_window": rec.entry_window,
                "bullets": rec.bullets,
                "cautions": rec.cautions,
            },
            "generated_at_utc": utc_now_iso(),
            "demo_mode": bool(demo_mode),
        }

        st.download_button(
            "⬇️ Download Trend Timeseries (CSV)",
            data=trends_timeseries_to_csv_bytes(ts),
            file_name=f"{keyword}_trend_timeseries.csv".replace(" ", "_"),
            mime="text/csv",
        )
        st.download_button(
            "⬇️ Download Headlines (CSV)",
            data=headlines_to_csv_bytes(ndata.get("headlines", [])),
            file_name=f"{keyword}_headlines.csv".replace(" ", "_"),
            mime="text/csv",
        )
        st.download_button(
            "⬇️ Download Full Analysis (JSON)",
            data=analysis_to_json_bytes(analysis_blob),
            file_name=f"{keyword}_analysis.json".replace(" ", "_"),
            mime="application/json",
        )


def render_compare(keywords: list[str]) -> None:
    # normalize & dedupe while preserving order
    normed = []
    seen = set()
    for k in keywords:
        nk = _norm_kw(k)
        if not nk:
            continue
        lk = nk.lower()
        if lk in seen:
            continue
        seen.add(lk)
        normed.append(nk)
    keywords = normed[:3]

    days = news_window_to_days(news_window_label)

    run_key = _run_key("compare", keywords)

    if "compare_cache" not in st.session_state:
        st.session_state["compare_cache"] = {}

    if run_key in st.session_state["compare_cache"]:
        tcmp = st.session_state["compare_cache"][run_key]
    else:
        with st.spinner("Fetching Google Trends (compare mode)..."):
            tcmp = fetch_trends_compare(
                keywords,
                timeframe_label=timeframe_label,
                region_label=region_label,
                use_cache=use_cache,
                force_demo=demo_mode,
            )
        st.session_state["compare_cache"][run_key] = tcmp

    _render_trends_status_pills(tcmp.get("meta", {}))
    _show_trends_meta_warnings(tcmp.get("meta", {}))

    st.subheader("Comparison Overview")

    series = tcmp.get("series", {}) or {}
    if not series:
        st.error("No comparison trend data returned.")
        return

    # IMPORTANT: the service returns series using cleaned keywords.
    # We normalize by lower-match mapping so user-entered casing doesn't break lookups.
    series_by_lower = {str(k).lower(): v for k, v in series.items()}

    dfs = []
    label_map = []  # [(display_label, series_key_lower)]
    for kw in keywords:
        lk = kw.lower()
        ts = series_by_lower.get(lk, []) or []
        df = pd.DataFrame(ts)
        if df.empty:
            continue
        df["date"] = pd.to_datetime(df["date"])
        df = df.rename(columns={"value": kw})  # display with user's keyword casing
        dfs.append(df[["date", kw]])
        label_map.append((kw, lk))

    if not dfs:
        st.error("No valid timeseries to compare.")
        return

    merged = dfs[0]
    for d in dfs[1:]:
        merged = merged.merge(d, on="date", how="inner")
    merged = merged.sort_values("date")

    if compare_trends_chart:
        try:
            compare_trends_chart(merged, height=chart_height)
        except Exception:
            st.line_chart(merged.set_index("date"), height=min(chart_height + 20, 420))
    else:
        st.line_chart(merged.set_index("date"), height=min(chart_height + 20, 420))

    rows = []
    per_keyword_blocks = []

    # Use the *actual* series returned for each keyword (avoid mismatch causing blank panels)
    for display_kw, lk in label_map:
        ts = series_by_lower.get(lk, []) or []

        metrics = compute_trend_metrics(ts)
        adv = compute_advanced_trend_metrics(ts)

        ndata = fetch_news_gdelt(
            display_kw,
            context_label=context_label,
            days=days,
            max_records=60,
            use_cache=use_cache,
        )
        sdata = score_headlines_vader(display_kw, ndata.get("headlines", []), use_cache=use_cache)
        news_q = compute_news_quality(ndata.get("fetched_count", 0), ndata.get("matched_count", 0))
        cov = coverage_strength(
            fetched_count=int(ndata.get("fetched_count", 0)),
            matched_count=int(ndata.get("matched_count", 0)),
        )

        sentiment_0_100 = float(sdata.get("sentiment_0_100", 50.0))

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
            keyword=display_kw,
            metrics=metrics,
            news_q=news_q,
            score=pro,
            forecast=forecast,
            region_label=region_label,
        )

        rows.append(
            {
                "Keyword": display_kw,
                "Pro Score": pro.pro_score,
                "Confidence": conf.confidence,
                "Growth %": metrics.growth_pct,
                "Momentum %": adv.momentum_pct,
                "Sentiment (0-100)": sentiment_0_100,
                "Coverage": cov.label,
                "Forecast": f"{forecast.direction} ({forecast.confidence})",
                "Class": pro.classification,
                "Decision": rec.decision,
            }
        )

        per_keyword_blocks.append((display_kw, metrics, adv, sdata, news_q, cov, pro, conf, forecast, rec, ndata))

    summary = pd.DataFrame(rows).sort_values(["Pro Score", "Confidence"], ascending=False)
    st.dataframe(summary, use_container_width=True)

    best = summary.iloc[0]["Keyword"] if not summary.empty else None
    if best:
        st.success(f"Highest future-demand candidate (by Pro Score): **{best}**")

    with st.expander("Per-keyword details"):
        for (kw, metrics, adv, sdata, news_q, cov, pro, conf, forecast, rec, ndata) in per_keyword_blocks:
            st.markdown(f"## {kw}")

            k1, k2, k3, k4, k5 = st.columns(5)
            k1.metric("Pro Score", f"{pro.pro_score:.2f}/100")
            k2.metric("Confidence", f"{conf.confidence:.0f}/100")
            k3.metric("Growth %", f"{metrics.growth_pct:.2f}%")
            k4.metric("Momentum %", f"{adv.momentum_pct:.2f}%")
            k5.metric("Sentiment", f"{float(sdata.get('sentiment_0_100', 50.0)):.1f}/100")

            if getattr(conf, "explanation", ""):
                st.caption(f"Confidence: {conf.explanation}")
            if getattr(forecast, "explanation", ""):
                st.caption(f"Forecast: {forecast.explanation}")

            st.write(f"**Coverage:** {cov.label}  |  **Forecast:** {forecast.direction} ({forecast.confidence})")
            st.write(
                f"**Decision:** {rec.decision}  |  **Entry Window:** {rec.entry_window}  |  "
                f"**Hype vs Sustainable:** {rec.hype_label}"
            )

            st.markdown("### Scenarios")
            _render_scenarios_block(forecast)

            if rec.cautions:
                st.write("**Cautions:**")
                for c in rec.cautions:
                    st.write(f"• {c}")
            st.markdown("---")


# -----------------------------
# Run
# -----------------------------
if run:
    if mode == "Single Keyword":
        render_single(keywords[0])
    else:
        render_compare(keywords)
else:
    st.info(
        "Enter keyword(s) above, choose Region/Timeframe/Context on the left, "
        "then click **Run Trend Analysis**."
    )

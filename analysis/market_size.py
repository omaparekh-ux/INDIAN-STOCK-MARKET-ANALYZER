"""
Market size estimator (non-technical, money-first).

IMPORTANT:
True market size usually requires paid industry reports or proprietary datasets.
This module provides a best-effort, honest estimator using simple heuristics,
and always includes clear assumptions/limitations (so we don't overclaim).

Output is designed for non-technical users: "₹X–₹Y Cr", not TAM/SAM/SOM jargon.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from schemas.dashboard_schema import MarketSize


# -----------------------------
# Heuristic baselines (India-focused defaults)
# You can expand these over time as you add better data.
# Values are intentionally ranges and conservative.
# -----------------------------

_BASELINES_INR_CR: Dict[str, Tuple[int, int]] = {
    # Personal care
    "shampoo": (8000, 15000),
    "conditioner": (1200, 3000),
    "hair oil": (12000, 22000),
    "face wash": (4000, 9000),
    "soap": (18000, 35000),
    "body wash": (2000, 6000),

    # FMCG / food
    "protein powder": (2500, 7000),
    "whey protein": (2500, 7000),
    "energy drink": (1800, 4500),
    "snacks": (25000, 70000),

    # Home / fragrance (relevant to your domain)
    "incense sticks": (8000, 20000),
    "agarbatti": (8000, 20000),
    "dhoop": (2500, 7000),
    "diffuser oil": (800, 2500),
    "reed diffuser": (400, 1500),
    "air freshener": (2500, 8000),

    # Default catch-all
    "_default": (500, 5000),
}


def _normalize_product_type(product_type: str) -> str:
    return " ".join(product_type.strip().lower().split())


def _format_inr_cr(lo: float, hi: float) -> str:
    # Round nicely for humans
    lo_i = int(round(lo))
    hi_i = int(round(hi))
    return f"₹{lo_i:,}–{hi_i:,} Cr"


def _format_inr_range(lo_lakh: float, hi_lakh: float) -> str:
    lo_i = int(round(lo_lakh))
    hi_i = int(round(hi_lakh))
    return f"₹{lo_i}–{hi_i} Lakhs/month"


def _pick_baseline(product_type_norm: str) -> Tuple[int, int]:
    # Exact key match
    if product_type_norm in _BASELINES_INR_CR:
        return _BASELINES_INR_CR[product_type_norm]

    # Soft match (contains)
    for k, v in _BASELINES_INR_CR.items():
        if k != "_default" and k in product_type_norm:
            return v

    return _BASELINES_INR_CR["_default"]


def estimate_market_size(
    product_type: str,
    geo: str = "IN",
    demand_signal: Optional[Dict[str, float]] = None,
    price_signal: Optional[Dict[str, float]] = None,
) -> MarketSize:
    """
    Best-effort market size estimator.

    Parameters
    ----------
    product_type:
        User input product category (e.g., "shampoo").
    geo:
        Region code ("IN" default). For now, baselines are India-focused.
    demand_signal:
        Optional metrics such as growth_rate_pct, momentum, or normalized interest score.
        Example: {"growth_rate_pct": 12.5, "momentum": 0.7}
    price_signal:
        Optional pricing hints.
        Example: {"median_price": 349.0}

    Returns
    -------
    MarketSize (dataclass)
        Money-first explanation with honest assumptions.
    """
    pt = _normalize_product_type(product_type)
    baseline_lo, baseline_hi = _pick_baseline(pt)

    # Mild adjustments based on demand growth (keep conservative)
    growth_pct = None
    if demand_signal:
        growth_pct = demand_signal.get("growth_rate_pct")

    adj = 1.0
    if isinstance(growth_pct, (int, float)):
        # Cap adjustment to avoid overclaiming
        if growth_pct >= 25:
            adj = 1.15
        elif growth_pct >= 10:
            adj = 1.08
        elif growth_pct <= -10:
            adj = 0.92
        elif growth_pct <= -25:
            adj = 0.85

    est_lo = baseline_lo * adj
    est_hi = baseline_hi * adj

    # Realistic "new brand" revenue range (very rough, non-technical)
    # Use category competitiveness heuristics if available later.
    # For now: assume a small entrant targets a tiny slice of market.
    # Example: 0.02%–0.08% of annual market value.
    new_brand_lo_cr = est_lo * 0.0002
    new_brand_hi_cr = est_hi * 0.0008

    # Convert to a friendlier statement (monthly lakhs) as well in explanation
    # 1 Cr/year = ~8.33 Lakhs/month
    new_brand_lo_lakh_m = new_brand_lo_cr * 8.33
    new_brand_hi_lakh_m = new_brand_hi_cr * 8.33

    assumptions: List[str] = [
        "Market size is an approximate range based on category baselines and demand momentum, not a paid industry report.",
        "The estimate is best suited for early decision-making (Enter/Test/Wait) rather than investor-grade valuation.",
        "Results can vary based on channel (online vs offline), price tier (mass vs premium), and regional focus.",
    ]

    explanation = (
        f"This is a best-effort estimate for {product_type} in {geo}. "
        f"For a new brand, a realistic early target could be around "
        f"{_format_inr_cr(new_brand_lo_cr, new_brand_hi_cr)} per year "
        f"(roughly {_format_inr_range(new_brand_lo_lakh_m, new_brand_hi_lakh_m)}), "
        "depending on execution, differentiation, and distribution."
    )

    return MarketSize(
        estimated_annual_value=_format_inr_cr(est_lo, est_hi),
        realistic_new_brand_range=_format_inr_cr(new_brand_lo_cr, new_brand_hi_cr),
        explanation=explanation,
        assumptions=assumptions,
    )

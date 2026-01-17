"""
Explainability engine for the AI Market Trend Analyzer.

This module answers the non-technical question:
"Why did the system give this verdict and confidence?"

It converts analytical signals into plain-English reasons.
"""

from typing import Dict, List, Tuple


# -----------------------------
# Helper utilities
# -----------------------------

def _label_strength(value: float) -> str:
    """
    Converts a normalized metric (0–1 or -1–1) into a human label.
    """
    if value >= 0.6:
        return "strong"
    if value >= 0.2:
        return "moderate"
    if value <= -0.6:
        return "weak"
    if value <= -0.2:
        return "concerning"
    return "mixed"


# -----------------------------
# Core explainability
# -----------------------------

def explain_verdict_drivers(
    metrics: Dict[str, float],
    verdict: str,
) -> Tuple[List[str], List[str]]:
    """
    Returns (positive_drivers, negative_drivers) in plain English.

    Expected metrics keys (if available):
    - demand_momentum
    - growth_rate
    - sentiment_score
    - competition_intensity
    - price_pressure
    - trust_risk
    """
    positives: List[str] = []
    negatives: List[str] = []

    # Demand & growth
    dm = metrics.get("demand_momentum")
    if isinstance(dm, (int, float)):
        strength = _label_strength(dm)
        if dm > 0:
            positives.append(f"Demand momentum is {strength}")
        else:
            negatives.append(f"Demand momentum is {strength}")

    gr = metrics.get("growth_rate")
    if isinstance(gr, (int, float)):
        if gr >= 10:
            positives.append("Category is growing at a healthy pace")
        elif gr <= -5:
            negatives.append("Category growth is slowing or declining")

    # Sentiment
    ss = metrics.get("sentiment_score")
    if isinstance(ss, (int, float)):
        if ss >= 0.6:
            positives.append("Consumer sentiment is largely positive")
        elif ss <= 0.4:
            negatives.append("Consumer sentiment shows dissatisfaction")

    # Competition
    ci = metrics.get("competition_intensity")
    if isinstance(ci, (int, float)):
        if ci >= 0.7:
            negatives.append("Competition is intense with many similar offerings")
        elif ci <= 0.4:
            positives.append("Competition is manageable for new entrants")

    # Pricing pressure
    pp = metrics.get("price_pressure")
    if isinstance(pp, (int, float)):
        if pp >= 0.7:
            negatives.append("Strong price pressure may reduce margins")
        elif pp <= 0.4:
            positives.append("Pricing environment allows reasonable margins")

    # Trust risk
    tr = metrics.get("trust_risk")
    if isinstance(tr, (int, float)) and tr >= 0.6:
        negatives.append("Trust and authenticity concerns exist in this category")

    # Fallback if sparse
    if not positives and not negatives:
        positives.append("Overall signals are balanced without extreme risks")

    return positives, negatives


def explain_confidence(
    data_coverage: Dict[str, str],
) -> str:
    """
    Explains confidence level in plain English.
    """
    strong = sum(1 for v in data_coverage.values() if v == "strong")
    weak = sum(1 for v in data_coverage.values() if v == "weak")

    if strong >= 3 and weak == 0:
        return "Multiple data sources strongly support this conclusion."
    if weak >= 2:
        return "Some important data sources are limited or weak, reducing confidence."
    return "The conclusion is supported by available data, but with some uncertainty."


def explain_biggest_risk(
    metrics: Dict[str, float],
) -> str:
    """
    Identifies the single most important risk in plain English.
    """
    # Priority order for risks
    if metrics.get("competition_intensity", 0) >= 0.7:
        return "High competition may make differentiation and customer acquisition difficult."
    if metrics.get("price_pressure", 0) >= 0.7:
        return "Aggressive pricing could squeeze margins."
    if metrics.get("trust_risk", 0) >= 0.6:
        return "Trust and authenticity issues could affect repeat purchases."
    if metrics.get("growth_rate", 0) <= -5:
        return "Slowing demand may limit growth potential."

    return "Execution quality will play a major role in success."

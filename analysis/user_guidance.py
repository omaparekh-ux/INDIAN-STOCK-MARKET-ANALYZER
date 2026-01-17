"""
Non-technical, action-oriented guidance for users of the
AI Market Trend Analyzer.

This module converts analytical signals into clear,
human-friendly advice.
"""

from typing import List, Dict


# -----------------------------
# Focus areas
# -----------------------------

def get_focus_areas(
    competition_intensity: str,
    price_pressure: str,
    trust_risks: List[str],
) -> List[str]:
    """
    Suggests where the user should focus effort and money.
    """
    focus = []

    if competition_intensity == "High":
        focus.append("Strong branding and clear differentiation")
    else:
        focus.append("Speed to market and visibility")

    if price_pressure in {"Aggressive", "Increasing"}:
        focus.append("Cost control and supply-chain efficiency")
    else:
        focus.append("Value-based pricing and positioning")

    if trust_risks:
        focus.append("Trust-building (authentic claims, reviews, transparency)")

    return focus


# -----------------------------
# Common mistakes
# -----------------------------

def get_common_mistakes(product_type: str) -> List[str]:
    """
    Typical mistakes observed across product categories.
    """
    return [
        "Entering the market without a clear point of differentiation",
        "Competing only on price and eroding margins",
        "Overpromising benefits that the product cannot consistently deliver",
        "Underestimating marketing and distribution costs",
        "Ignoring early customer feedback and complaints"
    ]


# -----------------------------
# Who should NOT enter
# -----------------------------

def get_who_should_not_enter(
    competition_intensity: str,
    entry_barrier: str,
) -> List[str]:
    """
    Helps users self-filter if the market is unsuitable for them.
    """
    warnings = []

    if competition_intensity == "High":
        warnings.append("Founders with very limited branding or marketing budget")

    if entry_barrier == "High":
        warnings.append("First-time entrepreneurs without industry experience")

    warnings.append("Anyone expecting quick profits without sustained effort")

    return warnings


# -----------------------------
# Execution difficulty
# -----------------------------

def get_execution_difficulty(
    competition_intensity: str,
    trust_risks: List[str],
) -> str:
    """
    Classifies execution difficulty in plain language.
    """
    if competition_intensity == "High" or trust_risks:
        return "Hard"
    if competition_intensity == "Medium":
        return "Moderate"
    return "Easy"


# -----------------------------
# Startup budget estimate
# -----------------------------

def estimate_startup_budget(
    competition_intensity: str,
    execution_difficulty: str,
) -> str:
    """
    Provides a rough, non-technical startup budget range.
    """
    if execution_difficulty == "Hard":
        return "₹20–50 Lakhs"
    if execution_difficulty == "Moderate":
        return "₹8–20 Lakhs"
    return "₹3–8 Lakhs"

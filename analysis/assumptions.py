"""
Centralized assumptions, limitations, and ethical/compliance notes
for the AI Market Trend Analyzer.

This module ensures transparency and builds trust with non-technical users.
"""

from typing import List, Dict


# -----------------------------
# Core assumptions
# -----------------------------

def get_core_assumptions(product_type: str) -> List[str]:
    """
    High-level assumptions used across the analysis.
    Written in plain English for non-technical users.
    """
    return [
        "Online search and discussion trends reasonably reflect real buying interest.",
        "The product quality assumed is average for the category, not premium or poor.",
        "No sudden regulatory bans or major policy changes are assumed.",
        "Market behavior in the recent past is assumed to continue in the near term.",
        "The analysis focuses on demand-side signals more than supply-side internals."
    ]


# -----------------------------
# Limitations
# -----------------------------

def get_limitations(product_type: str) -> List[str]:
    """
    Clear limitations of what the system cannot guarantee or predict.
    """
    return [
        "This analysis does not guarantee business success or profitability.",
        "Execution quality, branding, distribution, and operations strongly affect outcomes.",
        "Exact market size figures may differ from paid industry reports.",
        "Short-term viral spikes or sudden news events may temporarily distort trends.",
        "Offline-only demand and untracked regional markets may be underrepresented."
    ]


# -----------------------------
# Ethical & compliance notes
# -----------------------------

def get_ethics_and_compliance_notes(product_type: str) -> List[str]:
    """
    Notes related to consumer trust, ethics, and regulatory sensitivity.
    """
    sensitive_categories = {"supplements", "health", "medicine", "cosmetics", "baby products"}

    notes = [
        "Product claims should be truthful and supported by evidence.",
        "Misleading marketing can harm consumer trust and brand longevity."
    ]

    if product_type.lower() in sensitive_categories:
        notes.append(
            "This product category may be subject to additional regulatory scrutiny and compliance requirements."
        )

    return notes

"""
Timeline simulator for non-technical users.

Generates a simple, realistic "If you start today..." timeline
that explains what typically happens in phases, without jargon.
"""

from typing import List
from schemas.dashboard_schema import Timeline, TimelinePhase


def generate_timeline(
    competition_intensity: str,
    execution_difficulty: str,
) -> Timeline:
    """
    Builds a phase-wise timeline based on market difficulty.
    """

    phases: List[TimelinePhase] = []

    # Phase 1: Setup
    phases.append(
        TimelinePhase(
            phase="Setup",
            duration="0–2 months",
            expectations=(
                "Product sourcing or formulation, branding basics, "
                "supplier onboarding, compliance checks, and initial market research."
            ),
        )
    )

    # Phase 2: Traction
    if competition_intensity == "High":
        traction_expectation = (
            "Slow initial traction. Marketing costs may be high. "
            "Expect experimentation with pricing, ads, and positioning."
        )
    else:
        traction_expectation = (
            "Early customer traction possible. Feedback-driven improvements "
            "can help refine positioning quickly."
        )

    phases.append(
        TimelinePhase(
            phase="Traction",
            duration="3–6 months",
            expectations=traction_expectation,
        )
    )

    # Phase 3: Scale / Make-or-break
    if execution_difficulty == "Hard":
        scale_expectation = (
            "This is a make-or-break phase. Brands that fail to differentiate "
            "or control costs often drop out. Strong branding and repeat customers matter."
        )
    else:
        scale_expectation = (
            "If early traction is positive, scaling distribution and marketing "
            "can lead to stable growth."
        )

    phases.append(
        TimelinePhase(
            phase="Scale / Decide",
            duration="6–12 months",
            expectations=scale_expectation,
        )
    )

    return Timeline(phases=phases)

"""Session profiles for the tiered model runner.

Maps trigger types to session profiles that control which models run.
"""

from __future__ import annotations

from enum import Enum


class SessionProfile(str, Enum):
    """Controls which models are invoked for a session."""

    FULL = "full"  # Haiku scan → Sonnet investigate → Opus validate
    LIGHT = "light"  # Haiku scan → mini investigation (3 turns) if ROUTINE, else full
    CRISIS = "crisis"  # Skip scan, go straight to Sonnet
    REVIEW = "review"  # Skip scan and validation, Sonnet only
    BUDGET_SAVING = "budget_saving"  # Haiku scan only, no trades


# Default mapping from trigger type to session profile.
# Budget note: 3 FULL sessions ≈ $0.15-0.30/day (well within $3 daily cap).
# Close stays LIGHT — ROUTINE gets a 3-turn mini investigation, not full 8 turns.
SESSION_PROFILE_MAP: dict[str, SessionProfile] = {
    "morning": SessionProfile.FULL,
    "midday": SessionProfile.FULL,
    "close": SessionProfile.LIGHT,
    "event": SessionProfile.CRISIS,
    "weekly_review": SessionProfile.REVIEW,
    "manual": SessionProfile.FULL,
}


def get_session_profile(
    trigger_type: str,
    budget_exhausted: bool = False,
) -> SessionProfile:
    """Resolve session profile from trigger type and budget status.

    Args:
        trigger_type: morning/midday/close/event/weekly_review.
        budget_exhausted: If True, force BUDGET_SAVING regardless of trigger.

    Returns:
        SessionProfile for this session.
    """
    if budget_exhausted:
        return SessionProfile.BUDGET_SAVING
    return SESSION_PROFILE_MAP.get(trigger_type, SessionProfile.LIGHT)

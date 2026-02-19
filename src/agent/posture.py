"""Posture system for Portfolio B risk posture declarations.

The agent declares a posture (balanced, defensive, crisis, aggressive) each session.
PostureManager resolves effective limits and gates aggressive mode behind track record.

Dependency-free: no imports from storage/analysis to avoid circular deps.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class PostureLevel(str, Enum):
    BALANCED = "balanced"
    DEFENSIVE = "defensive"
    CRISIS = "crisis"
    AGGRESSIVE = "aggressive"


@dataclass(frozen=True)
class PostureLimits:
    """Risk limits for a given posture level."""

    max_single_position_pct: float
    max_sector_concentration: float
    max_equity_pct: float  # max % of portfolio in equities (rest = cash)


POSTURE_CONFIGS: dict[PostureLevel, PostureLimits] = {
    PostureLevel.BALANCED: PostureLimits(
        max_single_position_pct=0.35,
        max_sector_concentration=0.50,
        max_equity_pct=0.80,
    ),
    PostureLevel.DEFENSIVE: PostureLimits(
        max_single_position_pct=0.25,
        max_sector_concentration=0.35,
        max_equity_pct=0.50,
    ),
    PostureLevel.CRISIS: PostureLimits(
        max_single_position_pct=0.15,
        max_sector_concentration=0.25,
        max_equity_pct=0.30,
    ),
    PostureLevel.AGGRESSIVE: PostureLimits(
        max_single_position_pct=0.35,
        max_sector_concentration=0.50,
        max_equity_pct=0.95,
    ),
}

class PostureManager:
    """Manages posture declarations and resolves effective risk limits."""

    def get_limits(self, posture: PostureLevel) -> PostureLimits:
        """Get the raw limits for a posture level."""
        return POSTURE_CONFIGS[posture]

    def resolve_effective_limits(
        self,
        posture: PostureLevel,
        total_trades: int = 0,
        win_rate_pct: float = 0.0,
        avg_alpha_vs_spy: float | None = None,
    ) -> tuple[PostureLimits, PostureLevel]:
        """Resolve effective limits for the declared posture.

        In paper trading mode, all postures are available immediately —
        no gate on aggressive. The agent learns by experimenting.

        Args:
            posture: Declared posture level.
            total_trades: Total trades (kept for interface compatibility).
            win_rate_pct: Win rate (kept for interface compatibility).
            avg_alpha_vs_spy: Alpha vs SPY (kept for interface compatibility).

        Returns:
            (limits, effective_posture) — always matches declared posture.
        """
        return POSTURE_CONFIGS[posture], posture

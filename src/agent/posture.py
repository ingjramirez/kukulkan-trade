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

# Gate thresholds for aggressive posture
AGGRESSIVE_MIN_TRADES = 50
AGGRESSIVE_MIN_WIN_RATE = 55.0
AGGRESSIVE_MIN_ALPHA = 0.0  # must be positive


class PostureManager:
    """Manages posture declarations and resolves effective risk limits."""

    def get_limits(self, posture: PostureLevel) -> PostureLimits:
        """Get the raw limits for a posture level."""
        return POSTURE_CONFIGS[posture]

    def validate_aggressive(
        self,
        total_trades: int,
        win_rate_pct: float,
        avg_alpha_vs_spy: float | None,
    ) -> tuple[bool, str]:
        """Check if aggressive posture is allowed based on track record.

        Args:
            total_trades: Total completed trades.
            win_rate_pct: Win rate percentage.
            avg_alpha_vs_spy: Average alpha vs SPY (or None).

        Returns:
            (allowed, reason) — reason explains why it was blocked.
        """
        alpha = avg_alpha_vs_spy if avg_alpha_vs_spy is not None else -1.0

        if total_trades < AGGRESSIVE_MIN_TRADES:
            return False, f"Need {AGGRESSIVE_MIN_TRADES}+ trades (have {total_trades})"
        if win_rate_pct < AGGRESSIVE_MIN_WIN_RATE:
            return False, f"Need {AGGRESSIVE_MIN_WIN_RATE}%+ win rate (have {win_rate_pct:.1f}%)"
        if alpha < AGGRESSIVE_MIN_ALPHA:
            return False, f"Need positive alpha vs SPY (have {alpha:+.2f}%)"
        return True, "Aggressive mode approved"

    def resolve_effective_limits(
        self,
        posture: PostureLevel,
        total_trades: int = 0,
        win_rate_pct: float = 0.0,
        avg_alpha_vs_spy: float | None = None,
    ) -> tuple[PostureLimits, PostureLevel]:
        """Resolve effective limits, falling back to balanced if aggressive gate fails.

        Args:
            posture: Declared posture level.
            total_trades: Total trades (for aggressive gate).
            win_rate_pct: Win rate (for aggressive gate).
            avg_alpha_vs_spy: Alpha vs SPY (for aggressive gate).

        Returns:
            (limits, effective_posture) — effective may differ from declared if gated.
        """
        if posture == PostureLevel.AGGRESSIVE:
            allowed, _ = self.validate_aggressive(total_trades, win_rate_pct, avg_alpha_vs_spy)
            if not allowed:
                return POSTURE_CONFIGS[PostureLevel.BALANCED], PostureLevel.BALANCED
        return POSTURE_CONFIGS[posture], posture

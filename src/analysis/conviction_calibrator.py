"""Conviction calibration — analyze accuracy per conviction level.

Groups trade outcomes by conviction (high/medium/low) to identify
overconfidence, underconfidence, and validated conviction patterns.
Multipliers are informational only — shown to agent, not auto-applied.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.analysis.outcome_tracker import TradeOutcome
from src.analysis.track_record import LOSS_THRESHOLD, WIN_THRESHOLD

MIN_TRADES_PER_BUCKET = 15


@dataclass(frozen=True)
class ConvictionBucket:
    """Stats for a single conviction level."""

    conviction: str  # high, medium, low
    total: int
    wins: int
    losses: int
    win_rate_pct: float
    avg_pnl_pct: float
    assessment: str  # validated, overconfident, underconfident, insufficient, neutral
    suggested_multiplier: float  # informational only


class ConvictionCalibrator:
    """Analyzes trade accuracy per conviction level."""

    def calibrate(self, outcomes: list[TradeOutcome]) -> list[ConvictionBucket]:
        """Group outcomes by conviction and assess calibration.

        Args:
            outcomes: Trade outcomes (from OutcomeTracker).

        Returns:
            List of ConvictionBucket, one per conviction level.
        """
        if not outcomes:
            return []

        groups: dict[str, list[TradeOutcome]] = {}
        for o in outcomes:
            groups.setdefault(o.conviction, []).append(o)

        buckets: list[ConvictionBucket] = []
        for conviction in ("high", "medium", "low"):
            group = groups.get(conviction, [])
            if not group:
                continue
            total = len(group)
            wins = len([o for o in group if o.pnl_pct > WIN_THRESHOLD])
            losses = len([o for o in group if o.pnl_pct < LOSS_THRESHOLD])
            avg_pnl = sum(o.pnl_pct for o in group) / total
            decisive = wins + losses
            win_rate = (wins / decisive * 100) if decisive > 0 else 0.0

            assessment, multiplier = self._assess(conviction, total, win_rate, avg_pnl)
            buckets.append(
                ConvictionBucket(
                    conviction=conviction,
                    total=total,
                    wins=wins,
                    losses=losses,
                    win_rate_pct=round(win_rate, 1),
                    avg_pnl_pct=round(avg_pnl, 2),
                    assessment=assessment,
                    suggested_multiplier=multiplier,
                )
            )
        return buckets

    def format_for_prompt(self, buckets: list[ConvictionBucket]) -> str:
        """Format calibration buckets as concise text for pinned context.

        Args:
            buckets: List of ConvictionBucket from calibrate().

        Returns:
            Formatted text string for prompt injection.
        """
        if not buckets:
            return ""

        lines = ["## Conviction Calibration"]
        for b in buckets:
            icon = {"validated": "+", "overconfident": "!", "underconfident": "^"}.get(b.assessment, " ")
            lines.append(
                f"  {icon} {b.conviction}: {b.win_rate_pct:.0f}%W, "
                f"{b.avg_pnl_pct:+.1f}% avg ({b.total} trades) — {b.assessment}"
            )
            if b.assessment in ("overconfident", "underconfident"):
                lines.append(f"    Suggested sizing: {b.suggested_multiplier:.1f}x")
        return "\n".join(lines)

    @staticmethod
    def _assess(
        conviction: str,
        total: int,
        win_rate: float,
        avg_pnl: float,
    ) -> tuple[str, float]:
        """Assess calibration for a conviction level.

        Returns:
            (assessment, suggested_multiplier)
        """
        if total < MIN_TRADES_PER_BUCKET:
            return "insufficient", 1.0

        if conviction == "high":
            if win_rate > 60.0 and avg_pnl > 1.0:
                return "validated", 1.2
            if win_rate < 50.0:
                return "overconfident", 0.7
            return "neutral", 1.0

        if conviction == "low":
            if win_rate > 60.0 and avg_pnl > 1.0:
                return "underconfident", 1.3
            return "neutral", 1.0

        # medium
        if win_rate > 60.0 and avg_pnl > 1.0:
            return "validated", 1.1
        if win_rate < 45.0:
            return "overconfident", 0.8
        return "neutral", 1.0

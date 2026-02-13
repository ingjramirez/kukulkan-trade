"""Compute win/loss statistics from trade outcomes.

Groups outcomes by sector, regime, session, and conviction to identify
what trading patterns work well and which don't.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.analysis.outcome_tracker import TradeOutcome

# Thresholds: >0.5% = win, <-0.5% = loss, in between = scratch
WIN_THRESHOLD = 0.5
LOSS_THRESHOLD = -0.5


@dataclass(frozen=True)
class CategoryWinRate:
    """Win rate stats for a category slice."""

    category: str
    value: str
    total: int
    wins: int
    losses: int
    win_rate_pct: float
    avg_pnl_pct: float
    avg_alpha_vs_spy: float | None


@dataclass(frozen=True)
class TrackRecordStats:
    """Aggregated track record across dimensions."""

    total_trades: int
    wins: int
    losses: int
    scratches: int
    win_rate_pct: float
    avg_pnl_pct: float
    avg_alpha_vs_spy: float | None
    by_sector: list[CategoryWinRate] = field(default_factory=list)
    by_conviction: list[CategoryWinRate] = field(default_factory=list)
    best_sector: str | None = None
    worst_sector: str | None = None


class TrackRecord:
    """Computes win/loss statistics from trade outcomes."""

    def compute(self, outcomes: list[TradeOutcome]) -> TrackRecordStats:
        """Group outcomes by dimension and compute win rates.

        Args:
            outcomes: List of TradeOutcome from OutcomeTracker.

        Returns:
            TrackRecordStats with overall and per-dimension stats.
        """
        if not outcomes:
            return TrackRecordStats(
                total_trades=0,
                wins=0,
                losses=0,
                scratches=0,
                win_rate_pct=0.0,
                avg_pnl_pct=0.0,
                avg_alpha_vs_spy=None,
            )

        wins = [o for o in outcomes if o.pnl_pct > WIN_THRESHOLD]
        losses = [o for o in outcomes if o.pnl_pct < LOSS_THRESHOLD]
        scratches = [o for o in outcomes if LOSS_THRESHOLD <= o.pnl_pct <= WIN_THRESHOLD]

        total_non_scratch = len(wins) + len(losses)
        win_rate = (len(wins) / total_non_scratch * 100) if total_non_scratch > 0 else 0.0
        avg_pnl = sum(o.pnl_pct for o in outcomes) / len(outcomes)

        alphas = [o.alpha_vs_spy for o in outcomes if o.alpha_vs_spy is not None]
        avg_alpha = sum(alphas) / len(alphas) if alphas else None

        # By sector
        by_sector = self._group_by(outcomes, "sector")
        # By conviction
        by_conviction = self._group_by(outcomes, "conviction")

        # Best/worst sector
        best_sector = None
        worst_sector = None
        if by_sector:
            sorted_sectors = sorted(by_sector, key=lambda c: c.avg_pnl_pct, reverse=True)
            best_sector = sorted_sectors[0].value
            worst_sector = sorted_sectors[-1].value

        return TrackRecordStats(
            total_trades=len(outcomes),
            wins=len(wins),
            losses=len(losses),
            scratches=len(scratches),
            win_rate_pct=round(win_rate, 1),
            avg_pnl_pct=round(avg_pnl, 2),
            avg_alpha_vs_spy=round(avg_alpha, 2) if avg_alpha is not None else None,
            by_sector=by_sector,
            by_conviction=by_conviction,
            best_sector=best_sector,
            worst_sector=worst_sector,
        )

    @staticmethod
    def format_for_prompt(stats: TrackRecordStats) -> str:
        """Format track record stats for the system prompt (~200 tokens).

        Args:
            stats: Computed TrackRecordStats.

        Returns:
            Formatted text for prompt injection.
        """
        if stats.total_trades == 0:
            return "No completed trades to analyze yet."

        lines = [
            f"Win rate: {stats.win_rate_pct:.0f}% ({stats.wins}W/{stats.losses}L/{stats.scratches}S "
            f"from {stats.total_trades} trades)",
            f"Avg P&L: {stats.avg_pnl_pct:+.2f}%",
        ]

        if stats.avg_alpha_vs_spy is not None:
            lines.append(f"Avg alpha vs SPY: {stats.avg_alpha_vs_spy:+.2f}%")

        if stats.best_sector and stats.worst_sector and stats.best_sector != stats.worst_sector:
            lines.append(f"Best sector: {stats.best_sector} | Worst: {stats.worst_sector}")

        # Conviction breakdown
        if stats.by_conviction:
            conv_parts = []
            for c in sorted(stats.by_conviction, key=lambda x: x.value):
                conv_parts.append(f"{c.value}: {c.win_rate_pct:.0f}%W ({c.avg_pnl_pct:+.1f}%)")
            lines.append(f"By conviction: {', '.join(conv_parts)}")

        return "\n".join(lines)

    @staticmethod
    def _group_by(
        outcomes: list[TradeOutcome],
        attr: str,
    ) -> list[CategoryWinRate]:
        """Group outcomes by an attribute and compute stats per group."""
        groups: dict[str, list[TradeOutcome]] = {}
        for o in outcomes:
            key = getattr(o, attr, "Unknown")
            groups.setdefault(key, []).append(o)

        result: list[CategoryWinRate] = []
        for value, group in sorted(groups.items()):
            wins = [o for o in group if o.pnl_pct > WIN_THRESHOLD]
            losses = [o for o in group if o.pnl_pct < LOSS_THRESHOLD]
            total_decisive = len(wins) + len(losses)
            win_rate = (len(wins) / total_decisive * 100) if total_decisive > 0 else 0.0
            avg_pnl = sum(o.pnl_pct for o in group) / len(group)
            alphas = [o.alpha_vs_spy for o in group if o.alpha_vs_spy is not None]
            avg_alpha = sum(alphas) / len(alphas) if alphas else None

            result.append(
                CategoryWinRate(
                    category=attr,
                    value=value,
                    total=len(group),
                    wins=len(wins),
                    losses=len(losses),
                    win_rate_pct=round(win_rate, 1),
                    avg_pnl_pct=round(avg_pnl, 2),
                    avg_alpha_vs_spy=round(avg_alpha, 2) if avg_alpha is not None else None,
                )
            )
        return result

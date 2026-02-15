"""Empirical playbook generator — regime×sector win rate matrix.

Groups trade outcomes by regime and sector to produce actionable
recommendations (sweet_spot, solid, avoid, neutral, insufficient_data).
"""

from __future__ import annotations

from dataclasses import dataclass

from src.analysis.outcome_tracker import TradeOutcome
from src.analysis.track_record import LOSS_THRESHOLD, WIN_THRESHOLD

MIN_TRADES_PER_CELL = 10


@dataclass(frozen=True)
class PlaybookCell:
    """Single cell in the regime×sector playbook matrix."""

    regime: str
    sector: str
    total: int
    wins: int
    losses: int
    win_rate_pct: float
    avg_pnl_pct: float
    recommendation: str  # sweet_spot, solid, avoid, neutral, insufficient_data


class PlaybookGenerator:
    """Generates empirical playbooks from trade outcomes."""

    def generate(self, outcomes: list[TradeOutcome]) -> list[PlaybookCell]:
        """Group outcomes by regime×sector and classify each cell.

        Args:
            outcomes: Trade outcomes (from OutcomeTracker).

        Returns:
            List of PlaybookCell, one per regime×sector combination.
        """
        if not outcomes:
            return []

        # Group by (regime, sector)
        groups: dict[tuple[str, str], list[TradeOutcome]] = {}
        for o in outcomes:
            regime = o.regime_at_entry or "Unknown"
            groups.setdefault((regime, o.sector), []).append(o)

        cells: list[PlaybookCell] = []
        for (regime, sector), group in sorted(groups.items()):
            total = len(group)
            wins = len([o for o in group if o.pnl_pct > WIN_THRESHOLD])
            losses = len([o for o in group if o.pnl_pct < LOSS_THRESHOLD])
            avg_pnl = sum(o.pnl_pct for o in group) / total
            decisive = wins + losses
            win_rate = (wins / decisive * 100) if decisive > 0 else 0.0

            recommendation = self._classify(total, win_rate, avg_pnl)
            cells.append(
                PlaybookCell(
                    regime=regime,
                    sector=sector,
                    total=total,
                    wins=wins,
                    losses=losses,
                    win_rate_pct=round(win_rate, 1),
                    avg_pnl_pct=round(avg_pnl, 2),
                    recommendation=recommendation,
                )
            )
        return cells

    def format_for_prompt(self, cells: list[PlaybookCell]) -> str:
        """Format playbook cells as concise text for the agent's pinned context.

        Only shows actionable cells (sweet_spot, solid, avoid).

        Args:
            cells: List of PlaybookCell from generate().

        Returns:
            Formatted text string for prompt injection.
        """
        if not cells:
            return ""

        actionable = [c for c in cells if c.recommendation in ("sweet_spot", "solid", "avoid")]
        if not actionable:
            return ""

        lines = ["## Empirical Playbook"]
        for c in actionable:
            icon = {"sweet_spot": "+", "solid": "+", "avoid": "-"}.get(c.recommendation, " ")
            lines.append(
                f"  {icon} {c.regime}/{c.sector}: {c.win_rate_pct:.0f}%W, "
                f"{c.avg_pnl_pct:+.1f}% avg ({c.total} trades) — {c.recommendation}"
            )
        return "\n".join(lines)

    @staticmethod
    def _classify(total: int, win_rate: float, avg_pnl: float) -> str:
        """Classify a cell based on stats."""
        if total < MIN_TRADES_PER_CELL:
            return "insufficient_data"
        if win_rate > 65.0:
            return "sweet_spot"
        if win_rate > 55.0:
            return "solid"
        if win_rate < 45.0:
            return "avoid"
        return "neutral"

"""Trend analyzer for weekly improvement snapshots.

Computes linear regression slope on win_rate and P&L series
to classify performance as improving/stable/declining.
"""

from __future__ import annotations

from dataclasses import dataclass

import structlog

from src.storage.database import Database

log = structlog.get_logger()

# Minimum snapshots needed for trend analysis
MIN_SNAPSHOTS = 3

# Relative slope threshold: >5% per week = improving/declining
SLOPE_THRESHOLD = 5.0


@dataclass(frozen=True)
class TrendDataPoint:
    """A single data point in the trend series."""

    week_label: str  # e.g. "2026-02-10"
    win_rate_pct: float | None
    avg_pnl_pct: float | None
    total_trades: int


@dataclass(frozen=True)
class TrendResult:
    """Result of trend analysis across multiple weeks."""

    classification: str  # "improving", "stable", "declining", "insufficient_data"
    win_rate_slope: float  # percentage points per week
    pnl_slope: float  # percentage points per week
    data_points: list[TrendDataPoint]
    weeks_analyzed: int


class TrendAnalyzer:
    """Analyzes performance trends from improvement snapshots."""

    async def compute_trend(
        self,
        db: Database,
        tenant_id: str = "default",
        weeks: int = 8,
    ) -> TrendResult:
        """Load snapshots and compute performance trend.

        Args:
            db: Database instance.
            tenant_id: Tenant to analyze.
            weeks: Number of recent weeks to include.

        Returns:
            TrendResult with classification and slope data.
        """
        snapshots = await db.get_improvement_snapshots(tenant_id=tenant_id, limit=weeks)

        if len(snapshots) < MIN_SNAPSHOTS:
            return TrendResult(
                classification="insufficient_data",
                win_rate_slope=0.0,
                pnl_slope=0.0,
                data_points=[],
                weeks_analyzed=len(snapshots),
            )

        # Reverse to chronological order (oldest first)
        snapshots = list(reversed(snapshots))

        data_points = [
            TrendDataPoint(
                week_label=s.week_start.isoformat() if s.week_start else "",
                win_rate_pct=s.win_rate_pct,
                avg_pnl_pct=s.avg_pnl_pct,
                total_trades=s.total_trades,
            )
            for s in snapshots
        ]

        # Compute slopes using simple linear regression
        win_rates = [s.win_rate_pct for s in snapshots if s.win_rate_pct is not None]
        pnl_rates = [s.avg_pnl_pct for s in snapshots if s.avg_pnl_pct is not None]

        win_rate_slope = _linear_slope(win_rates) if len(win_rates) >= MIN_SNAPSHOTS else 0.0
        pnl_slope = _linear_slope(pnl_rates) if len(pnl_rates) >= MIN_SNAPSHOTS else 0.0

        # Classify based on win rate slope (primary) and P&L slope (secondary)
        classification = _classify(win_rate_slope, pnl_slope)

        return TrendResult(
            classification=classification,
            win_rate_slope=round(win_rate_slope, 2),
            pnl_slope=round(pnl_slope, 2),
            data_points=data_points,
            weeks_analyzed=len(snapshots),
        )


def _linear_slope(values: list[float]) -> float:
    """Compute linear regression slope (y units per x unit).

    Uses least-squares: slope = Σ((x-x̄)(y-ȳ)) / Σ((x-x̄)²)
    where x is the week index (0, 1, 2, ...).
    """
    n = len(values)
    if n < 2:
        return 0.0

    x_mean = (n - 1) / 2.0
    y_mean = sum(values) / n

    numerator = sum((i - x_mean) * (y - y_mean) for i, y in enumerate(values))
    denominator = sum((i - x_mean) ** 2 for i in range(n))

    if denominator == 0:
        return 0.0

    return numerator / denominator


def _classify(win_rate_slope: float, pnl_slope: float) -> str:
    """Classify trend as improving/stable/declining.

    Primary signal: win rate slope > ±5 pp/week.
    Secondary confirmation: P&L slope direction.
    """
    if win_rate_slope > SLOPE_THRESHOLD:
        return "improving"
    elif win_rate_slope < -SLOPE_THRESHOLD:
        return "declining"
    elif pnl_slope > SLOPE_THRESHOLD:
        return "improving"
    elif pnl_slope < -SLOPE_THRESHOLD:
        return "declining"
    else:
        return "stable"

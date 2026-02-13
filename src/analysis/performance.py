"""Performance tracking and statistics for portfolio analysis.

Computes inception return, peak value, drawdown, win rate,
and other metrics from daily snapshots and trades.
"""

from dataclasses import dataclass

import pandas as pd
import structlog

from src.storage.database import Database

log = structlog.get_logger()


@dataclass(frozen=True)
class PerformanceStats:
    """Portfolio performance statistics."""

    portfolio: str
    initial_value: float
    current_value: float
    inception_return_pct: float
    peak_value: float
    drawdown_pct: float
    win_rate_pct: float | None
    total_trades: int
    winning_trades: int
    losing_trades: int
    best_day_pct: float | None
    worst_day_pct: float | None
    days_tracked: int
    spy_return_pct: float | None = None
    alpha_pct: float | None = None


class PerformanceTracker:
    """Computes portfolio performance metrics from DB data."""

    async def get_portfolio_stats(
        self,
        db: Database,
        portfolio_name: str,
        initial_value: float,
        spy_closes: pd.Series | None = None,
    ) -> PerformanceStats:
        """Compute performance statistics for a portfolio.

        Args:
            db: Database instance.
            portfolio_name: A or B.
            initial_value: Starting allocation.
            spy_closes: SPY close prices Series (index=dates). Optional.

        Returns:
            PerformanceStats with computed metrics.
        """
        snapshots = await db.get_snapshots(portfolio_name)
        trades = await db.get_trades(portfolio_name)

        # Current value
        current_value = snapshots[-1].total_value if snapshots else initial_value
        inception_return = ((current_value - initial_value) / initial_value) * 100

        # Peak and drawdown
        peak = initial_value
        max_dd = 0.0
        for s in snapshots:
            if s.total_value > peak:
                peak = s.total_value
            dd = ((peak - s.total_value) / peak) * 100
            if dd > max_dd:
                max_dd = dd

        # Best/worst day
        daily_returns = [s.daily_return_pct for s in snapshots if s.daily_return_pct is not None]
        best_day = max(daily_returns) if daily_returns else None
        worst_day = min(daily_returns) if daily_returns else None

        # Win rate from trades
        winning = 0
        losing = 0
        for t in trades:
            if t.side == "SELL":
                # Simple heuristic: compare sell price to buy avg
                positions = await db.get_positions(portfolio_name)
                pos = next((p for p in positions if p.ticker == t.ticker), None)
                # If position still exists, compare against avg_price
                # If not, we sold all — treat as profitable if sell price > 0
                if pos and t.price > pos.avg_price:
                    winning += 1
                elif pos and t.price <= pos.avg_price:
                    losing += 1

        # Fallback: count using snapshot daily returns
        if not trades:
            winning = sum(1 for r in daily_returns if r is not None and r > 0)
            losing = sum(1 for r in daily_returns if r is not None and r < 0)

        total_trades = len(trades)
        win_rate = None
        if winning + losing > 0:
            win_rate = (winning / (winning + losing)) * 100

        # SPY benchmark
        spy_return_pct: float | None = None
        alpha_pct: float | None = None
        if spy_closes is not None and len(snapshots) >= 2:
            first_date = snapshots[0].date
            last_date = snapshots[-1].date
            try:
                spy_idx = spy_closes.index
                # Find nearest dates in SPY data
                spy_at_start = spy_closes.loc[spy_idx >= pd.Timestamp(first_date)]
                spy_at_end = spy_closes.loc[spy_idx <= pd.Timestamp(last_date)]
                if len(spy_at_start) > 0 and len(spy_at_end) > 0:
                    start_price = float(spy_at_start.iloc[0])
                    end_price = float(spy_at_end.iloc[-1])
                    if start_price > 0:
                        spy_return_pct = round(
                            ((end_price - start_price) / start_price) * 100,
                            2,
                        )
                        alpha_pct = round(inception_return - spy_return_pct, 2)
            except Exception:
                pass  # SPY data issue — leave as None

        return PerformanceStats(
            portfolio=portfolio_name,
            initial_value=initial_value,
            current_value=round(current_value, 2),
            inception_return_pct=round(inception_return, 2),
            peak_value=round(peak, 2),
            drawdown_pct=round(max_dd, 2),
            win_rate_pct=round(win_rate, 2) if win_rate is not None else None,
            total_trades=total_trades,
            winning_trades=winning,
            losing_trades=losing,
            best_day_pct=round(best_day, 2) if best_day is not None else None,
            worst_day_pct=round(worst_day, 2) if worst_day is not None else None,
            days_tracked=len(snapshots),
            spy_return_pct=spy_return_pct,
            alpha_pct=alpha_pct,
        )

    def format_for_prompt(self, stats: PerformanceStats) -> str:
        """Format performance stats for the agent system prompt.

        Args:
            stats: Computed performance statistics.

        Returns:
            Multi-line text block for prompt injection.
        """
        lines = [
            f"Portfolio {stats.portfolio} Performance:",
            f"  Value: ${stats.current_value:,.2f} (inception: {stats.inception_return_pct:+.2f}%)",
            f"  Peak: ${stats.peak_value:,.2f} | Drawdown: {stats.drawdown_pct:.2f}%",
            f"  Days tracked: {stats.days_tracked}",
        ]
        if stats.win_rate_pct is not None:
            wins, losses = stats.winning_trades, stats.losing_trades
            lines.append(f"  Win rate: {stats.win_rate_pct:.0f}% ({wins}W / {losses}L)")
        if stats.best_day_pct is not None:
            best = stats.best_day_pct
            worst = stats.worst_day_pct
            lines.append(f"  Best day: {best:+.2f}% | Worst day: {worst:+.2f}%")
        if stats.spy_return_pct is not None:
            lines.append(f"  vs SPY: {stats.spy_return_pct:+.2f}% | Alpha: {stats.alpha_pct:+.2f}%")
        return "\n".join(lines)

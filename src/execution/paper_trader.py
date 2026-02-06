"""Paper trading executor.

Simulates trade execution locally. Updates positions, cash, and logs trades.
No IBKR dependency — pure in-memory simulation backed by SQLite.
"""

from datetime import date

import structlog

from src.storage.database import Database
from src.storage.models import TradeSchema

log = structlog.get_logger()


class PaperTrader:
    """Simulates trade execution with portfolio state in SQLite."""

    def __init__(self, db: Database) -> None:
        self._db = db

    async def initialize_portfolios(self) -> None:
        """Create A, B, C portfolios with starting cash if they don't exist."""
        for name in ("A", "B", "C"):
            existing = await self._db.get_portfolio(name)
            if existing is None:
                await self._db.upsert_portfolio(name, cash=33_333.0, total_value=33_333.0)
                log.info("portfolio_initialized", portfolio=name, cash=33_333.0)

    async def execute_trades(self, trades: list[TradeSchema]) -> list[TradeSchema]:
        """Execute a batch of trade signals.

        Processes sells before buys to free up cash.

        Args:
            trades: List of validated TradeSchema objects.

        Returns:
            List of successfully executed trades.
        """
        # Sort: sells first, then buys
        sells = [t for t in trades if t.side.value == "SELL"]
        buys = [t for t in trades if t.side.value == "BUY"]
        executed: list[TradeSchema] = []

        for trade in sells + buys:
            success = await self._execute_single(trade)
            if success:
                executed.append(trade)

        log.info(
            "trades_executed",
            portfolio=trades[0].portfolio.value if trades else "?",
            attempted=len(trades),
            executed=len(executed),
        )
        return executed

    async def _execute_single(self, trade: TradeSchema) -> bool:
        """Execute a single trade, updating positions and cash.

        Args:
            trade: Validated trade signal.

        Returns:
            True if executed successfully, False if rejected.
        """
        portfolio_name = trade.portfolio.value
        portfolio = await self._db.get_portfolio(portfolio_name)
        if portfolio is None:
            log.error("portfolio_not_found", portfolio=portfolio_name)
            return False

        positions = await self._db.get_positions(portfolio_name)
        position_map = {p.ticker: p for p in positions}

        if trade.side.value == "BUY":
            cost = trade.total
            if cost > portfolio.cash:
                log.warning(
                    "insufficient_cash",
                    portfolio=portfolio_name,
                    needed=cost,
                    available=portfolio.cash,
                )
                return False

            # Update position
            existing = position_map.get(trade.ticker)
            if existing:
                total_shares = existing.shares + trade.shares
                total_cost = (existing.shares * existing.avg_price) + cost
                new_avg = total_cost / total_shares
                await self._db.upsert_position(
                    portfolio_name, trade.ticker, total_shares, new_avg
                )
            else:
                await self._db.upsert_position(
                    portfolio_name, trade.ticker, trade.shares, trade.price
                )

            # Deduct cash
            await self._db.upsert_portfolio(
                portfolio_name,
                cash=portfolio.cash - cost,
                total_value=portfolio.total_value,  # will be recalculated
            )

        elif trade.side.value == "SELL":
            existing = position_map.get(trade.ticker)
            if existing is None or existing.shares < trade.shares:
                log.warning(
                    "insufficient_shares",
                    portfolio=portfolio_name,
                    ticker=trade.ticker,
                    needed=trade.shares,
                    available=existing.shares if existing else 0,
                )
                return False

            remaining = existing.shares - trade.shares
            await self._db.upsert_position(
                portfolio_name, trade.ticker, remaining, existing.avg_price
            )

            # Add proceeds to cash
            proceeds = trade.total
            await self._db.upsert_portfolio(
                portfolio_name,
                cash=portfolio.cash + proceeds,
                total_value=portfolio.total_value,
            )

        # Log the trade
        await self._db.log_trade(
            portfolio=portfolio_name,
            ticker=trade.ticker,
            side=trade.side.value,
            shares=trade.shares,
            price=trade.price,
            reason=trade.reason,
        )

        log.info(
            "trade_executed",
            portfolio=portfolio_name,
            ticker=trade.ticker,
            side=trade.side.value,
            shares=trade.shares,
            price=trade.price,
        )
        return True

    async def take_snapshot(
        self,
        portfolio_name: str,
        snapshot_date: date,
        prices: dict[str, float],
    ) -> None:
        """Record end-of-day portfolio snapshot.

        Args:
            portfolio_name: A, B, or C.
            snapshot_date: Date of the snapshot.
            prices: Dict of ticker -> current price.
        """
        portfolio = await self._db.get_portfolio(portfolio_name)
        if portfolio is None:
            return

        positions = await self._db.get_positions(portfolio_name)
        positions_value = sum(
            p.shares * prices.get(p.ticker, p.avg_price) for p in positions
        )
        total_value = portfolio.cash + positions_value

        # Calculate daily return
        snapshots = await self._db.get_snapshots(portfolio_name)
        daily_return_pct = None
        cumulative_return_pct = None
        if snapshots:
            prev = snapshots[-1]
            if prev.total_value > 0:
                daily_return_pct = ((total_value - prev.total_value) / prev.total_value) * 100
            cumulative_return_pct = ((total_value - 33_333.0) / 33_333.0) * 100
        else:
            cumulative_return_pct = ((total_value - 33_333.0) / 33_333.0) * 100

        await self._db.save_snapshot(
            portfolio=portfolio_name,
            snapshot_date=snapshot_date,
            total_value=total_value,
            cash=portfolio.cash,
            positions_value=positions_value,
            daily_return_pct=daily_return_pct,
            cumulative_return_pct=cumulative_return_pct,
        )

        # Update portfolio total value
        await self._db.upsert_portfolio(portfolio_name, portfolio.cash, total_value)

        log.info(
            "snapshot_taken",
            portfolio=portfolio_name,
            total_value=round(total_value, 2),
            daily_return=round(daily_return_pct, 2) if daily_return_pct else None,
        )

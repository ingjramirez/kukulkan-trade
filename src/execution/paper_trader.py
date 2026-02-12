"""Paper trading executor.

Simulates trade execution locally. Updates positions, cash, and logs trades.
Pure in-memory simulation backed by SQLite.
"""

from datetime import date

import structlog

from src.storage.database import Database
from src.storage.models import TradeSchema
from src.utils.allocations import DEFAULT_ALLOCATIONS, TenantAllocations

log = structlog.get_logger()


class PaperTrader:
    """Simulates trade execution with portfolio state in SQLite."""

    def __init__(self, db: Database) -> None:
        self._db = db

    async def initialize_portfolios(
        self,
        allocations: TenantAllocations | None = None,
        tenant_id: str = "default",
    ) -> None:
        """Create A and B portfolios with starting cash if they don't exist."""
        alloc = allocations or DEFAULT_ALLOCATIONS
        for name, cash in [
            ("A", alloc.portfolio_a_cash),
            ("B", alloc.portfolio_b_cash),
        ]:
            existing = await self._db.get_portfolio(name, tenant_id=tenant_id)
            if existing is None:
                await self._db.upsert_portfolio(
                    name, cash=cash, total_value=cash, tenant_id=tenant_id,
                )
                log.info("portfolio_initialized", portfolio=name, cash=cash)

    async def execute_trades(
        self, trades: list[TradeSchema], tenant_id: str = "default",
    ) -> list[TradeSchema]:
        """Execute a batch of trade signals.

        Processes sells before buys to free up cash.

        Args:
            trades: List of validated TradeSchema objects.
            tenant_id: Tenant UUID for data isolation.

        Returns:
            List of successfully executed trades.
        """
        # Sort: sells first, then buys
        sells = [t for t in trades if t.side.value == "SELL"]
        buys = [t for t in trades if t.side.value == "BUY"]
        executed: list[TradeSchema] = []

        for trade in sells + buys:
            success = await self._execute_single(trade, tenant_id=tenant_id)
            if success:
                executed.append(trade)

        log.info(
            "trades_executed",
            portfolio=trades[0].portfolio.value if trades else "?",
            attempted=len(trades),
            executed=len(executed),
        )
        return executed

    async def _execute_single(
        self, trade: TradeSchema, tenant_id: str = "default",
    ) -> bool:
        """Execute a single trade, updating positions and cash.

        Args:
            trade: Validated trade signal.
            tenant_id: Tenant UUID for data isolation.

        Returns:
            True if executed successfully, False if rejected.
        """
        portfolio_name = trade.portfolio.value
        portfolio = await self._db.get_portfolio(portfolio_name, tenant_id=tenant_id)
        if portfolio is None:
            log.error("portfolio_not_found", portfolio=portfolio_name)
            return False

        positions = await self._db.get_positions(portfolio_name, tenant_id=tenant_id)
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
                    portfolio_name, trade.ticker, total_shares, new_avg,
                    tenant_id=tenant_id,
                )
            else:
                await self._db.upsert_position(
                    portfolio_name, trade.ticker, trade.shares, trade.price,
                    tenant_id=tenant_id,
                )

            # Deduct cash
            await self._db.upsert_portfolio(
                portfolio_name,
                cash=portfolio.cash - cost,
                total_value=portfolio.total_value,  # will be recalculated
                tenant_id=tenant_id,
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
                portfolio_name, trade.ticker, remaining, existing.avg_price,
                tenant_id=tenant_id,
            )

            # Add proceeds to cash
            proceeds = trade.total
            await self._db.upsert_portfolio(
                portfolio_name,
                cash=portfolio.cash + proceeds,
                total_value=portfolio.total_value,
                tenant_id=tenant_id,
            )

        # Log the trade
        await self._db.log_trade(
            portfolio=portfolio_name,
            ticker=trade.ticker,
            side=trade.side.value,
            shares=trade.shares,
            price=trade.price,
            reason=trade.reason,
            tenant_id=tenant_id,
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
        allocations: TenantAllocations | None = None,
        tenant_id: str = "default",
    ) -> None:
        """Record end-of-day portfolio snapshot.

        Args:
            portfolio_name: A or B.
            snapshot_date: Date of the snapshot.
            prices: Dict of ticker -> current price.
            allocations: Tenant allocations for initial value reference.
            tenant_id: Tenant UUID for data isolation.
        """
        alloc = allocations or DEFAULT_ALLOCATIONS

        portfolio = await self._db.get_portfolio(
            portfolio_name, tenant_id=tenant_id,
        )
        if portfolio is None:
            return

        positions = await self._db.get_positions(
            portfolio_name, tenant_id=tenant_id,
        )
        positions_value = sum(
            p.shares * prices.get(p.ticker, p.avg_price) for p in positions
        )
        total_value = portfolio.cash + positions_value

        initial_value = alloc.for_portfolio(portfolio_name)

        # Calculate daily return
        snapshots = await self._db.get_snapshots(
            portfolio_name, tenant_id=tenant_id,
        )
        daily_return_pct = None
        cumulative_return_pct = None
        if snapshots:
            prev = snapshots[-1]
            if prev.total_value > 0:
                daily_return_pct = ((total_value - prev.total_value) / prev.total_value) * 100
            cumulative_return_pct = ((total_value - initial_value) / initial_value) * 100
        else:
            cumulative_return_pct = ((total_value - initial_value) / initial_value) * 100

        await self._db.save_snapshot(
            portfolio=portfolio_name,
            snapshot_date=snapshot_date,
            total_value=total_value,
            cash=portfolio.cash,
            positions_value=positions_value,
            daily_return_pct=daily_return_pct,
            cumulative_return_pct=cumulative_return_pct,
            tenant_id=tenant_id,
        )

        # Update position current prices
        await self._db.update_position_prices(
            portfolio_name, prices, tenant_id=tenant_id,
        )

        # Update portfolio total value
        await self._db.upsert_portfolio(
            portfolio_name, portfolio.cash, total_value, tenant_id=tenant_id,
        )

        log.info(
            "snapshot_taken",
            portfolio=portfolio_name,
            total_value=round(total_value, 2),
            daily_return=round(daily_return_pct, 2) if daily_return_pct else None,
        )

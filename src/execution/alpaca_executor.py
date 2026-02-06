"""Alpaca trade executor using alpaca-py.

REST-based executor for Alpaca paper/live trading.
Same 3-method interface as PaperTrader and IBKRExecutor.
"""

from datetime import date

import structlog
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide as AlpacaSide
from alpaca.trading.enums import TimeInForce
from alpaca.trading.requests import MarketOrderRequest

from src.storage.database import Database
from src.storage.models import TradeSchema

log = structlog.get_logger()


class AlpacaExecutor:
    """Executes trades via Alpaca REST API and logs to our database.

    Alpaca sees one account. Our DB tracks the A/B split
    using the portfolio field on each trade and position.
    """

    def __init__(self, db: Database, client: TradingClient) -> None:
        self._db = db
        self._client = client

    async def initialize_portfolios(self) -> None:
        """Create portfolio rows if they don't exist."""
        from config.strategies import PORTFOLIO_A, PORTFOLIO_B

        for name, alloc in [("A", PORTFOLIO_A.allocation_usd), ("B", PORTFOLIO_B.allocation_usd)]:
            existing = await self._db.get_portfolio(name)
            if existing is None:
                await self._db.upsert_portfolio(name, cash=alloc, total_value=alloc)
                log.info("portfolio_initialized_alpaca", portfolio=name, cash=alloc)

    async def execute_trades(self, trades: list[TradeSchema]) -> list[TradeSchema]:
        """Submit market orders to Alpaca and log fills to DB.

        Processes sells before buys to free up cash.

        Args:
            trades: List of validated TradeSchema objects.

        Returns:
            List of successfully executed trades.
        """
        sells = [t for t in trades if t.side.value == "SELL"]
        buys = [t for t in trades if t.side.value == "BUY"]
        executed: list[TradeSchema] = []

        for trade in sells + buys:
            success = await self._execute_single(trade)
            if success:
                executed.append(trade)

        log.info(
            "alpaca_trades_executed",
            attempted=len(trades),
            executed=len(executed),
        )
        return executed

    async def _execute_single(self, trade: TradeSchema) -> bool:
        """Execute a single trade via Alpaca.

        Args:
            trade: Validated trade signal.

        Returns:
            True if executed successfully.
        """
        portfolio_name = trade.portfolio.value
        side = AlpacaSide.BUY if trade.side.value == "BUY" else AlpacaSide.SELL

        try:
            order_request = MarketOrderRequest(
                symbol=trade.ticker,
                qty=int(trade.shares),
                side=side,
                time_in_force=TimeInForce.DAY,
                client_order_id=f"atlas-{portfolio_name}-{trade.ticker}",
            )
            order = self._client.submit_order(order_request)

            # Use filled price if available, fall back to estimated
            fill_price = trade.price
            if order.filled_avg_price is not None:
                fill_price = float(order.filled_avg_price)

            log.info(
                "alpaca_order_submitted",
                ticker=trade.ticker,
                side=trade.side.value,
                shares=int(trade.shares),
                order_id=str(order.id),
                status=str(order.status),
                fill_price=fill_price,
            )

            # Update portfolio state in our DB
            portfolio = await self._db.get_portfolio(portfolio_name)
            if portfolio is None:
                return False

            positions = await self._db.get_positions(portfolio_name)
            position_map = {p.ticker: p for p in positions}

            if trade.side.value == "BUY":
                cost = trade.shares * fill_price
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
                        portfolio_name, trade.ticker, trade.shares, fill_price
                    )
                await self._db.upsert_portfolio(
                    portfolio_name,
                    cash=portfolio.cash - cost,
                    total_value=portfolio.total_value,
                )
            elif trade.side.value == "SELL":
                existing = position_map.get(trade.ticker)
                if existing is None:
                    return False
                remaining = existing.shares - trade.shares
                await self._db.upsert_position(
                    portfolio_name, trade.ticker, remaining, existing.avg_price
                )
                proceeds = trade.shares * fill_price
                await self._db.upsert_portfolio(
                    portfolio_name,
                    cash=portfolio.cash + proceeds,
                    total_value=portfolio.total_value,
                )

            # Log trade
            await self._db.log_trade(
                portfolio=portfolio_name,
                ticker=trade.ticker,
                side=trade.side.value,
                shares=trade.shares,
                price=fill_price,
                reason=trade.reason,
            )

            return True

        except Exception as e:
            log.error(
                "alpaca_order_failed",
                ticker=trade.ticker,
                side=trade.side.value,
                error=str(e),
            )
            return False

    async def take_snapshot(
        self,
        portfolio_name: str,
        snapshot_date: date,
        prices: dict[str, float],
    ) -> None:
        """Record end-of-day portfolio snapshot.

        Args:
            portfolio_name: A or B.
            snapshot_date: Date of the snapshot.
            prices: Dict of ticker -> current price.
        """
        from config.strategies import PORTFOLIO_A, PORTFOLIO_B

        portfolio = await self._db.get_portfolio(portfolio_name)
        if portfolio is None:
            return

        positions = await self._db.get_positions(portfolio_name)
        positions_value = sum(
            p.shares * prices.get(p.ticker, p.avg_price) for p in positions
        )
        total_value = portfolio.cash + positions_value

        initial_value = PORTFOLIO_A.allocation_usd if portfolio_name == "A" else PORTFOLIO_B.allocation_usd

        snapshots = await self._db.get_snapshots(portfolio_name)
        daily_return_pct = None
        cumulative_return_pct = ((total_value - initial_value) / initial_value) * 100
        if snapshots:
            prev = snapshots[-1]
            if prev.total_value > 0:
                daily_return_pct = ((total_value - prev.total_value) / prev.total_value) * 100

        await self._db.save_snapshot(
            portfolio=portfolio_name,
            snapshot_date=snapshot_date,
            total_value=total_value,
            cash=portfolio.cash,
            positions_value=positions_value,
            daily_return_pct=daily_return_pct,
            cumulative_return_pct=cumulative_return_pct,
        )

        await self._db.upsert_portfolio(portfolio_name, portfolio.cash, total_value)

        log.info(
            "alpaca_snapshot_taken",
            portfolio=portfolio_name,
            total_value=round(total_value, 2),
            daily_return=round(daily_return_pct, 2) if daily_return_pct else None,
        )

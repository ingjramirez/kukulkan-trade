"""Alpaca trade executor using alpaca-py.

REST-based executor for Alpaca paper/live trading.
Same 3-method interface as PaperTrader and IBKRExecutor.
"""

import asyncio
from datetime import date

import structlog
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide as AlpacaSide
from alpaca.trading.enums import TimeInForce
from alpaca.trading.requests import MarketOrderRequest

from src.storage.database import Database
from src.storage.models import TradeSchema

log = structlog.get_logger()

# Terminal order states that stop fill polling
_TERMINAL_STATES = frozenset({"filled", "canceled", "expired", "rejected"})
_PARTIAL_FILL_STATE = "partially_filled"


class AlpacaExecutor:
    """Executes trades via Alpaca REST API and logs to our database.

    Alpaca sees one account. Our DB tracks the A/B split
    using the portfolio field on each trade and position.
    """

    def __init__(
        self,
        db: Database,
        client: TradingClient,
        fill_timeout: float = 30.0,
        fill_poll_interval: float = 1.0,
    ) -> None:
        self._db = db
        self._client = client
        self._fill_timeout = fill_timeout
        self._fill_poll_interval = fill_poll_interval

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

    async def _wait_for_fill(self, order_id: str) -> dict:
        """Poll Alpaca for order terminal state.

        Args:
            order_id: Alpaca order ID to poll.

        Returns:
            Dict with status, filled_qty, filled_avg_price.
        """
        elapsed = 0.0
        while elapsed < self._fill_timeout:
            order = self._client.get_order_by_id(order_id)
            status = str(order.status).lower()

            if status in _TERMINAL_STATES or status == _PARTIAL_FILL_STATE:
                filled_qty = float(order.filled_qty) if order.filled_qty else 0.0
                filled_price = (
                    float(order.filled_avg_price) if order.filled_avg_price else None
                )
                log.info(
                    "alpaca_fill_resolved",
                    order_id=order_id,
                    status=status,
                    filled_qty=filled_qty,
                    filled_price=filled_price,
                )
                return {
                    "status": status,
                    "filled_qty": filled_qty,
                    "filled_avg_price": filled_price,
                }

            await asyncio.sleep(self._fill_poll_interval)
            elapsed += self._fill_poll_interval

        log.warning("alpaca_fill_timeout", order_id=order_id, timeout=self._fill_timeout)
        return {"status": "timeout", "filled_qty": 0.0, "filled_avg_price": None}

    async def _execute_single(self, trade: TradeSchema) -> bool:
        """Execute a single trade via Alpaca with fill verification.

        Args:
            trade: Validated trade signal.

        Returns:
            True if executed successfully (filled or partially filled).
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

            log.info(
                "alpaca_order_submitted",
                ticker=trade.ticker,
                side=trade.side.value,
                shares=int(trade.shares),
                order_id=str(order.id),
                status=str(order.status),
            )

            # Wait for fill
            fill = await self._wait_for_fill(str(order.id))
            fill_status = fill["status"]

            if fill_status in ("rejected", "canceled", "expired", "timeout"):
                log.warning(
                    "alpaca_order_not_filled",
                    order_id=str(order.id),
                    status=fill_status,
                    ticker=trade.ticker,
                )
                return False

            # Use actual fill data
            filled_qty = fill["filled_qty"]
            fill_price = fill["filled_avg_price"] or trade.price

            if filled_qty <= 0:
                return False

            await self._update_portfolio_state(
                portfolio_name=portfolio_name,
                ticker=trade.ticker,
                side=trade.side.value,
                filled_shares=filled_qty,
                fill_price=fill_price,
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

    async def _update_portfolio_state(
        self,
        portfolio_name: str,
        ticker: str,
        side: str,
        filled_shares: float,
        fill_price: float,
        reason: str,
    ) -> None:
        """Update DB portfolio, positions, and trade log after a fill.

        Args:
            portfolio_name: A or B.
            ticker: Ticker symbol.
            side: BUY or SELL.
            filled_shares: Actual filled quantity.
            fill_price: Actual average fill price.
            reason: Trade reason string.
        """
        portfolio = await self._db.get_portfolio(portfolio_name)
        if portfolio is None:
            return

        positions = await self._db.get_positions(portfolio_name)
        position_map = {p.ticker: p for p in positions}

        if side == "BUY":
            cost = filled_shares * fill_price
            existing = position_map.get(ticker)
            if existing:
                total_shares = existing.shares + filled_shares
                total_cost = (existing.shares * existing.avg_price) + cost
                new_avg = total_cost / total_shares
                await self._db.upsert_position(
                    portfolio_name, ticker, total_shares, new_avg
                )
            else:
                await self._db.upsert_position(
                    portfolio_name, ticker, filled_shares, fill_price
                )
            await self._db.upsert_portfolio(
                portfolio_name,
                cash=portfolio.cash - cost,
                total_value=portfolio.total_value,
            )
        elif side == "SELL":
            existing = position_map.get(ticker)
            if existing is None:
                return
            remaining = existing.shares - filled_shares
            await self._db.upsert_position(
                portfolio_name, ticker, remaining, existing.avg_price
            )
            proceeds = filled_shares * fill_price
            await self._db.upsert_portfolio(
                portfolio_name,
                cash=portfolio.cash + proceeds,
                total_value=portfolio.total_value,
            )

        # Log trade
        await self._db.log_trade(
            portfolio=portfolio_name,
            ticker=ticker,
            side=side,
            shares=filled_shares,
            price=fill_price,
            reason=reason,
        )

    async def sync_positions(self) -> dict[str, list[dict]]:
        """Compare Alpaca account positions with our DB and log drift.

        Returns:
            Dict with 'alpaca' and 'drift' keys.
        """
        try:
            alpaca_positions = self._client.get_all_positions()
        except Exception as e:
            log.error("alpaca_sync_failed", error=str(e))
            return {"alpaca": [], "drift": []}

        alpaca_map: dict[str, float] = {}
        for pos in alpaca_positions:
            alpaca_map[pos.symbol] = float(pos.qty)

        # Build our combined DB positions across A and B
        db_map: dict[str, float] = {}
        for pname in ("A", "B"):
            positions = await self._db.get_positions(pname)
            for p in positions:
                db_map[p.ticker] = db_map.get(p.ticker, 0) + p.shares

        # Detect drift
        all_tickers = set(alpaca_map) | set(db_map)
        drift: list[dict] = []
        for ticker in sorted(all_tickers):
            alpaca_qty = alpaca_map.get(ticker, 0)
            db_qty = db_map.get(ticker, 0)
            if abs(alpaca_qty - db_qty) > 0.01:
                entry = {
                    "ticker": ticker,
                    "alpaca_qty": alpaca_qty,
                    "db_qty": db_qty,
                    "diff": alpaca_qty - db_qty,
                }
                drift.append(entry)
                log.warning("position_drift_detected", **entry)

        if not drift:
            log.info("positions_in_sync", tickers=len(all_tickers))

        return {
            "alpaca": [{"symbol": t, "qty": q} for t, q in alpaca_map.items()],
            "drift": drift,
        }

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

        initial_value = (
            PORTFOLIO_A.allocation_usd
            if portfolio_name == "A"
            else PORTFOLIO_B.allocation_usd
        )

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

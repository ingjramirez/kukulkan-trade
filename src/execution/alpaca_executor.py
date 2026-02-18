"""Alpaca trade executor using alpaca-py.

REST-based executor for Alpaca paper/live trading.
Same 3-method interface as PaperTrader.
"""

import asyncio
from datetime import date, datetime
from typing import Any

import structlog
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide as AlpacaSide
from alpaca.trading.enums import TimeInForce
from alpaca.trading.requests import MarketOrderRequest

from src.storage.database import Database
from src.storage.models import TradeSchema
from src.utils.allocations import DEFAULT_ALLOCATIONS, TenantAllocations
from src.utils.retry import retry_broker_read
from src.utils.ticker_mapping import is_crypto_ticker, to_alpaca_format, to_canonical_format

log = structlog.get_logger()

# Terminal order states that stop fill polling
_TERMINAL_STATES = frozenset({"filled", "canceled", "expired", "rejected"})
_PARTIAL_FILL_STATE = "partially_filled"


def _order_status(raw_status) -> str:
    """Extract lowercase status string from Alpaca OrderStatus enum or string.

    Alpaca SDK returns enums like OrderStatus.FILLED where str() gives
    'orderstatus.filled'. We need just 'filled'.
    """
    s = str(raw_status).lower()
    return s.rsplit(".", 1)[-1] if "." in s else s


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
        fill_poll_interval: float = 5.0,
    ) -> None:
        self._db = db
        self._client = client
        self._fill_timeout = fill_timeout
        self._fill_poll_interval = fill_poll_interval
        # Track timed-out orders for reconciliation
        self._pending_orders: list[dict] = []

    @retry_broker_read
    def _fetch_alpaca_order(self, order_id: str) -> Any:
        """Fetch a single order by ID (with retry on transient errors)."""
        return self._client.get_order_by_id(order_id)

    @retry_broker_read
    def _fetch_alpaca_positions(self) -> list:
        """Fetch all positions from Alpaca (with retry on transient errors)."""
        return self._client.get_all_positions()

    async def initialize_portfolios(
        self,
        allocations: TenantAllocations | None = None,
        tenant_id: str = "default",
    ) -> None:
        """Create portfolio rows if they don't exist."""
        alloc = allocations or DEFAULT_ALLOCATIONS
        for name, cash in [
            ("A", alloc.portfolio_a_cash),
            ("B", alloc.portfolio_b_cash),
        ]:
            existing = await self._db.get_portfolio(name, tenant_id=tenant_id)
            if existing is None:
                await self._db.upsert_portfolio(
                    name,
                    cash=cash,
                    total_value=cash,
                    tenant_id=tenant_id,
                )
                log.info("portfolio_initialized_alpaca", portfolio=name, cash=cash)

    async def execute_trades(
        self,
        trades: list[TradeSchema],
        tenant_id: str = "default",
    ) -> list[TradeSchema]:
        """Submit market orders to Alpaca and log fills to DB.

        Processes sells before buys to free up cash.
        After all orders, reconciles any that timed out during polling.

        Args:
            trades: List of validated TradeSchema objects.
            tenant_id: Tenant UUID (unused — Alpaca handles its own state).

        Returns:
            List of successfully executed trades.
        """
        sells = [t for t in trades if t.side.value == "SELL"]
        buys = [t for t in trades if t.side.value == "BUY"]
        executed: list[TradeSchema] = []
        self._pending_orders.clear()

        for trade in sells + buys:
            success = await self._execute_single(trade)
            if success:
                executed.append(trade)

        # Reconcile timed-out orders that may have filled after timeout
        if self._pending_orders:
            reconciled = await self._reconcile_pending_orders()
            executed.extend(reconciled)

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
            order = await asyncio.to_thread(
                self._fetch_alpaca_order,
                order_id,
            )
            status = _order_status(order.status)

            if status in _TERMINAL_STATES or status == _PARTIAL_FILL_STATE:
                filled_qty = float(order.filled_qty) if order.filled_qty else 0.0
                filled_price = float(order.filled_avg_price) if order.filled_avg_price else None
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
        crypto = is_crypto_ticker(trade.ticker)

        try:
            alpaca_symbol = to_alpaca_format(trade.ticker)
            qty = round(trade.shares, 8) if crypto else int(trade.shares)
            tif = TimeInForce.GTC if crypto else TimeInForce.DAY

            order_request = MarketOrderRequest(
                symbol=alpaca_symbol,
                qty=qty,
                side=side,
                time_in_force=tif,
                client_order_id=f"kk-{portfolio_name}-{trade.ticker}-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
            )
            order = self._client.submit_order(order_request)

            log.info(
                "alpaca_order_submitted",
                ticker=trade.ticker,
                side=trade.side.value,
                shares=qty,
                order_id=str(order.id),
                status=str(order.status),
                crypto=crypto,
            )

            # Wait for fill
            fill = await self._wait_for_fill(str(order.id))
            fill_status = fill["status"]

            if fill_status in ("rejected", "canceled", "expired"):
                log.warning(
                    "alpaca_order_not_filled",
                    order_id=str(order.id),
                    status=fill_status,
                    ticker=trade.ticker,
                )
                return False

            if fill_status == "timeout":
                log.warning(
                    "alpaca_order_timeout_will_reconcile",
                    order_id=str(order.id),
                    ticker=trade.ticker,
                )
                self._pending_orders.append(
                    {
                        "order_id": str(order.id),
                        "trade": trade,
                    }
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
                await self._db.upsert_position(portfolio_name, ticker, total_shares, new_avg)
            else:
                await self._db.upsert_position(portfolio_name, ticker, filled_shares, fill_price)
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
            await self._db.upsert_position(portfolio_name, ticker, remaining, existing.avg_price)
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

    async def _reconcile_pending_orders(self) -> list[TradeSchema]:
        """Check timed-out orders for late fills and log them to DB.

        Returns:
            List of trades that were reconciled (filled after timeout).
        """
        if not self._pending_orders:
            return []

        log.info("reconciliation_starting", pending_orders=len(self._pending_orders))
        reconciled: list[TradeSchema] = []

        for entry in self._pending_orders:
            order_id = entry["order_id"]
            trade: TradeSchema = entry["trade"]
            try:
                order = await asyncio.to_thread(self._fetch_alpaca_order, order_id)
                status = _order_status(order.status)
                filled_qty = float(order.filled_qty) if order.filled_qty else 0.0
                filled_price = float(order.filled_avg_price) if order.filled_avg_price else None

                if status in ("filled", "partially_filled") and filled_qty > 0:
                    fill_price = filled_price or trade.price
                    await self._update_portfolio_state(
                        portfolio_name=trade.portfolio.value,
                        ticker=trade.ticker,
                        side=trade.side.value,
                        filled_shares=filled_qty,
                        fill_price=fill_price,
                        reason=trade.reason,
                    )
                    reconciled.append(trade)
                    log.info(
                        "reconciliation_filled",
                        order_id=order_id,
                        ticker=trade.ticker,
                        filled_qty=filled_qty,
                        filled_price=fill_price,
                    )
                else:
                    log.warning(
                        "reconciliation_still_unfilled",
                        order_id=order_id,
                        ticker=trade.ticker,
                        status=status,
                    )
            except Exception as e:
                log.error(
                    "reconciliation_check_failed",
                    order_id=order_id,
                    ticker=trade.ticker,
                    error=str(e),
                )

        self._pending_orders.clear()
        log.info(
            "reconciliation_complete",
            checked=len(self._pending_orders) + len(reconciled),
            reconciled=len(reconciled),
        )
        return reconciled

    async def sync_positions(self) -> dict[str, list[dict]]:
        """Sync DB positions to match Alpaca's actual state.

        Alpaca is the source of truth. When drift is detected, DB is
        corrected to match. Positions are assigned to whichever portfolio
        already owns them in the DB; unknown positions default to B.

        Also syncs portfolio cash from the Alpaca account balance.

        Returns:
            Dict with 'alpaca', 'drift', and 'corrections' keys.
        """
        try:
            alpaca_positions = self._fetch_alpaca_positions()
        except Exception as e:
            log.error("alpaca_sync_failed", error=str(e))
            return {"alpaca": [], "drift": [], "corrections": []}

        alpaca_map: dict[str, float] = {}
        alpaca_price_map: dict[str, float] = {}
        for pos in alpaca_positions:
            qty = float(pos.qty)
            # Skip short positions (negative qty) — bot doesn't short
            if qty <= 0:
                log.warning(
                    "alpaca_short_position_skipped",
                    ticker=pos.symbol,
                    qty=qty,
                )
                continue
            canonical = to_canonical_format(pos.symbol)
            alpaca_map[canonical] = qty
            alpaca_price_map[canonical] = float(pos.avg_entry_price)

        # Build DB position map: ticker -> (portfolio, shares, avg_price)
        db_positions: dict[str, dict] = {}
        for pname in ("A", "B"):
            positions = await self._db.get_positions(pname)
            for p in positions:
                if p.ticker not in db_positions:
                    db_positions[p.ticker] = {
                        "portfolio": pname,
                        "shares": p.shares,
                        "avg_price": p.avg_price,
                    }
                else:
                    # Ticker in both portfolios — accumulate
                    db_positions[p.ticker]["shares"] += p.shares

        # Detect drift
        all_tickers = set(alpaca_map) | set(db_positions)
        drift: list[dict] = []
        corrections: list[dict] = []

        for ticker in sorted(all_tickers):
            alpaca_qty = alpaca_map.get(ticker, 0)
            db_entry = db_positions.get(ticker)
            db_qty = db_entry["shares"] if db_entry else 0

            if abs(alpaca_qty - db_qty) < 0.01:
                continue

            entry = {
                "ticker": ticker,
                "alpaca_qty": alpaca_qty,
                "db_qty": db_qty,
                "diff": alpaca_qty - db_qty,
            }
            drift.append(entry)
            log.warning("position_drift_detected", **entry)

            # Correct DB to match Alpaca
            portfolio = db_entry["portfolio"] if db_entry else "B"
            avg_price = alpaca_price_map.get(ticker) or (db_entry["avg_price"] if db_entry else 0)

            await self._db.upsert_position(
                portfolio,
                ticker,
                alpaca_qty,
                avg_price,
            )
            corrections.append(
                {
                    "ticker": ticker,
                    "portfolio": portfolio,
                    "old_qty": db_qty,
                    "new_qty": alpaca_qty,
                }
            )
            log.info(
                "position_drift_corrected",
                ticker=ticker,
                portfolio=portfolio,
                old_qty=db_qty,
                new_qty=alpaca_qty,
            )

        # Log Alpaca cash for reference (but do NOT override portfolio cash —
        # each portfolio tracks its own cash through trade execution)
        try:
            account = self._client.get_account()
            alpaca_cash = float(account.cash)
            log.info("alpaca_cash_reference", total_cash=alpaca_cash)
        except Exception as e:
            log.warning("alpaca_cash_read_failed", error=str(e))

        if not drift:
            log.info("positions_in_sync", tickers=len(all_tickers))
        else:
            log.info(
                "position_sync_complete",
                drift_count=len(drift),
                corrections=len(corrections),
            )

        return {
            "alpaca": [{"symbol": t, "qty": q} for t, q in alpaca_map.items()],
            "drift": drift,
            "corrections": corrections,
        }

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

        portfolio = await self._db.get_portfolio(portfolio_name, tenant_id=tenant_id)
        if portfolio is None:
            return

        positions = await self._db.get_positions(portfolio_name, tenant_id=tenant_id)
        positions_value = sum(p.shares * prices.get(p.ticker, p.avg_price) for p in positions)
        total_value = portfolio.cash + positions_value

        initial_value = alloc.for_portfolio(portfolio_name)

        snapshots = await self._db.get_snapshots(
            portfolio_name,
            tenant_id=tenant_id,
        )
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
            tenant_id=tenant_id,
        )

        await self._db.update_position_prices(
            portfolio_name,
            prices,
            tenant_id=tenant_id,
        )
        await self._db.upsert_portfolio(
            portfolio_name,
            portfolio.cash,
            total_value,
            tenant_id=tenant_id,
        )

        log.info(
            "alpaca_snapshot_taken",
            portfolio=portfolio_name,
            total_value=round(total_value, 2),
            daily_return=round(daily_return_pct, 2) if daily_return_pct else None,
        )

    async def get_open_orders(self) -> list[dict]:
        """List open orders from Alpaca (for sentinel fill verification).

        Returns:
            List of dicts with order_id, ticker, status, qty, filled_qty, created_at.
        """
        from alpaca.trading.enums import QueryOrderStatus
        from alpaca.trading.requests import GetOrdersRequest

        def _fetch() -> list:
            request = GetOrdersRequest(status=QueryOrderStatus.OPEN)
            return self._client.get_orders(filter=request)

        raw_orders = await asyncio.to_thread(_fetch)

        return [
            {
                "order_id": str(o.id),
                "ticker": to_canonical_format(o.symbol),
                "status": _order_status(o.status),
                "qty": float(o.qty) if o.qty else 0,
                "filled_qty": float(o.filled_qty) if o.filled_qty else 0,
                "created_at": o.created_at,
            }
            for o in raw_orders
        ]

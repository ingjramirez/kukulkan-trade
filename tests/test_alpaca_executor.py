"""Tests for Alpaca executor — mocked TradingClient, real in-memory DB."""

from datetime import date
from unittest.mock import MagicMock

import pytest

from src.execution.alpaca_executor import AlpacaExecutor
from src.storage.database import Database
from src.storage.models import OrderSide, PortfolioName, TradeSchema


def _make_trade(
    ticker: str = "XLK",
    side: OrderSide = OrderSide.BUY,
    shares: float = 10,
    price: float = 200.0,
    portfolio: PortfolioName = PortfolioName.A,
    reason: str = "test",
) -> TradeSchema:
    return TradeSchema(
        ticker=ticker, side=side, shares=shares, price=price,
        portfolio=portfolio, reason=reason,
    )


def _mock_order(filled_avg_price=200.0, order_id="abc123", status="filled",
                filled_qty=None):
    """Create a mock Alpaca order response."""
    order = MagicMock()
    order.id = order_id
    order.status = status
    order.filled_avg_price = filled_avg_price
    order.filled_qty = filled_qty if filled_qty is not None else (
        10 if status == "filled" else 0
    )
    return order


@pytest.fixture
async def db():
    database = Database(url="sqlite+aiosqlite:///:memory:")
    await database.init_db()
    yield database
    await database.close()


@pytest.fixture
def mock_client():
    client = MagicMock()
    # submit_order returns initial order with "new" status
    client.submit_order.return_value = _mock_order(
        filled_avg_price=None, status="new", filled_qty=0,
    )
    # get_order_by_id returns filled order on first poll
    client.get_order_by_id.return_value = _mock_order(
        filled_avg_price=200.0, status="filled", filled_qty=10,
    )
    return client


class TestAlpacaExecutorInit:
    async def test_initialize_portfolios(self, db: Database, mock_client) -> None:
        executor = AlpacaExecutor(db, mock_client)
        await executor.initialize_portfolios()

        port_a = await db.get_portfolio("A")
        port_b = await db.get_portfolio("B")
        assert port_a is not None
        assert port_b is not None
        assert port_a.cash == 33_000.0
        assert port_b.cash == 66_000.0

    async def test_initialize_idempotent(self, db: Database, mock_client) -> None:
        executor = AlpacaExecutor(db, mock_client)
        await executor.initialize_portfolios()
        await executor.initialize_portfolios()

        port_a = await db.get_portfolio("A")
        assert port_a is not None
        assert port_a.cash == 33_000.0


class TestAlpacaExecutorTrades:
    async def test_execute_buy(self, db: Database, mock_client) -> None:
        executor = AlpacaExecutor(db, mock_client, fill_timeout=2, fill_poll_interval=0.01)
        await executor.initialize_portfolios()

        trade = _make_trade(ticker="XLK", side=OrderSide.BUY, shares=10, price=200.0)
        executed = await executor.execute_trades([trade])

        assert len(executed) == 1
        mock_client.submit_order.assert_called_once()

        # Check position created
        positions = await db.get_positions("A")
        assert len(positions) == 1
        assert positions[0].ticker == "XLK"
        assert positions[0].shares == 10

    async def test_execute_buy_updates_cash(self, db: Database, mock_client) -> None:
        executor = AlpacaExecutor(db, mock_client, fill_timeout=2, fill_poll_interval=0.01)
        await executor.initialize_portfolios()

        trade = _make_trade(ticker="XLK", side=OrderSide.BUY, shares=10, price=200.0)
        await executor.execute_trades([trade])

        portfolio = await db.get_portfolio("A")
        # 33000 - (10 * 200) = 31000
        assert portfolio.cash == 31_000.0

    async def test_execute_sell(self, db: Database, mock_client) -> None:
        mock_client.get_order_by_id.return_value = _mock_order(
            filled_avg_price=210.0, status="filled", filled_qty=5,
        )
        executor = AlpacaExecutor(db, mock_client, fill_timeout=2, fill_poll_interval=0.01)
        await executor.initialize_portfolios()

        await db.upsert_position("A", "XLK", 10, 200.0)
        trade = _make_trade(ticker="XLK", side=OrderSide.SELL, shares=5, price=210.0)
        executed = await executor.execute_trades([trade])

        assert len(executed) == 1

        # Check position updated
        positions = await db.get_positions("A")
        xlk = next(p for p in positions if p.ticker == "XLK")
        assert xlk.shares == 5

    async def test_execute_sell_updates_cash(self, db: Database, mock_client) -> None:
        mock_client.get_order_by_id.return_value = _mock_order(
            filled_avg_price=210.0, status="filled", filled_qty=5,
        )
        executor = AlpacaExecutor(db, mock_client, fill_timeout=2, fill_poll_interval=0.01)
        await executor.initialize_portfolios()

        await db.upsert_position("A", "XLK", 10, 200.0)
        trade = _make_trade(ticker="XLK", side=OrderSide.SELL, shares=5, price=210.0)
        await executor.execute_trades([trade])

        portfolio = await db.get_portfolio("A")
        # 33000 + (5 * 210) = 34050
        assert portfolio.cash == 34_050.0

    async def test_sells_before_buys(self, db: Database, mock_client) -> None:
        mock_client.get_order_by_id.return_value = _mock_order(
            filled_avg_price=200.0, status="filled", filled_qty=5,
        )
        executor = AlpacaExecutor(db, mock_client, fill_timeout=2, fill_poll_interval=0.01)
        await executor.initialize_portfolios()
        await db.upsert_position("A", "XLK", 10, 200.0)

        sell = _make_trade(ticker="XLK", side=OrderSide.SELL, shares=5, price=200.0)
        buy = _make_trade(ticker="AAPL", side=OrderSide.BUY, shares=3, price=150.0)
        await executor.execute_trades([buy, sell])

        # Sell should be submitted first
        calls = mock_client.submit_order.call_args_list
        assert len(calls) == 2
        first_order = calls[0][0][0]
        assert first_order.symbol == "XLK"

    async def test_order_failure_returns_empty(self, db: Database, mock_client) -> None:
        mock_client.submit_order.side_effect = Exception("API error")
        executor = AlpacaExecutor(db, mock_client, fill_timeout=2, fill_poll_interval=0.01)
        await executor.initialize_portfolios()

        trade = _make_trade()
        executed = await executor.execute_trades([trade])

        assert len(executed) == 0

    async def test_uses_fill_price(self, db: Database, mock_client) -> None:
        mock_client.get_order_by_id.return_value = _mock_order(
            filled_avg_price=205.0, status="filled", filled_qty=10,
        )
        executor = AlpacaExecutor(db, mock_client, fill_timeout=2, fill_poll_interval=0.01)
        await executor.initialize_portfolios()

        trade = _make_trade(ticker="XLK", side=OrderSide.BUY, shares=10, price=200.0)
        await executor.execute_trades([trade])

        # Position avg_price should be fill price, not estimated
        positions = await db.get_positions("A")
        assert positions[0].avg_price == 205.0

    async def test_no_fill_price_uses_estimated(self, db: Database, mock_client) -> None:
        mock_client.get_order_by_id.return_value = _mock_order(
            filled_avg_price=None, status="filled", filled_qty=10,
        )
        executor = AlpacaExecutor(db, mock_client, fill_timeout=2, fill_poll_interval=0.01)
        await executor.initialize_portfolios()

        trade = _make_trade(ticker="XLK", side=OrderSide.BUY, shares=10, price=200.0)
        await executor.execute_trades([trade])

        positions = await db.get_positions("A")
        assert positions[0].avg_price == 200.0

    async def test_trade_logged_to_db(self, db: Database, mock_client) -> None:
        executor = AlpacaExecutor(db, mock_client, fill_timeout=2, fill_poll_interval=0.01)
        await executor.initialize_portfolios()

        trade = _make_trade(
            ticker="XLK", side=OrderSide.BUY,
            shares=10, price=200.0, reason="momentum",
        )
        await executor.execute_trades([trade])

        trades = await db.get_trades("A")
        assert len(trades) == 1
        assert trades[0].ticker == "XLK"
        assert trades[0].side == "BUY"
        assert trades[0].reason == "momentum"

    async def test_order_id_format(self, db: Database, mock_client) -> None:
        executor = AlpacaExecutor(db, mock_client, fill_timeout=2, fill_poll_interval=0.01)
        await executor.initialize_portfolios()

        trade = _make_trade(ticker="AAPL", portfolio=PortfolioName.B)
        await executor.execute_trades([trade])

        order_request = mock_client.submit_order.call_args[0][0]
        assert order_request.client_order_id == "kk-B-AAPL"


class TestAlpacaFillPolling:
    """Tests for the fill verification polling loop."""

    async def test_wait_for_fill_immediate(self, db: Database, mock_client) -> None:
        """Order fills on first poll."""
        mock_client.get_order_by_id.return_value = _mock_order(
            filled_avg_price=200.0, status="filled", filled_qty=10,
        )
        executor = AlpacaExecutor(db, mock_client, fill_timeout=5, fill_poll_interval=0.01)

        result = await executor._wait_for_fill("order-123")

        assert result["status"] == "filled"
        assert result["filled_qty"] == 10
        assert result["filled_avg_price"] == 200.0

    async def test_wait_for_fill_partial(self, db: Database, mock_client) -> None:
        """Partial fill is accepted."""
        mock_client.get_order_by_id.return_value = _mock_order(
            filled_avg_price=200.0, status="partially_filled", filled_qty=5,
        )
        executor = AlpacaExecutor(db, mock_client, fill_timeout=5, fill_poll_interval=0.01)
        await executor.initialize_portfolios()

        trade = _make_trade(ticker="XLK", side=OrderSide.BUY, shares=10, price=200.0)
        executed = await executor.execute_trades([trade])

        assert len(executed) == 1
        positions = await db.get_positions("A")
        assert positions[0].shares == 5  # Only partial fill qty

    async def test_wait_for_fill_rejected(self, db: Database, mock_client) -> None:
        """Rejected order returns False."""
        mock_client.get_order_by_id.return_value = _mock_order(
            filled_avg_price=None, status="rejected", filled_qty=0,
        )
        executor = AlpacaExecutor(db, mock_client, fill_timeout=5, fill_poll_interval=0.01)
        await executor.initialize_portfolios()

        trade = _make_trade()
        executed = await executor.execute_trades([trade])

        assert len(executed) == 0

    async def test_wait_for_fill_timeout(self, db: Database, mock_client) -> None:
        """Timeout when order stays in 'new' state."""
        mock_client.get_order_by_id.return_value = _mock_order(
            filled_avg_price=None, status="new", filled_qty=0,
        )
        executor = AlpacaExecutor(db, mock_client, fill_timeout=0.05, fill_poll_interval=0.01)
        await executor.initialize_portfolios()

        trade = _make_trade()
        executed = await executor.execute_trades([trade])

        assert len(executed) == 0

    async def test_wait_for_fill_canceled(self, db: Database, mock_client) -> None:
        """Canceled order returns False."""
        mock_client.get_order_by_id.return_value = _mock_order(
            filled_avg_price=None, status="canceled", filled_qty=0,
        )
        executor = AlpacaExecutor(db, mock_client, fill_timeout=5, fill_poll_interval=0.01)
        await executor.initialize_portfolios()

        trade = _make_trade()
        executed = await executor.execute_trades([trade])

        assert len(executed) == 0

    async def test_wait_for_fill_expired(self, db: Database, mock_client) -> None:
        """Expired order returns False."""
        mock_client.get_order_by_id.return_value = _mock_order(
            filled_avg_price=None, status="expired", filled_qty=0,
        )
        executor = AlpacaExecutor(db, mock_client, fill_timeout=5, fill_poll_interval=0.01)
        await executor.initialize_portfolios()

        trade = _make_trade()
        executed = await executor.execute_trades([trade])

        assert len(executed) == 0


class TestAlpacaSyncPositions:
    """Tests for the position sync/drift detection."""

    async def test_sync_no_drift(self, db: Database, mock_client) -> None:
        """No drift when Alpaca and DB match."""
        executor = AlpacaExecutor(db, mock_client, fill_timeout=2, fill_poll_interval=0.01)
        await executor.initialize_portfolios()
        await db.upsert_position("A", "XLK", 10, 200.0)

        # Mock Alpaca positions
        mock_pos = MagicMock()
        mock_pos.symbol = "XLK"
        mock_pos.qty = "10"
        mock_client.get_all_positions.return_value = [mock_pos]

        result = await executor.sync_positions()
        assert len(result["drift"]) == 0
        assert len(result["alpaca"]) == 1

    async def test_sync_detects_drift(self, db: Database, mock_client) -> None:
        """Drift detected when quantities differ."""
        executor = AlpacaExecutor(db, mock_client, fill_timeout=2, fill_poll_interval=0.01)
        await executor.initialize_portfolios()
        await db.upsert_position("A", "XLK", 10, 200.0)

        # Mock Alpaca with different quantity
        mock_pos = MagicMock()
        mock_pos.symbol = "XLK"
        mock_pos.qty = "15"
        mock_client.get_all_positions.return_value = [mock_pos]

        result = await executor.sync_positions()
        assert len(result["drift"]) == 1
        assert result["drift"][0]["ticker"] == "XLK"
        assert result["drift"][0]["alpaca_qty"] == 15.0
        assert result["drift"][0]["db_qty"] == 10.0

    async def test_sync_handles_api_error(self, db: Database, mock_client) -> None:
        """Graceful handling when Alpaca API fails."""
        mock_client.get_all_positions.side_effect = Exception("API down")
        executor = AlpacaExecutor(db, mock_client, fill_timeout=2, fill_poll_interval=0.01)

        result = await executor.sync_positions()
        assert result == {"alpaca": [], "drift": []}


class TestAlpacaExecutorSnapshot:
    async def test_take_snapshot(self, db: Database, mock_client) -> None:
        executor = AlpacaExecutor(db, mock_client)
        await executor.initialize_portfolios()

        prices = {"XLK": 200.0}
        await executor.take_snapshot("A", date(2026, 2, 6), prices)

        snapshots = await db.get_snapshots("A")
        assert len(snapshots) == 1
        assert snapshots[0].total_value > 0

    async def test_snapshot_with_positions(self, db: Database, mock_client) -> None:
        executor = AlpacaExecutor(db, mock_client)
        await executor.initialize_portfolios()
        await db.upsert_position("A", "XLK", 10, 200.0)
        await db.upsert_portfolio("A", cash=31_000.0, total_value=33_000.0)

        prices = {"XLK": 210.0}
        await executor.take_snapshot("A", date(2026, 2, 6), prices)

        snapshots = await db.get_snapshots("A")
        assert len(snapshots) == 1
        # 31000 cash + 10*210 = 33100
        assert snapshots[0].total_value == 33100.0

    async def test_snapshot_cumulative_return(self, db: Database, mock_client) -> None:
        executor = AlpacaExecutor(db, mock_client)
        await executor.initialize_portfolios()

        await executor.take_snapshot("A", date(2026, 2, 6), {})

        snapshots = await db.get_snapshots("A")
        # Initial value = 33000, current = 33000 → 0%
        assert snapshots[0].cumulative_return_pct == 0.0


class TestAlpacaExecutorInterface:
    async def test_same_interface_as_paper_trader(self) -> None:
        """AlpacaExecutor has the same 3 methods as PaperTrader."""
        from src.execution.paper_trader import PaperTrader

        for method in ("initialize_portfolios", "execute_trades", "take_snapshot"):
            assert hasattr(AlpacaExecutor, method)
            assert hasattr(PaperTrader, method)

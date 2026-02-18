"""Tests for Alpaca executor crypto order support."""

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
    portfolio: PortfolioName = PortfolioName.B,
    reason: str = "test",
) -> TradeSchema:
    return TradeSchema(ticker=ticker, side=side, shares=shares, price=price, portfolio=portfolio, reason=reason)


def _mock_order(filled_avg_price=200.0, order_id="abc123", status="filled", filled_qty=None, qty=10):
    order = MagicMock()
    order.id = order_id
    order.status = status
    order.filled_avg_price = filled_avg_price
    order.filled_qty = filled_qty if filled_qty is not None else (qty if status == "filled" else 0)
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
    client.submit_order.return_value = _mock_order(filled_avg_price=None, status="new", filled_qty=0)
    client.get_order_by_id.return_value = _mock_order(
        filled_avg_price=95000.0, status="filled", filled_qty=0.05, qty=0.05
    )
    return client


class TestCryptoOrderFormat:
    async def test_crypto_order_uses_alpaca_format(self, db: Database, mock_client) -> None:
        """BTC-USD should be sent to Alpaca as BTC/USD."""
        executor = AlpacaExecutor(db, mock_client, fill_timeout=0.1, fill_poll_interval=0.05)
        await executor.initialize_portfolios()

        trade = _make_trade(ticker="BTC-USD", shares=0.05, price=95000.0)
        await executor.execute_trades([trade])

        call_args = mock_client.submit_order.call_args
        order_request = call_args[0][0]
        assert order_request.symbol == "BTC/USD"

    async def test_crypto_order_uses_gtc(self, db: Database, mock_client) -> None:
        """Crypto orders should use GTC time-in-force, not DAY."""
        from alpaca.trading.enums import TimeInForce

        executor = AlpacaExecutor(db, mock_client, fill_timeout=0.1, fill_poll_interval=0.05)
        await executor.initialize_portfolios()

        trade = _make_trade(ticker="BTC-USD", shares=0.05, price=95000.0)
        await executor.execute_trades([trade])

        call_args = mock_client.submit_order.call_args
        order_request = call_args[0][0]
        assert order_request.time_in_force == TimeInForce.GTC

    async def test_equity_order_uses_day(self, db: Database, mock_client) -> None:
        """Equity orders should still use DAY time-in-force."""
        from alpaca.trading.enums import TimeInForce

        mock_client.get_order_by_id.return_value = _mock_order(
            filled_avg_price=200.0, status="filled", filled_qty=10, qty=10
        )
        executor = AlpacaExecutor(db, mock_client, fill_timeout=0.1, fill_poll_interval=0.05)
        await executor.initialize_portfolios()

        trade = _make_trade(ticker="XLK", shares=10, price=200.0)
        await executor.execute_trades([trade])

        call_args = mock_client.submit_order.call_args
        order_request = call_args[0][0]
        assert order_request.time_in_force == TimeInForce.DAY

    async def test_crypto_fractional_qty(self, db: Database, mock_client) -> None:
        """Crypto should preserve fractional qty."""
        executor = AlpacaExecutor(db, mock_client, fill_timeout=0.1, fill_poll_interval=0.05)
        await executor.initialize_portfolios()

        trade = _make_trade(ticker="BTC-USD", shares=0.05263158, price=95000.0)
        await executor.execute_trades([trade])

        call_args = mock_client.submit_order.call_args
        order_request = call_args[0][0]
        # Should be rounded to 8 decimal places, not truncated to int
        assert isinstance(order_request.qty, float)
        assert order_request.qty == round(0.05263158, 8)

    async def test_equity_whole_shares(self, db: Database, mock_client) -> None:
        """Equity orders should convert to integer shares."""
        mock_client.get_order_by_id.return_value = _mock_order(
            filled_avg_price=200.0, status="filled", filled_qty=10, qty=10
        )
        executor = AlpacaExecutor(db, mock_client, fill_timeout=0.1, fill_poll_interval=0.05)
        await executor.initialize_portfolios()

        trade = _make_trade(ticker="XLK", shares=10.7, price=200.0)
        await executor.execute_trades([trade])

        call_args = mock_client.submit_order.call_args
        order_request = call_args[0][0]
        assert order_request.qty == 10

    async def test_equity_order_symbol_unchanged(self, db: Database, mock_client) -> None:
        """Equity tickers should not be transformed."""
        mock_client.get_order_by_id.return_value = _mock_order(
            filled_avg_price=200.0, status="filled", filled_qty=10, qty=10
        )
        executor = AlpacaExecutor(db, mock_client, fill_timeout=0.1, fill_poll_interval=0.05)
        await executor.initialize_portfolios()

        trade = _make_trade(ticker="AAPL", shares=5, price=180.0)
        await executor.execute_trades([trade])

        call_args = mock_client.submit_order.call_args
        order_request = call_args[0][0]
        assert order_request.symbol == "AAPL"


class TestPositionSyncCrypto:
    async def test_position_sync_converts_crypto_ticker(self, db: Database, mock_client) -> None:
        """Alpaca BTC/USD positions should be stored as BTC-USD in DB."""
        pos = MagicMock()
        pos.symbol = "BTC/USD"
        pos.qty = "0.05"
        pos.avg_entry_price = "95000.00"
        mock_client.get_all_positions.return_value = [pos]
        mock_client.get_account.return_value = MagicMock(cash="50000.00")

        executor = AlpacaExecutor(db, mock_client)
        await executor.initialize_portfolios()

        result = await executor.sync_positions()

        # Alpaca map should use canonical ticker
        alpaca_tickers = [e["symbol"] for e in result["alpaca"]]
        assert "BTC-USD" in alpaca_tickers
        assert "BTC/USD" not in alpaca_tickers

    async def test_open_orders_converts_crypto_ticker(self, db: Database, mock_client) -> None:
        """Open orders with BTC/USD should be returned as BTC-USD."""
        order = MagicMock()
        order.id = "order123"
        order.symbol = "BTC/USD"
        order.status = "new"
        order.qty = "0.05"
        order.filled_qty = "0"
        order.created_at = "2026-02-18T10:00:00Z"
        mock_client.get_orders.return_value = [order]

        executor = AlpacaExecutor(db, mock_client)
        result = await executor.get_open_orders()

        assert result[0]["ticker"] == "BTC-USD"

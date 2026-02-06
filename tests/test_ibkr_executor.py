"""Tests for IBKR executor — mocked IBKRClient, real in-memory DB."""

from datetime import date
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.execution.ibkr_client import IBKRClient
from src.execution.ibkr_executor import IBKRExecutor
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


@pytest.fixture
async def db():
    database = Database(url="sqlite+aiosqlite:///:memory:")
    await database.init_db()
    yield database
    await database.close()


@pytest.fixture
def mock_client():
    client = MagicMock(spec=IBKRClient)
    client.is_connected.return_value = True
    # Mock place_market_order to return a filled trade
    mock_trade = MagicMock()
    mock_trade.orderStatus.status = "Filled"
    mock_trade.orderStatus.avgFillPrice = 200.0
    mock_trade.orderStatus.filled = 10
    mock_trade.isDone.return_value = True
    client.place_market_order = AsyncMock(return_value=mock_trade)
    return client


class TestIBKRExecutorInit:
    async def test_initialize_portfolios(self, db: Database, mock_client) -> None:
        executor = IBKRExecutor(db, mock_client)
        await executor.initialize_portfolios()

        for name in ("A", "B"):
            portfolio = await db.get_portfolio(name)
            assert portfolio is not None

    async def test_initialize_idempotent(self, db: Database, mock_client) -> None:
        executor = IBKRExecutor(db, mock_client)
        await executor.initialize_portfolios()
        await executor.initialize_portfolios()

        for name in ("A", "B"):
            portfolio = await db.get_portfolio(name)
            assert portfolio is not None


class TestIBKRExecutorTrades:
    async def test_execute_buy(self, db: Database, mock_client) -> None:
        executor = IBKRExecutor(db, mock_client)
        await executor.initialize_portfolios()

        trade = _make_trade(ticker="XLK", side=OrderSide.BUY, shares=10, price=200.0)
        executed = await executor.execute_trades([trade])

        assert len(executed) == 1
        mock_client.place_market_order.assert_called_once()

    async def test_execute_sell(self, db: Database, mock_client) -> None:
        executor = IBKRExecutor(db, mock_client)
        await executor.initialize_portfolios()

        # First buy
        await db.upsert_position("A", "XLK", 10, 200.0)
        trade = _make_trade(ticker="XLK", side=OrderSide.SELL, shares=5, price=210.0)
        executed = await executor.execute_trades([trade])

        assert len(executed) == 1

    async def test_order_failure_returns_empty(self, db: Database, mock_client) -> None:
        mock_client.place_market_order = AsyncMock(return_value=None)
        executor = IBKRExecutor(db, mock_client)
        await executor.initialize_portfolios()

        trade = _make_trade()
        executed = await executor.execute_trades([trade])

        assert len(executed) == 0

    async def test_order_reference_format(self, db: Database, mock_client) -> None:
        executor = IBKRExecutor(db, mock_client)
        await executor.initialize_portfolios()

        trade = _make_trade(ticker="AAPL", portfolio=PortfolioName.B)
        await executor.execute_trades([trade])

        call_kwargs = mock_client.place_market_order.call_args[1]
        assert call_kwargs["reference"] == "atlas-B-AAPL"


class TestIBKRExecutorSnapshot:
    async def test_take_snapshot(self, db: Database, mock_client) -> None:
        executor = IBKRExecutor(db, mock_client)
        await executor.initialize_portfolios()

        prices = {"XLK": 200.0}
        await executor.take_snapshot("A", date(2026, 2, 5), prices)

        snapshots = await db.get_snapshots("A")
        assert len(snapshots) == 1
        assert snapshots[0].total_value > 0

    async def test_snapshot_with_positions(self, db: Database, mock_client) -> None:
        executor = IBKRExecutor(db, mock_client)
        await executor.initialize_portfolios()
        await db.upsert_position("A", "XLK", 10, 200.0)
        await db.upsert_portfolio("A", cash=31_000.0, total_value=33_000.0)

        prices = {"XLK": 210.0}
        await executor.take_snapshot("A", date(2026, 2, 5), prices)

        snapshots = await db.get_snapshots("A")
        assert len(snapshots) == 1
        # 31000 cash + 10*210 = 33100
        assert snapshots[0].total_value == 33100.0


class TestIBKRExecutorFallback:
    async def test_paper_trader_fallback_interface(self) -> None:
        """IBKRExecutor has the same 3 methods as PaperTrader."""
        from src.execution.paper_trader import PaperTrader

        for method in ("initialize_portfolios", "execute_trades", "take_snapshot"):
            assert hasattr(IBKRExecutor, method)
            assert hasattr(PaperTrader, method)

"""Tests for the paper trading executor."""

from datetime import date

import pytest

from src.execution.paper_trader import PaperTrader
from src.storage.database import Database
from src.storage.models import OrderSide, PortfolioName, TradeSchema


@pytest.fixture
async def trader():
    """Create a paper trader with in-memory database."""
    db = Database(url="sqlite+aiosqlite:///:memory:")
    await db.init_db()
    pt = PaperTrader(db)
    await pt.initialize_portfolios()
    yield pt
    await db.close()


class TestInitialization:
    async def test_portfolios_created(self, trader: PaperTrader) -> None:
        portfolio_a = await trader._db.get_portfolio("A")
        assert portfolio_a is not None
        assert portfolio_a.cash == 33_000.0

        portfolio_b = await trader._db.get_portfolio("B")
        assert portfolio_b is not None
        assert portfolio_b.cash == 66_000.0

    async def test_idempotent_init(self, trader: PaperTrader) -> None:
        # Second init should not change anything
        await trader.initialize_portfolios()
        portfolio = await trader._db.get_portfolio("A")
        assert portfolio.cash == 33_000.0


class TestExecuteTrades:
    async def test_buy_deducts_cash(self, trader: PaperTrader) -> None:
        trade = TradeSchema(
            portfolio=PortfolioName.A,
            ticker="XLK",
            side=OrderSide.BUY,
            shares=100.0,
            price=200.0,
        )
        executed = await trader.execute_trades([trade])
        assert len(executed) == 1

        portfolio = await trader._db.get_portfolio("A")
        assert portfolio.cash == 33_000.0 - 20_000.0

        positions = await trader._db.get_positions("A")
        assert len(positions) == 1
        assert positions[0].ticker == "XLK"
        assert positions[0].shares == 100.0

    async def test_sell_adds_cash(self, trader: PaperTrader) -> None:
        # First buy
        buy = TradeSchema(
            portfolio=PortfolioName.A, ticker="XLK", side=OrderSide.BUY,
            shares=100.0, price=200.0,
        )
        await trader.execute_trades([buy])

        # Then sell
        sell = TradeSchema(
            portfolio=PortfolioName.A, ticker="XLK", side=OrderSide.SELL,
            shares=50.0, price=210.0,
        )
        executed = await trader.execute_trades([sell])
        assert len(executed) == 1

        portfolio = await trader._db.get_portfolio("A")
        expected_cash = 33_000.0 - 20_000.0 + 10_500.0
        assert portfolio.cash == expected_cash

        positions = await trader._db.get_positions("A")
        assert positions[0].shares == 50.0

    async def test_insufficient_cash_rejected(self, trader: PaperTrader) -> None:
        trade = TradeSchema(
            portfolio=PortfolioName.A, ticker="XLK", side=OrderSide.BUY,
            shares=1000.0, price=200.0,  # $200K > $33K cash
        )
        executed = await trader.execute_trades([trade])
        assert len(executed) == 0

    async def test_insufficient_shares_rejected(self, trader: PaperTrader) -> None:
        trade = TradeSchema(
            portfolio=PortfolioName.A, ticker="XLK", side=OrderSide.SELL,
            shares=100.0, price=200.0,
        )
        executed = await trader.execute_trades([trade])
        assert len(executed) == 0

    async def test_sells_before_buys(self, trader: PaperTrader) -> None:
        # Buy first to have a position
        buy = TradeSchema(
            portfolio=PortfolioName.A, ticker="XLF", side=OrderSide.BUY,
            shares=100.0, price=40.0,
        )
        await trader.execute_trades([buy])

        # Now sell XLF and buy XLK in same batch
        trades = [
            TradeSchema(
                portfolio=PortfolioName.A, ticker="XLK", side=OrderSide.BUY,
                shares=50.0, price=200.0,
            ),
            TradeSchema(
                portfolio=PortfolioName.A, ticker="XLF", side=OrderSide.SELL,
                shares=100.0, price=42.0,
            ),
        ]
        executed = await trader.execute_trades(trades)
        # Both should succeed (sell first frees cash)
        assert len(executed) == 2

    async def test_trade_logged(self, trader: PaperTrader) -> None:
        trade = TradeSchema(
            portfolio=PortfolioName.A, ticker="XLK", side=OrderSide.BUY,
            shares=10.0, price=200.0, reason="test trade",
        )
        await trader.execute_trades([trade])
        trades = await trader._db.get_trades("A")
        assert len(trades) == 1
        assert trades[0].reason == "test trade"


class TestSnapshot:
    async def test_take_snapshot(self, trader: PaperTrader) -> None:
        buy = TradeSchema(
            portfolio=PortfolioName.A, ticker="XLK", side=OrderSide.BUY,
            shares=100.0, price=200.0,
        )
        await trader.execute_trades([buy])

        await trader.take_snapshot("A", date(2026, 2, 5), {"XLK": 205.0})

        snapshots = await trader._db.get_snapshots("A")
        assert len(snapshots) == 1
        # cash = 33000 - 20000 = 13000, positions = 100*205 = 20500
        assert snapshots[0].total_value == 13_000.0 + 20_500.0

    async def test_daily_return_calculation(self, trader: PaperTrader) -> None:
        buy = TradeSchema(
            portfolio=PortfolioName.A, ticker="XLK", side=OrderSide.BUY,
            shares=100.0, price=200.0,
        )
        await trader.execute_trades([buy])

        # Day 1
        await trader.take_snapshot("A", date(2026, 2, 4), {"XLK": 200.0})
        # Day 2 — XLK goes up
        await trader.take_snapshot("A", date(2026, 2, 5), {"XLK": 210.0})

        snapshots = await trader._db.get_snapshots("A")
        assert len(snapshots) == 2
        assert snapshots[1].daily_return_pct is not None
        assert snapshots[1].daily_return_pct > 0  # price went up

"""Tests for PaperTrader tenant_id propagation."""

import pytest

from src.execution.paper_trader import PaperTrader
from src.storage.database import Database
from src.storage.models import OrderSide, PortfolioName, TradeSchema


@pytest.fixture
async def db():
    database = Database(url="sqlite+aiosqlite:///:memory:")
    await database.init_db()
    yield database
    await database.close()


@pytest.fixture
async def trader(db: Database) -> PaperTrader:
    return PaperTrader(db)


def _buy(ticker: str = "AAPL", shares: float = 10, price: float = 150.0) -> TradeSchema:
    return TradeSchema(
        portfolio=PortfolioName.B, ticker=ticker,
        side=OrderSide.BUY, shares=shares, price=price,
        reason="test buy",
    )


def _sell(ticker: str = "AAPL", shares: float = 10, price: float = 160.0) -> TradeSchema:
    return TradeSchema(
        portfolio=PortfolioName.B, ticker=ticker,
        side=OrderSide.SELL, shares=shares, price=price,
        reason="test sell",
    )


async def test_execute_trades_with_tenant_id(db: Database, trader: PaperTrader):
    """BUY trade updates correct tenant's portfolio and positions."""
    await trader.initialize_portfolios(tenant_id="tenant-1")
    executed = await trader.execute_trades([_buy()], tenant_id="tenant-1")
    assert len(executed) == 1

    # Verify tenant-1 has the position
    positions = await db.get_positions("B", tenant_id="tenant-1")
    assert len(positions) == 1
    assert positions[0].ticker == "AAPL"
    assert positions[0].shares == 10

    # Cash should be reduced
    portfolio = await db.get_portfolio("B", tenant_id="tenant-1")
    assert portfolio.cash == pytest.approx(66_000.0 - 1500.0)


async def test_sell_trade_with_tenant_id(db: Database, trader: PaperTrader):
    """SELL trade correctly updates tenant-scoped position and cash."""
    await trader.initialize_portfolios(tenant_id="tenant-1")
    await trader.execute_trades([_buy()], tenant_id="tenant-1")
    executed = await trader.execute_trades([_sell()], tenant_id="tenant-1")
    assert len(executed) == 1

    # Position should be gone (0 shares → deleted)
    positions = await db.get_positions("B", tenant_id="tenant-1")
    assert len(positions) == 0

    # Cash should be restored minus spread
    portfolio = await db.get_portfolio("B", tenant_id="tenant-1")
    expected_cash = 66_000.0 - (10 * 150) + (10 * 160)
    assert portfolio.cash == pytest.approx(expected_cash)


async def test_tenant_isolation(db: Database, trader: PaperTrader):
    """Trades for tenant-1 don't affect tenant-2."""
    await trader.initialize_portfolios(tenant_id="tenant-1")
    await trader.initialize_portfolios(tenant_id="tenant-2")

    # Buy in tenant-1
    await trader.execute_trades([_buy()], tenant_id="tenant-1")

    # tenant-1 should have a position
    pos1 = await db.get_positions("B", tenant_id="tenant-1")
    assert len(pos1) == 1

    # tenant-2 should have no positions
    pos2 = await db.get_positions("B", tenant_id="tenant-2")
    assert len(pos2) == 0

    # tenant-2 cash should be untouched
    p2 = await db.get_portfolio("B", tenant_id="tenant-2")
    assert p2.cash == pytest.approx(66_000.0)


async def test_insufficient_cash_tenant_scoped(db: Database, trader: PaperTrader):
    """Insufficient cash check uses tenant-scoped portfolio."""
    await trader.initialize_portfolios(tenant_id="tenant-1")

    big_buy = TradeSchema(
        portfolio=PortfolioName.B, ticker="AAPL",
        side=OrderSide.BUY, shares=1000, price=100.0,
        reason="too expensive",
    )
    executed = await trader.execute_trades([big_buy], tenant_id="tenant-1")
    assert len(executed) == 0


async def test_insufficient_shares_tenant_scoped(db: Database, trader: PaperTrader):
    """Insufficient shares check uses tenant-scoped positions."""
    await trader.initialize_portfolios(tenant_id="tenant-1")
    await trader.initialize_portfolios(tenant_id="tenant-2")

    # Buy in tenant-1
    await trader.execute_trades([_buy()], tenant_id="tenant-1")

    # Try to sell from tenant-2 (no position there)
    executed = await trader.execute_trades([_sell()], tenant_id="tenant-2")
    assert len(executed) == 0


async def test_trade_logged_with_tenant_id(db: Database, trader: PaperTrader):
    """Trade log entry uses tenant_id."""
    await trader.initialize_portfolios(tenant_id="tenant-1")
    await trader.execute_trades([_buy()], tenant_id="tenant-1")

    trades1 = await db.get_trades("B", tenant_id="tenant-1")
    assert len(trades1) == 1
    assert trades1[0].ticker == "AAPL"

    # No trades logged for default tenant
    trades_default = await db.get_trades("B", tenant_id="default")
    assert len(trades_default) == 0


async def test_backward_compat_default_tenant(db: Database, trader: PaperTrader):
    """Default tenant_id still works when not specified."""
    await trader.initialize_portfolios()
    executed = await trader.execute_trades([_buy()])
    assert len(executed) == 1

    positions = await db.get_positions("B", tenant_id="default")
    assert len(positions) == 1


async def test_add_to_existing_position_tenant_scoped(db: Database, trader: PaperTrader):
    """Adding to an existing position works with tenant scoping."""
    await trader.initialize_portfolios(tenant_id="tenant-1")
    await trader.execute_trades([_buy(shares=5, price=100.0)], tenant_id="tenant-1")
    await trader.execute_trades([_buy(shares=5, price=200.0)], tenant_id="tenant-1")

    positions = await db.get_positions("B", tenant_id="tenant-1")
    assert len(positions) == 1
    assert positions[0].shares == 10
    # Weighted avg: (5*100 + 5*200) / 10 = 150
    assert positions[0].avg_price == pytest.approx(150.0)

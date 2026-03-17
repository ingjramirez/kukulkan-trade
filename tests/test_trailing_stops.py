"""Tests for trailing stops: CRUD, trigger logic, conviction matrix, orchestrator integration."""

import pytest

from config.risk_rules import TRAIL_PCT
from src.orchestrator import _get_trail_pct
from src.storage.database import Database
from src.storage.models import OrderSide, PortfolioName, TradeSchema


@pytest.fixture
async def db():
    database = Database(url="sqlite+aiosqlite:///:memory:")
    await database.init_db()
    await database.ensure_tenant("t1")
    await database.ensure_tenant("t2")
    yield database
    await database.close()


# ── CRUD ──────────────────────────────────────────────────────────────


async def test_create_trailing_stop(db: Database):
    """create_trailing_stop sets peak=entry and calculates stop_price."""
    stop = await db.create_trailing_stop(
        tenant_id="t1",
        portfolio="B",
        ticker="AAPL",
        entry_price=150.0,
        trail_pct=0.05,
    )
    assert stop.entry_price == 150.0
    assert stop.peak_price == 150.0
    assert stop.stop_price == pytest.approx(142.5)  # 150 * 0.95
    assert stop.is_active is True
    assert stop.trail_pct == 0.05


async def test_get_active_trailing_stops(db: Database):
    """get_active_trailing_stops returns only active stops."""
    await db.create_trailing_stop("t1", "B", "AAPL", 150.0, 0.05)
    await db.create_trailing_stop("t1", "B", "MSFT", 300.0, 0.07)
    stop3 = await db.create_trailing_stop("t1", "A", "XLK", 100.0, 0.10)
    await db.deactivate_trailing_stop(stop3.id)

    active = await db.get_active_trailing_stops("t1")
    assert len(active) == 2
    tickers = {s.ticker for s in active}
    assert tickers == {"AAPL", "MSFT"}


async def test_get_active_trailing_stops_portfolio_filter(db: Database):
    """Portfolio filter returns only stops for that portfolio."""
    await db.create_trailing_stop("t1", "A", "XLK", 100.0, 0.10)
    await db.create_trailing_stop("t1", "B", "AAPL", 150.0, 0.05)

    a_stops = await db.get_active_trailing_stops("t1", "A")
    assert len(a_stops) == 1
    assert a_stops[0].ticker == "XLK"


async def test_update_trailing_stop_peak(db: Database):
    """Updating peak_price also updates stop_price."""
    stop = await db.create_trailing_stop("t1", "B", "AAPL", 150.0, 0.05)
    await db.update_trailing_stop(
        stop.id,
        peak_price=180.0,
        stop_price=180.0 * 0.95,
    )
    active = await db.get_active_trailing_stops("t1", "B")
    assert len(active) == 1
    assert active[0].peak_price == 180.0
    assert active[0].stop_price == pytest.approx(171.0)


async def test_deactivate_trailing_stop(db: Database):
    """Deactivating a stop removes it from active list."""
    stop = await db.create_trailing_stop("t1", "B", "AAPL", 150.0, 0.05)
    await db.deactivate_trailing_stop(stop.id)
    active = await db.get_active_trailing_stops("t1")
    assert len(active) == 0


async def test_deactivate_trailing_stops_for_ticker(db: Database):
    """Deactivate all stops for a specific tenant/portfolio/ticker."""
    await db.create_trailing_stop("t1", "B", "AAPL", 150.0, 0.05)
    await db.create_trailing_stop("t1", "B", "MSFT", 300.0, 0.07)
    await db.deactivate_trailing_stops_for_ticker("t1", "B", "AAPL")

    active = await db.get_active_trailing_stops("t1", "B")
    assert len(active) == 1
    assert active[0].ticker == "MSFT"


async def test_tenant_isolation(db: Database):
    """Stops for tenant-1 don't appear for tenant-2."""
    await db.create_trailing_stop("t1", "B", "AAPL", 150.0, 0.05)
    await db.create_trailing_stop("t2", "B", "MSFT", 300.0, 0.07)

    t1_stops = await db.get_active_trailing_stops("t1")
    t2_stops = await db.get_active_trailing_stops("t2")
    assert len(t1_stops) == 1
    assert t1_stops[0].ticker == "AAPL"
    assert len(t2_stops) == 1
    assert t2_stops[0].ticker == "MSFT"


async def test_create_replaces_existing_active_stop(db: Database):
    """Creating a stop for the same ticker replaces the old one."""
    await db.create_trailing_stop("t1", "B", "AAPL", 150.0, 0.05)
    await db.create_trailing_stop("t1", "B", "AAPL", 160.0, 0.07)

    active = await db.get_active_trailing_stops("t1", "B")
    assert len(active) == 1
    assert active[0].entry_price == 160.0
    assert active[0].trail_pct == 0.07


# ── Conviction Matrix ────────────────────────────────────────────────


def test_trail_pct_conservative_high():
    assert TRAIL_PCT["conservative"]["high"] == 0.05


def test_trail_pct_aggressive_low():
    # Tightened from 0.15 → 0.12 (meta-agent 2026-03-17: reduce avg loss at low WR)
    assert TRAIL_PCT["aggressive"]["low"] == 0.12


def test_get_trail_pct_high_conviction():
    trade = TradeSchema(
        portfolio=PortfolioName.B,
        ticker="AAPL",
        side=OrderSide.BUY,
        shares=10,
        price=150.0,
        reason="high conviction tech play",
    )
    pct = _get_trail_pct("conservative", trade)
    assert pct == 0.05


def test_get_trail_pct_low_conviction():
    trade = TradeSchema(
        portfolio=PortfolioName.B,
        ticker="AAPL",
        side=OrderSide.BUY,
        shares=10,
        price=150.0,
        reason="low conviction speculative",
    )
    pct = _get_trail_pct("aggressive", trade)
    assert pct == 0.12  # tightened from 0.15 → 0.12 (meta-agent 2026-03-17)


def test_get_trail_pct_default_medium():
    trade = TradeSchema(
        portfolio=PortfolioName.B,
        ticker="AAPL",
        side=OrderSide.BUY,
        shares=10,
        price=150.0,
        reason="some reason",
    )
    pct = _get_trail_pct("standard", trade)
    assert pct == 0.10  # standard + medium


def test_get_trail_pct_unknown_strategy_falls_back():
    trade = TradeSchema(
        portfolio=PortfolioName.B,
        ticker="AAPL",
        side=OrderSide.BUY,
        shares=10,
        price=150.0,
        reason="test",
    )
    pct = _get_trail_pct("unknown_strategy", trade)
    assert pct == 0.07  # conservative medium fallback

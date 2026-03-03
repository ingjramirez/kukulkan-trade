"""Tests for direct trade execution from chat (execute_trade + set_trailing_stop with executor)."""

import pytest

from src.agent.tools.actions import ActionState, _execute_trade, _set_trailing_stop
from src.analysis.risk_manager import RiskManager
from src.execution.paper_trader import PaperTrader
from src.storage.database import Database


@pytest.fixture
async def db():
    database = Database(url="sqlite+aiosqlite:///:memory:")
    await database.init_db()
    yield database
    await database.close()


@pytest.fixture
async def setup(db: Database):
    """Create portfolio B with cash and a position for testing."""
    await db.upsert_portfolio("B", cash=50000.0, total_value=50000.0)
    return db


@pytest.fixture
def executor(db: Database):
    return PaperTrader(db)


@pytest.fixture
def risk_manager():
    return RiskManager()


@pytest.fixture
def state():
    return ActionState()


PRICES = {"NVDA": 120.0, "XLK": 200.0, "GLD": 460.0}


# ── Direct BUY execution ─────────────────────────────────────────────────────


async def test_direct_buy_fills(state, setup, executor, risk_manager):
    """BUY via direct execution creates position and deducts cash."""
    db = setup
    result = await _execute_trade(
        state, executor, risk_manager, db, "default", PRICES, "NVDA", "BUY", 10, reason="Test buy"
    )
    assert result["status"] == "filled"
    assert result["ticker"] == "NVDA"
    assert result["shares"] == 10
    assert result["price"] == 120.0
    assert result["total"] == 1200.0

    # Verify position created
    positions = await db.get_positions("B", tenant_id="default")
    nvda = next((p for p in positions if p.ticker == "NVDA"), None)
    assert nvda is not None
    assert nvda.shares == 10

    # Verify cash deducted
    portfolio = await db.get_portfolio("B", tenant_id="default")
    assert portfolio.cash == pytest.approx(50000.0 - 1200.0)

    # Verify recorded in state
    assert len(state.executed_trades) == 1
    assert state.executed_trades[0]["status"] == "filled"


async def test_direct_buy_insufficient_cash(state, setup, executor, risk_manager):
    """BUY blocked or rejected when exceeding limits."""
    db = setup
    result = await _execute_trade(
        state, executor, risk_manager, db, "default", PRICES, "GLD", "BUY", 200, reason="Too expensive"
    )
    # 200 * 460 = 92,000 > 50,000 cash → risk manager blocks at concentration limit
    assert result["status"] in ("blocked", "rejected")


async def test_direct_buy_no_price(state, setup, executor, risk_manager):
    """BUY fails gracefully when price unavailable and yfinance can't help."""
    db = setup
    result = await _execute_trade(
        state, executor, risk_manager, db, "default", {}, "ZZZZZ", "BUY", 10, reason="Unknown ticker"
    )
    # No price in dict, yfinance will likely fail for ZZZZZ
    assert result.get("error") or result.get("status") in ("error", "rejected")


# ── Direct SELL execution ────────────────────────────────────────────────────


async def test_direct_sell_fills(state, setup, executor, risk_manager):
    """SELL via direct execution reduces position."""
    db = setup
    # First buy to create position
    await _execute_trade(state, executor, risk_manager, db, "default", PRICES, "NVDA", "BUY", 20, reason="Setup")
    state.executed_trades.clear()

    # Now sell half
    result = await _execute_trade(
        state, executor, risk_manager, db, "default", PRICES, "NVDA", "SELL", 10, reason="Take profit"
    )
    assert result["status"] == "filled"
    assert result["side"] == "SELL"
    assert result["shares"] == 10

    positions = await db.get_positions("B", tenant_id="default")
    nvda = next((p for p in positions if p.ticker == "NVDA"), None)
    assert nvda is not None
    assert nvda.shares == 10


async def test_direct_sell_no_position(state, setup, executor, risk_manager):
    """SELL rejected when no position exists."""
    db = setup
    result = await _execute_trade(
        state, executor, risk_manager, db, "default", PRICES, "XLK", "SELL", 10, reason="No position"
    )
    assert result["status"] == "rejected"


# ── Risk check blocking ──────────────────────────────────────────────────────


async def test_risk_blocks_oversized_buy(state, setup, executor, risk_manager):
    """Risk manager blocks BUY that exceeds concentration limit (50%)."""
    db = setup
    await db.upsert_portfolio("B", cash=100000.0, total_value=100000.0, tenant_id="default")
    prices = {"NVDA": 100.0}

    # 600 shares * $100 = $60,000 = 60% of $100k portfolio → blocked at 50% limit
    result = await _execute_trade(
        state, executor, risk_manager, db, "default", prices, "NVDA", "BUY", 600, reason="Too big"
    )
    assert result["status"] == "blocked"


# ── Trailing stop direct creation ────────────────────────────────────────────


async def test_direct_trailing_stop_creates(state, setup):
    """Trailing stop creates DB row when db provided."""
    db = setup
    # Create a position first
    from src.storage.models import PositionRow

    async with db.session() as s:
        s.add(PositionRow(portfolio="B", ticker="NVDA", shares=50, avg_price=110.0, tenant_id="default"))
        await s.commit()

    result = await _set_trailing_stop(state, db, "default", PRICES, "NVDA", 0.07, reason="7% stop")
    assert result["status"] == "created"
    assert result["trail_pct"] == 0.07
    assert result["stop_price"] == round(120.0 * 0.93, 2)  # 120 * (1 - 0.07) = 111.60

    # Verify DB row
    stops = await db.get_active_trailing_stops("default", portfolio="B")
    nvda_stop = next((s for s in stops if s.ticker == "NVDA"), None)
    assert nvda_stop is not None
    assert nvda_stop.trail_pct == 0.07

    # Verify recorded in state
    assert len(state.trailing_stop_requests) == 1


async def test_direct_trailing_stop_no_position(state, setup):
    """Trailing stop fails when no position exists."""
    db = setup
    result = await _set_trailing_stop(state, db, "default", PRICES, "XLK", 0.05)
    assert "error" in result
    assert "No position" in result["error"]


async def test_direct_trailing_stop_replaces_existing(state, setup):
    """Creating a trailing stop for same ticker replaces the old one."""
    db = setup
    from src.storage.models import PositionRow

    async with db.session() as s:
        s.add(PositionRow(portfolio="B", ticker="NVDA", shares=50, avg_price=110.0, tenant_id="default"))
        await s.commit()

    await _set_trailing_stop(state, db, "default", PRICES, "NVDA", 0.05)
    await _set_trailing_stop(state, db, "default", PRICES, "NVDA", 0.10)

    stops = await db.get_active_trailing_stops("default", portfolio="B")
    nvda_stops = [s for s in stops if s.ticker == "NVDA"]
    assert len(nvda_stops) == 1
    assert nvda_stops[0].trail_pct == 0.10


# ── Fallback accumulation (no executor) ──────────────────────────────────────


async def test_fallback_accumulates_without_executor(state):
    """Without executor, _execute_trade accumulates like before."""
    result = await _execute_trade(state, None, None, None, "default", {}, "NVDA", "BUY", 50)
    assert result["status"] == "submitted"
    assert len(state.executed_trades) == 1
    assert len(state.proposed_trades) == 1


async def test_fallback_stop_accumulates_without_db(state):
    """Without db, _set_trailing_stop accumulates like before."""
    result = await _set_trailing_stop(state, None, "default", {}, "NVDA", 0.07)
    assert result["status"] == "ok"
    assert len(state.trailing_stop_requests) == 1


# ── Input validation (shared path) ──────────────────────────────────────────


async def test_direct_buy_validates_inputs(state, setup, executor, risk_manager):
    """Validation errors returned before execution attempt."""
    db = setup
    # Empty ticker
    r = await _execute_trade(state, executor, risk_manager, db, "default", PRICES, "", "BUY", 10)
    assert "error" in r

    # Invalid side
    r = await _execute_trade(state, executor, risk_manager, db, "default", PRICES, "NVDA", "SHORT", 10)
    assert "error" in r

    # Zero shares
    r = await _execute_trade(state, executor, risk_manager, db, "default", PRICES, "NVDA", "BUY", 0)
    assert "error" in r

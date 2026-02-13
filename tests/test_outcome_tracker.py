"""Tests for OutcomeTracker — trade outcome computation with benchmarks."""

import json
from datetime import date, datetime, timedelta, timezone
from unittest.mock import AsyncMock, patch

import pytest

from src.analysis.outcome_tracker import OutcomeTracker
from src.storage.database import Database
from src.storage.models import (
    AgentDecisionRow,
    TradeRow,
)


@pytest.fixture
async def db():
    database = Database(url="sqlite+aiosqlite:///:memory:")
    await database.init_db()
    yield database
    await database.close()


@pytest.fixture
def mock_yf_prices():
    """Patch yfinance to return controlled prices."""
    with patch("src.analysis.outcome_tracker.yf") as mock_yf:
        yield mock_yf


async def _seed_buy_trade(db: Database, ticker: str = "XLK", price: float = 100.0, days_ago: int = 10):
    """Helper to seed a BUY trade."""
    executed_at = datetime.now(timezone.utc) - timedelta(days=days_ago)
    async with db.session() as s:
        s.add(
            TradeRow(
                tenant_id="default",
                portfolio="B",
                ticker=ticker,
                side="BUY",
                shares=10,
                price=price,
                total=price * 10,
                reason="test buy",
                executed_at=executed_at,
            )
        )
        await s.commit()


async def _seed_position(db: Database, ticker: str = "XLK", shares: float = 10, avg_price: float = 100.0):
    """Helper to seed an open position."""
    await db.upsert_position("B", ticker, shares, avg_price)


async def _seed_decision(db: Database, decision_date: date, ticker: str = "XLK", conviction: str = "high"):
    """Helper to seed an agent decision."""
    trades_json = json.dumps(
        [{"ticker": ticker, "side": "BUY", "shares": 10, "price": 100, "reason": f"{conviction} conviction buy"}]
    )
    async with db.session() as s:
        s.add(
            AgentDecisionRow(
                tenant_id="default",
                date=decision_date,
                prompt_summary="test",
                response_summary="test",
                proposed_trades=trades_json,
                reasoning="test reasoning",
                model_used="test-model",
                tokens_used=100,
            )
        )
        await s.commit()


@pytest.mark.asyncio
async def test_empty_db_returns_no_outcomes(db):
    tracker = OutcomeTracker(db)
    outcomes = await tracker.get_recent_outcomes(days=30, tenant_id="default")
    assert outcomes == []


@pytest.mark.asyncio
async def test_buy_trade_with_open_position(db, mock_yf_prices):
    """A BUY trade with an open position uses current price for P&L."""
    await _seed_buy_trade(db, "XLK", price=100.0, days_ago=5)
    await _seed_position(db, "XLK", shares=10, avg_price=100.0)
    trade_date = date.today() - timedelta(days=5)
    await _seed_decision(db, trade_date, "XLK", "high")

    # Mock yfinance: current XLK=110, SPY=450
    mock_yf_prices.download = lambda *a, **kw: _mock_download_data({"XLK": 110.0, "SPY": 450.0})

    tracker = OutcomeTracker(db)
    # Patch both price methods
    tracker._fetch_current_prices = AsyncMock(return_value={"XLK": 110.0, "SPY": 450.0})
    tracker._fetch_prices_at_dates = AsyncMock(
        return_value={
            ("SPY", trade_date): 440.0,
            ("XLK", trade_date): 100.0,
        }
    )

    outcomes = await tracker.get_recent_outcomes(days=30, tenant_id="default")
    assert len(outcomes) == 1
    assert outcomes[0].ticker == "XLK"
    assert outcomes[0].pnl_pct == 10.0  # (110-100)/100 * 100
    assert outcomes[0].exit_price is None  # Still open


@pytest.mark.asyncio
async def test_pnl_calculation(db):
    """Verify P&L math with known entry and current prices."""
    await _seed_buy_trade(db, "AAPL", price=150.0, days_ago=7)
    await _seed_position(db, "AAPL", shares=10, avg_price=150.0)
    trade_date = date.today() - timedelta(days=7)
    await _seed_decision(db, trade_date, "AAPL", "medium")

    tracker = OutcomeTracker(db)
    tracker._fetch_current_prices = AsyncMock(return_value={"AAPL": 165.0, "SPY": 450.0, "XLK": 200.0})
    tracker._fetch_prices_at_dates = AsyncMock(
        return_value={
            ("SPY", trade_date): 440.0,
            ("XLK", trade_date): 190.0,
        }
    )

    outcomes = await tracker.get_recent_outcomes(days=30, tenant_id="default")
    assert len(outcomes) == 1
    assert outcomes[0].pnl_pct == 10.0  # (165-150)/150 * 100


@pytest.mark.asyncio
async def test_sector_lookup(db):
    """Verify sector is resolved from SECTOR_MAP."""
    await _seed_buy_trade(db, "AAPL", price=150.0, days_ago=3)
    await _seed_position(db, "AAPL", shares=5, avg_price=150.0)
    tracker = OutcomeTracker(db)
    tracker._fetch_current_prices = AsyncMock(return_value={"AAPL": 160.0, "SPY": 450.0, "XLK": 200.0})
    tracker._fetch_prices_at_dates = AsyncMock(return_value={})

    outcomes = await tracker.get_recent_outcomes(days=30, tenant_id="default")
    assert len(outcomes) == 1
    assert outcomes[0].sector == "Technology"


@pytest.mark.asyncio
async def test_alpha_calculation(db):
    """Verify alpha vs SPY calculation."""
    await _seed_buy_trade(db, "XLE", price=80.0, days_ago=5)
    await _seed_position(db, "XLE", shares=10, avg_price=80.0)
    trade_date = date.today() - timedelta(days=5)
    await _seed_decision(db, trade_date, "XLE")

    tracker = OutcomeTracker(db)
    # XLE: 80 → 88 = +10%, SPY: 440 → 448.8 = +2%, alpha = +8%
    tracker._fetch_current_prices = AsyncMock(return_value={"XLE": 88.0, "SPY": 448.8})
    tracker._fetch_prices_at_dates = AsyncMock(
        return_value={
            ("SPY", trade_date): 440.0,
            ("XLE", trade_date): 80.0,
        }
    )

    outcomes = await tracker.get_recent_outcomes(days=30, tenant_id="default")
    assert len(outcomes) == 1
    assert outcomes[0].alpha_vs_spy == 8.0


@pytest.mark.asyncio
async def test_conviction_extraction_from_decisions(db):
    """Verify conviction is extracted from proposed_trades in agent_decisions."""
    trade_date = date.today() - timedelta(days=3)
    await _seed_buy_trade(db, "NVDA", price=500.0, days_ago=3)
    await _seed_position(db, "NVDA", shares=5, avg_price=500.0)
    await _seed_decision(db, trade_date, "NVDA", "high")

    tracker = OutcomeTracker(db)
    tracker._fetch_current_prices = AsyncMock(return_value={"NVDA": 520.0, "SPY": 450.0, "XLK": 200.0})
    tracker._fetch_prices_at_dates = AsyncMock(return_value={})

    outcomes = await tracker.get_recent_outcomes(days=30, tenant_id="default")
    assert len(outcomes) == 1
    assert outcomes[0].conviction == "high"


@pytest.mark.asyncio
async def test_open_positions_filter(db):
    """get_open_position_outcomes only returns open positions."""
    # Trade 1: still open
    await _seed_buy_trade(db, "XLK", price=100.0, days_ago=5)
    await _seed_position(db, "XLK", shares=10, avg_price=100.0)

    # Trade 2: closed (no position)
    executed_at = datetime.now(timezone.utc) - timedelta(days=10)
    async with db.session() as s:
        s.add(
            TradeRow(
                tenant_id="default",
                portfolio="B",
                ticker="XLE",
                side="BUY",
                shares=5,
                price=80.0,
                total=400.0,
                executed_at=executed_at,
            )
        )
        await s.commit()
    # No position for XLE = closed
    sell_at = datetime.now(timezone.utc) - timedelta(days=3)
    async with db.session() as s:
        s.add(
            TradeRow(
                tenant_id="default",
                portfolio="B",
                ticker="XLE",
                side="SELL",
                shares=5,
                price=85.0,
                total=425.0,
                executed_at=sell_at,
            )
        )
        await s.commit()

    tracker = OutcomeTracker(db)
    tracker._fetch_current_prices = AsyncMock(return_value={"XLK": 110.0, "XLE": 85.0, "SPY": 450.0})
    tracker._fetch_prices_at_dates = AsyncMock(return_value={})

    open_outcomes = await tracker.get_open_position_outcomes(tenant_id="default")
    assert len(open_outcomes) == 1
    assert open_outcomes[0].ticker == "XLK"


@pytest.mark.asyncio
async def test_hold_days_calculation(db):
    """Verify hold_days is computed correctly."""
    days_ago = 15
    await _seed_buy_trade(db, "GLD", price=180.0, days_ago=days_ago)
    await _seed_position(db, "GLD", shares=10, avg_price=180.0)

    tracker = OutcomeTracker(db)
    tracker._fetch_current_prices = AsyncMock(return_value={"GLD": 185.0, "SPY": 450.0})
    tracker._fetch_prices_at_dates = AsyncMock(return_value={})

    outcomes = await tracker.get_recent_outcomes(days=30, tenant_id="default")
    assert len(outcomes) == 1
    assert outcomes[0].hold_days == days_ago


def _mock_download_data(prices: dict[str, float]):
    """Create a minimal DataFrame-like mock for yf.download."""
    import pandas as pd

    today = date.today()
    idx = pd.DatetimeIndex([today - timedelta(days=1), today])
    data = {}
    for ticker, price in prices.items():
        data[ticker] = [price - 1, price]
    df = pd.DataFrame(data, index=idx)
    return pd.DataFrame({"Close": df})

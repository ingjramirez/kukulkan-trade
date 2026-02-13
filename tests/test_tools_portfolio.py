"""Tests for portfolio investigation tools."""

import pytest

from src.agent.tools import ToolRegistry
from src.agent.tools.portfolio import (
    _get_current_positions,
    _get_portfolio_summary,
    _get_position_pnl,
    register_portfolio_tools,
)
from src.storage.database import Database


@pytest.fixture
async def db():
    database = Database(url="sqlite+aiosqlite:///:memory:")
    await database.init_db()
    yield database
    await database.close()


async def _seed_portfolio(db: Database, cash: float = 50000.0):
    await db.upsert_portfolio("B", cash=cash, total_value=cash)


async def _seed_positions(db: Database):
    await db.upsert_position("B", "XLK", shares=100, avg_price=200.0)
    await db.upsert_position("B", "XLE", shares=50, avg_price=80.0)


@pytest.mark.asyncio
async def test_empty_positions(db):
    result = await _get_current_positions(db, "default")
    assert result == []


@pytest.mark.asyncio
async def test_positions_with_data(db):
    await _seed_positions(db)
    result = await _get_current_positions(db, "default")
    assert len(result) == 2
    tickers = {p["ticker"] for p in result}
    assert "XLK" in tickers
    assert result[0]["sector"] in ("Technology", "Energy")


@pytest.mark.asyncio
async def test_position_pnl(db):
    await _seed_positions(db)
    prices = {"XLK": 220.0, "XLE": 85.0}
    result = await _get_position_pnl(db, "default", prices, "XLK")
    assert result["ticker"] == "XLK"
    assert result["pnl_pct"] == 10.0  # (220-200)/200 * 100
    assert result["current_price"] == 220.0


@pytest.mark.asyncio
async def test_position_pnl_not_found(db):
    result = await _get_position_pnl(db, "default", {}, "NONEXIST")
    assert "error" in result


@pytest.mark.asyncio
async def test_portfolio_summary(db):
    await _seed_portfolio(db, cash=20000.0)
    await _seed_positions(db)
    prices = {"XLK": 200.0, "XLE": 80.0}
    result = await _get_portfolio_summary(db, "default", prices)
    assert result["cash"] == 20000.0
    assert result["position_count"] == 2
    assert "Technology" in result["sector_exposure"]


@pytest.mark.asyncio
async def test_registration(db):
    registry = ToolRegistry()
    register_portfolio_tools(registry, db, "default", {"XLK": 200.0})
    names = registry.tool_names
    assert "get_current_positions" in names
    assert "get_position_pnl" in names
    assert "get_portfolio_summary" in names


@pytest.mark.asyncio
async def test_partial_binding(db):
    """Verify tools work through the registry after partial binding."""
    await _seed_portfolio(db, cash=10000.0)
    await _seed_positions(db)
    registry = ToolRegistry()
    register_portfolio_tools(registry, db, "default", {"XLK": 210.0, "XLE": 82.0})
    result = await registry.execute("get_current_positions", {})
    assert len(result) == 2

"""Tests for Phase 2 market tools — upgraded + new tools.

Tests cover: get_batch_technicals, get_sector_heatmap, get_market_overview,
get_earnings_calendar, and legacy aliases.
"""

import numpy as np
import pandas as pd
import pytest

from config.universe import SECTOR_ETF_MAP
from src.agent.tools import ToolRegistry
from src.agent.tools.market import (
    _get_batch_technicals,
    _get_earnings_calendar,
    _get_market_overview,
    _get_price_and_technicals,
    _get_sector_heatmap,
    register_market_tools,
)
from src.storage.database import Database


@pytest.fixture
async def db():
    database = Database(url="sqlite+aiosqlite:///:memory:")
    await database.init_db()
    yield database
    await database.close()


@pytest.fixture
def closes():
    """Synthetic close price DataFrame with sector ETFs + regular tickers."""
    dates = pd.date_range("2025-01-01", periods=100, freq="B")
    np.random.seed(42)
    data = {
        "SPY": 450 + np.cumsum(np.random.normal(0, 1.5, 100)),
        "XLK": 200 + np.cumsum(np.random.normal(0, 1, 100)),
        "XLE": 80 + np.cumsum(np.random.normal(0, 0.5, 100)),
        "XLF": 35 + np.cumsum(np.random.normal(0, 0.3, 100)),
        "XLV": 130 + np.cumsum(np.random.normal(0, 0.4, 100)),
        "NVDA": 120 + np.cumsum(np.random.normal(0, 2, 100)),
        "GLD": 180 + np.cumsum(np.random.normal(0, 0.3, 100)),
    }
    return pd.DataFrame(data, index=dates)


# ── get_batch_technicals ─────────────────────────────────────────────────────


async def test_batch_technicals_multiple_tickers(closes: pd.DataFrame):
    """get_batch_technicals returns data for multiple tickers."""
    result = await _get_batch_technicals(closes, ["XLK", "NVDA", "GLD"])
    assert result["tickers_requested"] == 3
    assert len(result["results"]) == 3
    for entry in result["results"]:
        assert "ticker" in entry
        assert "price" in entry
        assert "change_1d_pct" in entry


async def test_batch_technicals_includes_rsi_for_long_series(closes: pd.DataFrame):
    """get_batch_technicals includes RSI/MACD/SMA when data is sufficient."""
    result = await _get_batch_technicals(closes, ["XLK"])
    entry = result["results"][0]
    assert entry["ticker"] == "XLK"
    # With 100 data points, technicals should be computed
    assert "rsi_14" in entry
    assert "macd" in entry
    assert "sma_20" in entry
    assert "sma_50" in entry


async def test_batch_technicals_missing_ticker(closes: pd.DataFrame):
    """get_batch_technicals returns error for unknown tickers."""
    result = await _get_batch_technicals(closes, ["NONEXIST", "XLK"])
    assert len(result["results"]) == 2
    assert "error" in result["results"][0]
    assert result["results"][1]["ticker"] == "XLK"


async def test_batch_technicals_empty_tickers(closes: pd.DataFrame):
    """get_batch_technicals returns error for empty ticker list."""
    result = await _get_batch_technicals(closes, [])
    assert "error" in result


async def test_batch_technicals_caps_at_20(closes: pd.DataFrame):
    """get_batch_technicals caps at 20 tickers per call."""
    many_tickers = [f"T{i}" for i in range(25)]
    result = await _get_batch_technicals(closes, many_tickers)
    assert result["tickers_requested"] == 20  # Capped


async def test_batch_technicals_short_data():
    """get_batch_technicals skips technicals for short series."""
    dates = pd.date_range("2025-01-01", periods=10, freq="B")
    short_closes = pd.DataFrame({"XLK": range(200, 210)}, index=dates)
    result = await _get_batch_technicals(short_closes, ["XLK"])
    entry = result["results"][0]
    assert "price" in entry
    assert "rsi_14" not in entry  # Too short for technicals


# ── get_sector_heatmap ───────────────────────────────────────────────────────


async def test_sector_heatmap_returns_sectors(closes: pd.DataFrame):
    """get_sector_heatmap returns sector data for available ETFs."""
    result = await _get_sector_heatmap(closes)
    assert result["sector_count"] > 0
    assert "sectors" in result

    # Check that available sectors from our fixture are present
    for sector, etf in SECTOR_ETF_MAP.items():
        if etf in closes.columns:
            assert sector in result["sectors"]
            entry = result["sectors"][sector]
            assert "etf" in entry
            assert "price" in entry
            assert "change_1d_pct" in entry


async def test_sector_heatmap_includes_rsi(closes: pd.DataFrame):
    """get_sector_heatmap includes RSI for sector ETFs with sufficient data."""
    result = await _get_sector_heatmap(closes)
    # XLK has 100 data points, should have RSI
    if "Technology" in result["sectors"]:
        tech = result["sectors"]["Technology"]
        assert "rsi_14" in tech


async def test_sector_heatmap_sorted_by_5d(closes: pd.DataFrame):
    """get_sector_heatmap sorts sectors by 5-day return (best first)."""
    result = await _get_sector_heatmap(closes)
    sectors = list(result["sectors"].values())
    five_d_returns = [s.get("change_5d_pct", 0) for s in sectors]
    assert five_d_returns == sorted(five_d_returns, reverse=True)


# ── get_market_overview ──────────────────────────────────────────────────────


async def test_market_overview_basic(closes: pd.DataFrame):
    """get_market_overview returns regime, VIX, SPY stats."""
    result = await _get_market_overview(closes, vix=18.5, yield_curve=0.42, regime="BULL")
    assert result["regime"] == "BULL"
    assert result["vix"] == 18.5
    assert result["yield_curve_10y_2y"] == 0.42
    assert "spy_price" in result
    assert "spy_1d_pct" in result


async def test_market_overview_includes_heatmap(closes: pd.DataFrame):
    """get_market_overview includes sector 1-week heatmap."""
    result = await _get_market_overview(closes, vix=18.5, yield_curve=None, regime=None)
    assert "sector_heatmap_1w" in result


async def test_market_overview_none_values(closes: pd.DataFrame):
    """get_market_overview handles None vix/yield_curve gracefully."""
    result = await _get_market_overview(closes, vix=None, yield_curve=None, regime=None)
    assert result["vix"] is None
    assert result["yield_curve_10y_2y"] is None
    assert result["regime"] == "Unknown"


# ── get_earnings_calendar ────────────────────────────────────────────────────


async def test_earnings_calendar_empty(db: Database):
    """get_earnings_calendar returns empty when no earnings exist."""
    result = await _get_earnings_calendar(db, ["XLK", "NVDA"])
    assert result["earnings_count"] == 0
    assert result["earnings"] == []


async def test_earnings_calendar_no_tickers(db: Database):
    """get_earnings_calendar returns message when no tickers provided."""
    result = await _get_earnings_calendar(db, [], tickers=None)
    assert "message" in result


async def test_earnings_calendar_days_clamping(db: Database):
    """get_earnings_calendar clamps days_ahead to 1-30."""
    result = await _get_earnings_calendar(db, ["XLK"], days_ahead=100)
    assert result["days_ahead"] == 30

    result = await _get_earnings_calendar(db, ["XLK"], days_ahead=-5)
    assert result["days_ahead"] == 1


# ── Registration and aliases ─────────────────────────────────────────────────


async def test_registration_all_tools(closes: pd.DataFrame, db: Database):
    """register_market_tools registers all Phase 2 + legacy tools."""
    registry = ToolRegistry()
    register_market_tools(
        registry,
        closes,
        vix=18.5,
        yield_curve=0.4,
        regime="BULL",
        db=db,
        held_tickers=["XLK"],
    )

    names = registry.tool_names
    # Phase 2 tools
    assert "get_batch_technicals" in names
    assert "get_sector_heatmap" in names
    assert "get_market_overview" in names
    assert "get_earnings_calendar" in names
    # Phase 32 aliases
    assert "get_price_and_technicals" in names
    assert "get_market_context" in names


async def test_registration_without_db(closes: pd.DataFrame):
    """register_market_tools without db skips earnings calendar."""
    registry = ToolRegistry()
    register_market_tools(registry, closes, vix=18.5)

    names = registry.tool_names
    assert "get_batch_technicals" in names
    assert "get_earnings_calendar" not in names


async def test_alias_get_price_and_technicals(closes: pd.DataFrame):
    """Legacy get_price_and_technicals returns single-ticker result."""
    result = await _get_price_and_technicals(closes, "XLK")
    assert result["ticker"] == "XLK"
    assert "price" in result
    assert "change_1d_pct" in result

"""Tests for market data investigation tools."""

import numpy as np
import pandas as pd
import pytest

from src.agent.tools import ToolRegistry
from src.agent.tools.market import (
    _get_market_context,
    _get_price_and_technicals,
    register_market_tools,
)


@pytest.fixture
def closes():
    """Create a realistic closes DataFrame with enough data for technicals."""
    dates = pd.date_range("2025-01-01", periods=100, freq="B")
    np.random.seed(42)
    data = {
        "SPY": 450 + np.cumsum(np.random.randn(100) * 2),
        "XLK": 200 + np.cumsum(np.random.randn(100)),
        "XLE": 80 + np.cumsum(np.random.randn(100) * 0.5),
        "XLF": 40 + np.cumsum(np.random.randn(100) * 0.3),
    }
    return pd.DataFrame(data, index=dates)


@pytest.mark.asyncio
async def test_price_and_technicals(closes):
    result = await _get_price_and_technicals(closes, "XLK")
    assert result["ticker"] == "XLK"
    assert "price" in result
    assert "change_1d_pct" in result
    assert "rsi_14" in result
    assert "macd" in result


@pytest.mark.asyncio
async def test_ticker_not_found(closes):
    result = await _get_price_and_technicals(closes, "NONEXIST")
    assert "error" in result


@pytest.mark.asyncio
async def test_market_context(closes):
    result = await _get_market_context(closes, vix=22.5, yield_curve=0.15, regime="BULL")
    assert result["regime"] == "BULL"
    assert result["vix"] == 22.5
    assert "spy_price" in result
    assert "sector_heatmap_1w" in result


@pytest.mark.asyncio
async def test_registration(closes):
    registry = ToolRegistry()
    register_market_tools(registry, closes, vix=20.0, yield_curve=0.1, regime="NEUTRAL")
    assert "get_price_and_technicals" in registry.tool_names
    assert "get_market_context" in registry.tool_names


@pytest.mark.asyncio
async def test_short_data():
    """With very short data, returns basic price without technicals."""
    dates = pd.date_range("2025-01-01", periods=5, freq="B")
    closes = pd.DataFrame({"XLK": [200, 201, 202, 203, 204]}, index=dates)
    result = await _get_price_and_technicals(closes, "XLK")
    assert result["price"] == 204.0
    assert "rsi_14" not in result  # Not enough data


@pytest.mark.asyncio
async def test_no_spy():
    """Market context works even without SPY in the data."""
    dates = pd.date_range("2025-01-01", periods=10, freq="B")
    closes = pd.DataFrame({"XLK": range(100, 110)}, index=dates)
    result = await _get_market_context(closes, vix=None, yield_curve=None, regime=None)
    assert result["regime"] == "Unknown"
    assert "spy_price" not in result

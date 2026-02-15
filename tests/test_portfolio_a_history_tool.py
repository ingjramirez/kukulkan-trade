"""Tests for get_portfolio_a_history tool."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.agent.tools.news import _get_portfolio_a_history


def _mock_trade(ticker: str, side: str, shares: float, price: float, reason: str = ""):
    t = MagicMock()
    t.executed_at = datetime(2026, 2, 15, 10, 0, tzinfo=timezone.utc)
    t.ticker = ticker
    t.side = side
    t.shares = shares
    t.price = price
    t.reason = reason
    return t


def _mock_ranking(ticker: str, return_63d: float, rank: int):
    r = MagicMock()
    r.ticker = ticker
    r.return_63d = return_63d
    r.rank = rank
    return r


@pytest.mark.asyncio
async def test_returns_recent_trades():
    db = MagicMock()
    db.get_trades = AsyncMock(
        return_value=[
            _mock_trade("QQQ", "BUY", 100, 400.0, "Momentum rotation into tech"),
            _mock_trade("XLE", "SELL", 50, 85.0, "Momentum rotation out of energy"),
        ]
    )
    db.get_latest_momentum_rankings = AsyncMock(
        return_value=[
            _mock_ranking("QQQ", 15.5, 1),
            _mock_ranking("XLK", 12.3, 2),
        ]
    )

    result = await _get_portfolio_a_history(db, "default")
    assert result["portfolio"] == "A"
    assert result["total_entries"] == 2
    assert result["recent_trades"][0]["ticker"] == "QQQ"
    assert result["recent_trades"][0]["side"] == "BUY"
    assert len(result["momentum_rankings_top5"]) == 2


@pytest.mark.asyncio
async def test_limits_trades():
    trades = [_mock_trade("QQQ", "BUY", 100, 400.0) for _ in range(20)]
    db = MagicMock()
    db.get_trades = AsyncMock(return_value=trades)
    db.get_latest_momentum_rankings = AsyncMock(return_value=[])

    result = await _get_portfolio_a_history(db, "default", n_trades=5)
    assert result["total_entries"] == 5


@pytest.mark.asyncio
async def test_empty_history():
    db = MagicMock()
    db.get_trades = AsyncMock(return_value=[])
    db.get_latest_momentum_rankings = AsyncMock(return_value=[])

    result = await _get_portfolio_a_history(db, "default")
    assert result["total_entries"] == 0
    assert result["recent_trades"] == []


@pytest.mark.asyncio
async def test_read_only_note():
    db = MagicMock()
    db.get_trades = AsyncMock(return_value=[])
    db.get_latest_momentum_rankings = AsyncMock(return_value=[])

    result = await _get_portfolio_a_history(db, "default")
    assert "Read-only" in result["note"]

"""Tests for the after-hours P&L endpoint."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.api.routes.portfolios import get_after_hours_pnl
from src.utils.market_time import MarketPhase


def _make_snapshot(total_value: float = 100000.0, cash: float = 20000.0) -> MagicMock:
    s = MagicMock()
    s.total_value = total_value
    s.cash = cash
    return s


def _make_position(ticker: str, shares: float, current_price: float, avg_price: float = 0.0) -> MagicMock:
    p = MagicMock()
    p.ticker = ticker
    p.shares = shares
    p.current_price = current_price
    p.avg_price = avg_price or current_price
    return p


def _make_db(
    snapshot: MagicMock | None = None,
    positions_b: list | None = None,
    positions_a: list | None = None,
) -> AsyncMock:
    db = AsyncMock()
    db.get_last_market_hours_snapshot = AsyncMock(return_value=snapshot)
    db.get_positions = AsyncMock(side_effect=lambda name, **kw: positions_b or [] if name == "B" else positions_a or [])
    return db


class TestAfterHoursPnl:
    async def test_returns_inactive_during_market_hours(self) -> None:
        db = _make_db()
        with patch("src.api.routes.portfolios.get_market_phase", return_value=MarketPhase.MARKET):
            result = await get_after_hours_pnl(tenant_id="default", db=db)
        assert result["is_active"] is False
        assert result["market_phase"] == "market"

    async def test_returns_inactive_during_closed(self) -> None:
        db = _make_db()
        with patch("src.api.routes.portfolios.get_market_phase", return_value=MarketPhase.CLOSED):
            result = await get_after_hours_pnl(tenant_id="default", db=db)
        assert result["is_active"] is False

    async def test_returns_active_during_afterhours(self) -> None:
        snapshot = _make_snapshot(100000.0, 20000.0)
        positions = [_make_position("AAPL", 100, 150.0)]
        db = _make_db(snapshot=snapshot, positions_b=positions)

        with (
            patch("src.api.routes.portfolios.get_market_phase", return_value=MarketPhase.AFTERHOURS),
            patch("src.data.market_data.get_extended_hours_prices", return_value={"AAPL": 152.0}),
        ):
            result = await get_after_hours_pnl(tenant_id="default", db=db)

        assert result["is_active"] is True
        assert result["market_phase"] == "afterhours"

    async def test_returns_active_during_premarket(self) -> None:
        snapshot = _make_snapshot(100000.0, 20000.0)
        positions = [_make_position("AAPL", 100, 150.0)]
        db = _make_db(snapshot=snapshot, positions_b=positions)

        with (
            patch("src.api.routes.portfolios.get_market_phase", return_value=MarketPhase.PREMARKET),
            patch("src.data.market_data.get_extended_hours_prices", return_value={"AAPL": 148.0}),
        ):
            result = await get_after_hours_pnl(tenant_id="default", db=db)

        assert result["is_active"] is True
        assert result["market_phase"] == "premarket"

    async def test_change_calculation_correct(self) -> None:
        # Close: 100 shares * 150 = 15000 + 20000 cash = 35000 (but snapshot says 100000)
        snapshot = _make_snapshot(100000.0, 20000.0)
        positions = [_make_position("AAPL", 100, 150.0)]
        db = _make_db(snapshot=snapshot, positions_b=positions)

        with (
            patch("src.api.routes.portfolios.get_market_phase", return_value=MarketPhase.AFTERHOURS),
            patch("src.data.market_data.get_extended_hours_prices", return_value={"AAPL": 155.0}),
        ):
            result = await get_after_hours_pnl(tenant_id="default", db=db)

        # current = 100*155 + 20000 cash = 35500
        expected_current = 100 * 155.0 + 20000.0
        assert result["current_value"] == expected_current
        assert result["change"] == expected_current - 100000.0

    async def test_movers_sorted_by_contribution(self) -> None:
        snapshot = _make_snapshot(100000.0, 10000.0)
        positions = [
            _make_position("AAPL", 100, 150.0),
            _make_position("MSFT", 50, 400.0),
        ]
        db = _make_db(snapshot=snapshot, positions_b=positions)

        with (
            patch("src.api.routes.portfolios.get_market_phase", return_value=MarketPhase.AFTERHOURS),
            patch("src.data.market_data.get_extended_hours_prices", return_value={"AAPL": 151.0, "MSFT": 405.0}),
        ):
            result = await get_after_hours_pnl(tenant_id="default", db=db)

        movers = result["movers"]
        assert len(movers) == 2
        # MSFT: 50*(405-400) = 250, AAPL: 100*(151-150) = 100
        assert movers[0]["ticker"] == "MSFT"
        assert movers[1]["ticker"] == "AAPL"

    async def test_handles_missing_extended_prices(self) -> None:
        snapshot = _make_snapshot(100000.0, 20000.0)
        positions = [_make_position("AAPL", 100, 150.0)]
        db = _make_db(snapshot=snapshot, positions_b=positions)

        with (
            patch("src.api.routes.portfolios.get_market_phase", return_value=MarketPhase.AFTERHOURS),
            patch("src.data.market_data.get_extended_hours_prices", return_value={}),
        ):
            result = await get_after_hours_pnl(tenant_id="default", db=db)

        assert result["is_active"] is True
        assert result["movers"] == []  # No prices = no movers

    async def test_empty_positions_returns_zero(self) -> None:
        snapshot = _make_snapshot(50000.0, 50000.0)
        db = _make_db(snapshot=snapshot)

        with patch("src.api.routes.portfolios.get_market_phase", return_value=MarketPhase.AFTERHOURS):
            result = await get_after_hours_pnl(tenant_id="default", db=db)

        assert result["is_active"] is True
        assert result["change"] == 0.0
        assert result["movers"] == []

    async def test_no_snapshot_returns_inactive(self) -> None:
        db = _make_db(snapshot=None)
        with patch("src.api.routes.portfolios.get_market_phase", return_value=MarketPhase.AFTERHOURS):
            result = await get_after_hours_pnl(tenant_id="default", db=db)
        assert result["is_active"] is False
        assert "reason" in result

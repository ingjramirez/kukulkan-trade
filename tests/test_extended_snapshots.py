"""Tests for extended hours intraday snapshots."""

from unittest.mock import AsyncMock, MagicMock, patch

from src.intraday import collect_intraday_snapshot
from src.utils.market_time import MarketPhase


def _make_tenant(run_a: bool = False, run_b: bool = True) -> MagicMock:
    t = MagicMock()
    t.id = "default"
    t.run_portfolio_a = run_a
    t.run_portfolio_b = run_b
    return t


def _make_portfolio(cash: float = 10000.0) -> MagicMock:
    p = MagicMock()
    p.cash = cash
    return p


def _make_position(ticker: str, shares: float, avg_price: float) -> MagicMock:
    p = MagicMock()
    p.ticker = ticker
    p.shares = shares
    p.avg_price = avg_price
    return p


def _make_db(
    portfolio: MagicMock | None = None,
    positions: list | None = None,
) -> AsyncMock:
    db = AsyncMock()
    db.get_portfolio = AsyncMock(return_value=portfolio or _make_portfolio())
    db.get_positions = AsyncMock(return_value=positions or [])
    db.save_intraday_snapshot = AsyncMock()
    return db


class TestExtendedSnapshots:
    async def test_premarket_snapshot_tagged_correctly(self) -> None:
        positions = [_make_position("AAPL", 10, 150.0)]
        db = _make_db(positions=positions)
        tenant = _make_tenant()

        with patch("src.intraday._fetch_extended_prices", return_value={"AAPL": 152.0}):
            saved = await collect_intraday_snapshot(db, tenant, MarketPhase.PREMARKET)

        assert saved == 1
        call_kwargs = db.save_intraday_snapshot.call_args[1]
        assert call_kwargs["is_extended_hours"] is True
        assert call_kwargs["market_phase"] == "premarket"

    async def test_afterhours_snapshot_tagged_correctly(self) -> None:
        positions = [_make_position("MSFT", 5, 400.0)]
        db = _make_db(positions=positions)
        tenant = _make_tenant()

        with patch("src.intraday._fetch_extended_prices", return_value={"MSFT": 402.0}):
            saved = await collect_intraday_snapshot(db, tenant, MarketPhase.AFTERHOURS)

        assert saved == 1
        call_kwargs = db.save_intraday_snapshot.call_args[1]
        assert call_kwargs["is_extended_hours"] is True
        assert call_kwargs["market_phase"] == "afterhours"

    async def test_market_snapshot_not_extended(self) -> None:
        positions = [_make_position("AAPL", 10, 150.0)]
        db = _make_db(positions=positions)
        tenant = _make_tenant()

        with patch("src.intraday._fetch_alpaca_prices", return_value={"AAPL": 151.0}):
            saved = await collect_intraday_snapshot(db, tenant, MarketPhase.MARKET)

        assert saved == 1
        call_kwargs = db.save_intraday_snapshot.call_args[1]
        assert call_kwargs["is_extended_hours"] is False
        assert call_kwargs["market_phase"] == "market"

    async def test_extended_uses_extended_prices(self) -> None:
        """Pre-market should call _fetch_extended_prices, not _fetch_alpaca_prices."""
        db = _make_db(positions=[_make_position("NVDA", 3, 800.0)])
        tenant = _make_tenant()

        with (
            patch("src.intraday._fetch_extended_prices", return_value={"NVDA": 810.0}) as ext_mock,
            patch("src.intraday._fetch_alpaca_prices") as alp_mock,
        ):
            await collect_intraday_snapshot(db, tenant, MarketPhase.PREMARKET)

        ext_mock.assert_called_once()
        alp_mock.assert_not_called()

    async def test_market_uses_alpaca_prices(self) -> None:
        db = _make_db(positions=[_make_position("NVDA", 3, 800.0)])
        tenant = _make_tenant()

        with (
            patch("src.intraday._fetch_extended_prices") as ext_mock,
            patch("src.intraday._fetch_alpaca_prices", return_value={"NVDA": 805.0}) as alp_mock,
        ):
            await collect_intraday_snapshot(db, tenant, MarketPhase.MARKET)

        alp_mock.assert_called_once()
        ext_mock.assert_not_called()

    async def test_extended_fetch_failure_returns_zero(self) -> None:
        db = _make_db()
        tenant = _make_tenant()

        with patch("src.intraday._fetch_extended_prices", return_value=None):
            saved = await collect_intraday_snapshot(db, tenant, MarketPhase.AFTERHOURS)

        assert saved == 0
        db.save_intraday_snapshot.assert_not_called()

    async def test_sse_event_includes_market_phase(self) -> None:
        positions = [_make_position("AAPL", 10, 150.0)]
        db = _make_db(positions=positions)
        tenant = _make_tenant()

        mock_bus = AsyncMock()
        mock_bus.publish = AsyncMock()

        with (
            patch("src.intraday._fetch_extended_prices", return_value={"AAPL": 152.0}),
            patch("src.events.event_bus.event_bus", mock_bus),
        ):
            await collect_intraday_snapshot(db, tenant, MarketPhase.AFTERHOURS)

        mock_bus.publish.assert_called_once()
        event = mock_bus.publish.call_args[0][0]
        assert event.data["market_phase"] == "afterhours"
        assert event.data["is_extended_hours"] is True

    async def test_default_phase_is_market(self) -> None:
        """collect_intraday_snapshot defaults to MARKET phase."""
        db = _make_db(positions=[_make_position("AAPL", 10, 150.0)])
        tenant = _make_tenant()

        with patch("src.intraday._fetch_alpaca_prices", return_value={"AAPL": 151.0}):
            saved = await collect_intraday_snapshot(db, tenant)

        assert saved == 1
        call_kwargs = db.save_intraday_snapshot.call_args[1]
        assert call_kwargs["market_phase"] == "market"

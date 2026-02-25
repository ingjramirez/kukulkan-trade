"""Tests for the daily pipeline orchestrator.

Uses mocked market data and mocked Claude agent — no external API calls.
"""

from datetime import date
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.orchestrator import Orchestrator
from src.storage.database import Database


def _make_market_data(tickers: list[str], days: int = 250) -> dict[str, pd.DataFrame]:
    """Generate synthetic OHLCV data for testing."""
    np.random.seed(42)
    dates = pd.bdate_range(end="2026-02-05", periods=days)
    data = {}
    for i, t in enumerate(tickers):
        drift = 0.2 - i * 0.01
        close = 100 + np.cumsum(np.random.normal(drift, 1.5, days))
        df = pd.DataFrame(
            {
                "Open": close * 0.999,
                "High": close * 1.01,
                "Low": close * 0.99,
                "Close": close,
                "Volume": np.random.uniform(1e6, 1e8, days),
            },
            index=dates,
        )
        data[t] = df
    return data


@pytest.fixture
async def orchestrator():
    """Create an orchestrator with in-memory DB."""
    db = Database(url="sqlite+aiosqlite:///:memory:")
    await db.init_db()
    orch = Orchestrator(db)
    yield orch
    await db.close()


class TestOrchestrator:
    @patch("src.orchestrator.MacroDataFetcher")
    @patch("src.data.market_data.yf")
    async def test_run_daily_portfolios_a_and_b(self, mock_yf, mock_macro_cls, orchestrator: Orchestrator) -> None:
        """Test that portfolios A and B run through the pipeline with mocked data."""
        # Create synthetic data for A and B universe tickers
        tickers = [
            "XLK",
            "XLF",
            "XLV",
            "XLE",
            "XLI",
            "XLY",
            "XLP",
            "XLU",
            "XLB",
            "XLRE",
            "QQQ",
            "SMH",
            "XBI",
            "IWM",
            "EFA",
            "EEM",
            "TLT",
            "HYG",
            "GDX",
            "ARKK",
            "SH",
            "PSQ",
            "TBF",
            "GLD",
            "SLV",
            "USO",
            "IBIT",
            "AAPL",
            "MSFT",
            "GOOGL",
            "AMZN",
            "NVDA",
        ]
        fake_data = _make_market_data(tickers)

        # Mock the market data fetcher
        orchestrator._market_data.fetch_universe = AsyncMock(return_value=fake_data)

        # Mock macro data
        orchestrator._macro_data.get_latest_yield_curve = MagicMock(return_value=1.2)
        orchestrator._macro_data.get_latest_vix = MagicMock(return_value=18.0)

        # Mock Portfolio B to avoid Claude Code CLI calls
        orchestrator._run_portfolio_b = AsyncMock(return_value=([], "Test reasoning", "No tools"))

        summary = await orchestrator.run_daily(today=date(2026, 2, 5))

        # Verify pipeline ran
        assert summary["date"] == "2026-02-05"
        assert summary["tickers_fetched"] == len(tickers)
        assert "A" in summary["trades"]
        assert "B" in summary["trades"]
        assert summary["trades_executed"] >= 0

        # Verify portfolios exist in DB
        for name in ("A", "B"):
            portfolio = await orchestrator._db.get_portfolio(name)
            assert portfolio is not None

    @patch("src.orchestrator.MacroDataFetcher")
    async def test_handles_empty_market_data(self, mock_macro_cls, orchestrator: Orchestrator) -> None:
        """Pipeline should handle empty market data gracefully."""
        orchestrator._market_data.fetch_universe = AsyncMock(return_value={})

        summary = await orchestrator.run_daily(today=date(2026, 2, 5))

        assert len(summary["errors"]) > 0
        assert "No market data" in summary["errors"][0]

    @patch("src.orchestrator.MacroDataFetcher")
    @patch("src.data.market_data.yf")
    async def test_snapshots_recorded(self, mock_yf, mock_macro_cls, orchestrator: Orchestrator) -> None:
        """Verify that daily snapshots are taken for all portfolios."""
        tickers = ["XLK", "XLF", "QQQ", "GLD", "IBIT"]
        fake_data = _make_market_data(tickers)

        orchestrator._market_data.fetch_universe = AsyncMock(return_value=fake_data)
        orchestrator._macro_data.get_latest_yield_curve = MagicMock(return_value=0.5)
        orchestrator._macro_data.get_latest_vix = MagicMock(return_value=20.0)

        # Mock Portfolio B to avoid Claude Code CLI calls
        orchestrator._run_portfolio_b = AsyncMock(return_value=([], "Test reasoning", "No tools"))

        await orchestrator.run_daily(today=date(2026, 2, 5))

        # Check snapshots exist for both portfolios
        for name in ("A", "B"):
            snapshots = await orchestrator._db.get_snapshots(name)
            assert len(snapshots) == 1
            assert snapshots[0].date == date(2026, 2, 5)
            assert snapshots[0].total_value > 0


class TestPortfolioAReason:
    """Tests for _run_portfolio_a returning (trades, reason) tuple."""

    @patch("src.orchestrator.MacroDataFetcher")
    @patch("src.data.market_data.yf")
    async def test_holding_target_returns_reason(self, mock_yf, mock_macro_cls, orchestrator: Orchestrator) -> None:
        """When target is already held, reason explains hold."""
        tickers = [
            "XLK",
            "XLF",
            "XLV",
            "XLE",
            "XLI",
            "XLY",
            "XLP",
            "XLU",
            "XLB",
            "XLRE",
            "QQQ",
            "SMH",
            "XBI",
            "IWM",
            "EFA",
            "EEM",
            "TLT",
            "HYG",
            "GDX",
            "ARKK",
        ]
        fake_data = _make_market_data(tickers)
        closes = pd.DataFrame({t: df["Close"] for t, df in fake_data.items()}).sort_index()

        # Pre-populate with the momentum target already held
        target = orchestrator._strategy_a.get_target_ticker(closes)
        assert target is not None

        # Initialize portfolio and add position for target
        await orchestrator._executor.initialize_portfolios()
        await orchestrator._db.upsert_position(
            portfolio="A",
            ticker=target,
            shares=100,
            avg_price=100.0,
        )

        trades, reason = await orchestrator._run_portfolio_a(
            closes,
            date(2026, 2, 5),
        )
        assert trades == []
        assert f"Holding momentum target {target}" in reason

    @patch("src.orchestrator.MacroDataFetcher")
    @patch("src.data.market_data.yf")
    async def test_rebalance_returns_reason(self, mock_yf, mock_macro_cls, orchestrator: Orchestrator) -> None:
        """When trades are generated, reason mentions rebalancing."""
        tickers = [
            "XLK",
            "XLF",
            "XLV",
            "XLE",
            "XLI",
            "XLY",
            "XLP",
            "XLU",
            "XLB",
            "XLRE",
            "QQQ",
            "SMH",
            "XBI",
            "IWM",
            "EFA",
            "EEM",
            "TLT",
            "HYG",
            "GDX",
            "ARKK",
        ]
        fake_data = _make_market_data(tickers)
        closes = pd.DataFrame({t: df["Close"] for t, df in fake_data.items()}).sort_index()

        # Initialize portfolio with cash only — should trigger a buy
        await orchestrator._executor.initialize_portfolios()

        trades, reason = await orchestrator._run_portfolio_a(
            closes,
            date(2026, 2, 5),
        )
        assert len(trades) > 0
        assert "Rebalancing to" in reason


class TestBuildSyncWarning:
    """Tests for Orchestrator._build_sync_warning static method."""

    def test_none_input(self):
        assert Orchestrator._build_sync_warning(None) is None

    def test_clean_sync(self):
        result = {"alpaca": [{"ticker": "SPY"}], "drift": [], "corrections": []}
        assert Orchestrator._build_sync_warning(result) is None

    def test_sync_failure(self):
        result = {"alpaca": [], "drift": [], "corrections": [], "error": "Connection timed out"}
        warning = Orchestrator._build_sync_warning(result)
        assert "WARNING" in warning
        assert "stale" in warning
        assert "get_portfolio" in warning

    def test_drift_corrected(self):
        result = {
            "alpaca": [],
            "drift": [{"ticker": "SHY"}],
            "corrections": [{"ticker": "SHY", "alpaca_qty": 0, "db_qty": 50}],
        }
        warning = Orchestrator._build_sync_warning(result)
        assert "drift detected and corrected" in warning
        assert "1 position(s)" in warning


class TestBuildSyncMetadata:
    """Tests for Orchestrator._build_sync_metadata static method."""

    def test_none_input(self):
        assert Orchestrator._build_sync_metadata(None) is None

    def test_sync_failure(self):
        result = {"alpaca": [], "drift": [], "corrections": [], "error": "Timeout"}
        meta = Orchestrator._build_sync_metadata(result)
        assert meta["success"] is False
        assert meta["error"] == "Timeout"
        assert meta["drift_corrections"] == 0

    def test_clean_sync(self):
        result = {"alpaca": [{"ticker": "SPY"}], "drift": [], "corrections": []}
        meta = Orchestrator._build_sync_metadata(result)
        assert meta["success"] is True
        assert meta["drift_corrections"] == 0
        assert "corrections" not in meta

    def test_drift_corrected(self):
        corrections = [{"ticker": "SHY", "alpaca_qty": 0, "db_qty": 50}]
        result = {"alpaca": [], "drift": [{"ticker": "SHY"}], "corrections": corrections}
        meta = Orchestrator._build_sync_metadata(result)
        assert meta["success"] is True
        assert meta["drift_corrections"] == 1
        assert meta["corrections"] == corrections

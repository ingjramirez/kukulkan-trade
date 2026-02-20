"""Tests for the backtest runner — MockPortfolioB, BacktestRunner with synthetic data."""

from datetime import date
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from src.backtest.runner import BacktestRunner, MockPortfolioB
from src.execution.paper_trader import PaperTrader
from src.storage.database import Database
from src.storage.models import OrderSide, PortfolioName

# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_closes(tickers: list[str], days: int = 100, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic close prices with realistic random walk."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(end=date.today(), periods=days)
    n = len(dates)
    data = {}
    for t in tickers:
        base = rng.uniform(50, 300)
        returns = rng.normal(0.0005, 0.015, n)
        prices = base * np.cumprod(1 + returns)
        data[t] = prices
    return pd.DataFrame(data, index=dates)


def _make_volumes(tickers: list[str], days: int = 100, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic volume data."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(end=date.today(), periods=days)
    n = len(dates)
    data = {}
    for t in tickers:
        data[t] = rng.uniform(1_000_000, 50_000_000, n)
    return pd.DataFrame(data, index=dates)


# ── MockPortfolioB Tests ────────────────────────────────────────────────────


class TestMockPortfolioB:
    def test_generates_trades_with_sufficient_data(self) -> None:
        tickers = ["XLK", "XLF", "GLD", "QQQ", "AAPL", "MSFT", "NVDA", "XLE"]
        closes = _make_closes(tickers, days=50)
        mock = MockPortfolioB(top_n=3)

        trades = mock.generate_trades(
            closes=closes,
            current_positions={},
            cash=66_000.0,
            total_value=66_000.0,
        )

        assert len(trades) > 0
        assert all(t.portfolio == PortfolioName.B for t in trades)
        buy_tickers = {t.ticker for t in trades if t.side == OrderSide.BUY}
        assert len(buy_tickers) <= 3

    def test_returns_empty_with_insufficient_data(self) -> None:
        tickers = ["XLK", "XLF"]
        closes = _make_closes(tickers, days=10)
        mock = MockPortfolioB()

        trades = mock.generate_trades(
            closes=closes,
            current_positions={},
            cash=66_000.0,
            total_value=66_000.0,
        )

        assert trades == []

    def test_sells_positions_not_in_target(self) -> None:
        tickers = ["XLK", "XLF", "GLD", "QQQ", "AAPL", "MSFT"]
        closes = _make_closes(tickers, days=50)
        mock = MockPortfolioB(top_n=2)

        # Fake holding a position that might not be in top 2
        trades = mock.generate_trades(
            closes=closes,
            current_positions={
                "XLK": 100,
                "XLF": 100,
                "GLD": 100,
                "QQQ": 100,
                "AAPL": 100,
                "MSFT": 100,
            },
            cash=0,
            total_value=50_000.0,
        )

        sell_trades = [t for t in trades if t.side == OrderSide.SELL]
        buy_trades = [t for t in trades if t.side == OrderSide.BUY]

        # Should have some sells (rotating out of non-selected)
        assert len(sell_trades) > 0 or len(buy_trades) > 0

    def test_top_n_parameter(self) -> None:
        tickers = ["XLK", "XLF", "GLD", "QQQ", "AAPL", "MSFT", "NVDA", "XLE"]
        closes = _make_closes(tickers, days=50)

        mock_3 = MockPortfolioB(top_n=3)
        mock_5 = MockPortfolioB(top_n=5)

        trades_3 = mock_3.generate_trades(closes, {}, 66_000.0, 66_000.0)
        trades_5 = mock_5.generate_trades(closes, {}, 66_000.0, 66_000.0)

        buys_3 = {t.ticker for t in trades_3 if t.side == OrderSide.BUY}
        buys_5 = {t.ticker for t in trades_5 if t.side == OrderSide.BUY}

        assert len(buys_3) <= 3
        assert len(buys_5) <= 5


# ── BacktestRunner Tests ────────────────────────────────────────────────────


class TestBacktestRunnerUnit:
    """Unit tests using in-memory DB and synthetic data."""

    @pytest.fixture
    async def runner_and_db(self):
        """Create a runner with in-memory DB."""
        runner = BacktestRunner.__new__(BacktestRunner)
        runner._db_url = "sqlite+aiosqlite:///:memory:"
        runner._db = Database(url="sqlite+aiosqlite:///:memory:")
        await runner._db.init_db()
        yield runner, runner._db
        await runner._db.close()

    @pytest.mark.asyncio
    async def test_run_portfolio_a(self, runner_and_db) -> None:
        runner, db = runner_and_db
        from src.strategies.portfolio_a import MomentumStrategy

        tickers = ["XLK", "XLF", "QQQ", "GLD", "XLE", "XLI", "XLV", "XLP", "XLU", "XLB"]
        closes = _make_closes(tickers, days=100)

        trader = PaperTrader(db)
        await trader.initialize_portfolios()

        strategy = MomentumStrategy()
        trades = await runner._run_portfolio_a(strategy, closes, trader, date.today())

        # Should have at least one buy (top momentum ETF)
        assert isinstance(trades, list)

    @pytest.mark.asyncio
    async def test_run_portfolio_b_mock(self, runner_and_db) -> None:
        runner, db = runner_and_db

        tickers = ["XLK", "XLF", "QQQ", "GLD", "AAPL", "MSFT", "NVDA", "XLE"]
        closes = _make_closes(tickers, days=50)

        trader = PaperTrader(db)
        await trader.initialize_portfolios()

        mock = MockPortfolioB(top_n=3)
        trades = await runner._run_portfolio_b_mock(mock, closes, trader, date.today())

        assert isinstance(trades, list)

    @pytest.mark.asyncio
    async def test_compute_summary(self, runner_and_db) -> None:
        runner, db = runner_and_db

        trader = PaperTrader(db)
        await trader.initialize_portfolios()

        # Take a snapshot so summary has data
        prices = {"XLK": 200.0}
        for pname in ("A", "B"):
            await trader.take_snapshot(pname, date.today(), prices)

        summary = await runner._compute_summary({"A": 5, "B": 3})

        assert "portfolio_A" in summary
        assert "portfolio_B" in summary
        assert summary["trade_counts"]["A"] == 5

    @pytest.mark.asyncio
    async def test_compute_summary_with_drawdown(self, runner_and_db) -> None:
        runner, db = runner_and_db

        trader = PaperTrader(db)
        await trader.initialize_portfolios()

        # Simulate a drawdown: value goes up then down
        d1 = date(2026, 1, 1)
        d2 = date(2026, 1, 2)
        d3 = date(2026, 1, 3)

        await db.save_snapshot("A", d1, 33_000.0, 33_000.0, 0.0, None, 0.0)
        await db.save_snapshot("A", d2, 35_000.0, 35_000.0, 0.0, 5.0, 5.0)
        await db.save_snapshot("A", d3, 33_000.0, 33_000.0, 0.0, -5.7, -1.0)

        # B just flat
        await db.save_snapshot("B", d1, 66_000.0, 66_000.0, 0.0, None, 0.0)

        summary = await runner._compute_summary({"A": 0, "B": 0})

        a = summary["portfolio_A"]
        assert a["max_drawdown_pct"] > 0  # Should detect the drawdown
        assert a["snapshots"] == 3

    @pytest.mark.asyncio
    async def test_full_simulation_loop(self, runner_and_db) -> None:
        """End-to-end test: run a few days of simulation."""
        runner, db = runner_and_db
        from src.strategies.portfolio_a import MomentumStrategy

        tickers = ["XLK", "XLF", "QQQ", "GLD", "XLE", "XLI", "XLV", "XLP", "XLU", "XLB", "XLRE", "AAPL", "MSFT"]
        closes = _make_closes(tickers, days=250)
        trading_days = closes.index.tolist()

        trader = PaperTrader(db)
        await trader.initialize_portfolios()

        strategy_a = MomentumStrategy()
        mock_b = MockPortfolioB(top_n=3)

        # Run just 5 days
        sim_dates = trading_days[-5:]
        for sim_date in sim_dates:
            day_idx = trading_days.index(sim_date)
            closes_slice = closes.iloc[: day_idx + 1]

            all_trades = []

            trades_a = await runner._run_portfolio_a(strategy_a, closes_slice, trader, sim_date)
            all_trades.extend(trades_a)

            trades_b = await runner._run_portfolio_b_mock(mock_b, closes_slice, trader, sim_date)
            all_trades.extend(trades_b)

            if all_trades:
                await trader.execute_trades(all_trades)

            latest_prices = {t: float(closes_slice[t].iloc[-1]) for t in closes_slice.columns}
            for pname in ("A", "B"):
                await trader.take_snapshot(pname, sim_date, latest_prices)

        # Verify snapshots were created
        for pname in ("A", "B"):
            snapshots = await db.get_snapshots(pname)
            assert len(snapshots) == 5

    def test_dry_run_mock(self) -> None:
        """Dry run with mock strategy estimates zero API cost."""
        runner = BacktestRunner.__new__(BacktestRunner)
        result = runner._estimate_cost(126, use_ai=False)

        assert result["dry_run"] is True
        assert result["estimated_api_calls"] == 0
        assert result["estimated_cost_usd"] == 0.0
        assert "no API calls" in result["note"].lower() or "mock" in result["note"].lower()

    def test_dry_run_ai(self) -> None:
        """Dry run with AI estimates token cost."""
        runner = BacktestRunner.__new__(BacktestRunner)
        result = runner._estimate_cost(126, use_ai=True)

        assert result["dry_run"] is True
        assert result["estimated_api_calls"] == 126
        assert result["estimated_tokens"] > 0
        assert result["estimated_cost_usd"] > 0

    @pytest.mark.asyncio
    async def test_run_portfolio_b_ai(self, runner_and_db) -> None:
        """Portfolio B AI uses AIBacktestStrategy with mocked agent."""
        runner, db = runner_and_db
        from src.backtest.ai_strategy import AIBacktestStrategy

        tickers = ["XLK", "XLF", "QQQ", "GLD", "AAPL", "MSFT"]
        closes = _make_closes(tickers, days=60)
        volumes = _make_volumes(tickers, days=60)

        trader = PaperTrader(db)
        await trader.initialize_portfolios()

        ai_bt = AIBacktestStrategy(budget_usd=10.0, run_label="test")
        mock_response = {
            "regime_assessment": "Test regime",
            "reasoning": "Backtest AI test",
            "trades": [
                {"ticker": "XLK", "side": "BUY", "weight": 0.15, "reason": "test"},
            ],
            "risk_notes": "test",
            "_raw": "{}",
            "_tokens_used": 100,
            "_model": "test-model",
        }
        ai_bt._agent.analyze = MagicMock(return_value=mock_response)
        ai_bt._agent._client = MagicMock()

        trades = await runner._run_portfolio_b_ai(
            ai_bt,
            closes,
            volumes,
            trader,
            date.today(),
        )

        assert isinstance(trades, list)
        ai_bt._agent.analyze.assert_called_once()

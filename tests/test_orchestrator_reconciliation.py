"""Tests for equity reconciliation (Step 8 Alpaca prices + Step 8.5 drift correction).

Isolated from the main orchestrator test file for clarity.
"""

from datetime import date
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.orchestrator import Orchestrator
from src.storage.database import Database
from src.utils.allocations import resolve_allocations


def _make_closes(tickers: list[str], days: int = 5) -> pd.DataFrame:
    """Build a simple closes DataFrame for tests."""
    np.random.seed(42)
    dates = pd.bdate_range(end="2026-02-05", periods=days)
    data = {t: 100.0 + np.random.normal(0, 1, days) for t in tickers}
    return pd.DataFrame(data, index=dates)


def _make_alpaca_position(symbol: str, current_price: float) -> SimpleNamespace:
    """Fake Alpaca position object."""
    return SimpleNamespace(symbol=symbol, current_price=current_price)


def _make_alpaca_account(equity: float) -> SimpleNamespace:
    """Fake Alpaca account object."""
    return SimpleNamespace(equity=equity)


@pytest.fixture
async def db():
    """In-memory database."""
    database = Database(url="sqlite+aiosqlite:///:memory:")
    await database.init_db()
    yield database
    await database.close()


@pytest.fixture
async def orch(db: Database):
    """Orchestrator with mocked notifier (no Telegram)."""
    o = Orchestrator(db)
    o._notifier._token = ""
    o._notifier._chat_id = ""
    return o


# ── Step 8: Alpaca prices for snapshots ──────────────────────────────────


class TestAlpacaSnapshotPrices:
    async def test_alpaca_prices_used_for_snapshots(self, orch: Orchestrator) -> None:
        """When executor has _client, Alpaca position prices override yfinance."""
        closes = _make_closes(["AAPL", "MSFT", "GOOGL"])

        # Alpaca says AAPL = 155.50 (different from yfinance ~100)
        mock_client = MagicMock()
        mock_client.get_all_positions.return_value = [
            _make_alpaca_position("AAPL", 155.50),
        ]
        orch._executor = MagicMock()
        orch._executor._client = mock_client
        orch._executor.initialize_portfolios = AsyncMock()
        orch._executor.execute_trades = AsyncMock(return_value=[])
        orch._executor.take_snapshot = AsyncMock()

        # Set up the pipeline to run minimally
        orch._market_data.fetch_universe = AsyncMock(
            return_value={
                t: pd.DataFrame(
                    {
                        "Open": closes[t],
                        "High": closes[t],
                        "Low": closes[t],
                        "Close": closes[t],
                        "Volume": [1e6] * len(closes),
                    },
                    index=closes.index,
                )
                for t in closes.columns
            }
        )
        orch._macro_data.get_latest_yield_curve = MagicMock(return_value=None)
        orch._macro_data.get_latest_vix = MagicMock(return_value=None)
        mock_response = {
            "reasoning": "hold",
            "trades": [],
            "_raw": "{}",
            "_tokens_used": 0,
            "_model": "test",
        }
        orch._strategy_b._agent.analyze = MagicMock(return_value=mock_response)
        orch._strategy_b._agent._client = MagicMock()

        with patch("src.orchestrator.is_market_open", return_value=True):
            await orch.run_daily(today=date(2026, 2, 5))

        # Verify take_snapshot received Alpaca price for AAPL
        call_args = orch._executor.take_snapshot.call_args_list
        assert len(call_args) >= 1
        for call in call_args:
            prices = call.kwargs.get("latest_prices") or call.args[2]
            # AAPL should use Alpaca price, not yfinance
            assert prices["AAPL"] == 155.50
            # MSFT not in Alpaca positions → should use yfinance
            assert prices["MSFT"] != 155.50
            assert abs(prices["MSFT"] - float(closes["MSFT"].iloc[-1])) < 0.01

    async def test_alpaca_prices_fallback_to_yfinance(
        self,
        orch: Orchestrator,
    ) -> None:
        """Tickers not held on Alpaca fall back to yfinance closes."""
        closes = _make_closes(["AAPL", "MSFT"])

        mock_client = MagicMock()
        # Alpaca has no positions at all
        mock_client.get_all_positions.return_value = []
        orch._executor = MagicMock()
        orch._executor._client = mock_client
        orch._executor.initialize_portfolios = AsyncMock()
        orch._executor.execute_trades = AsyncMock(return_value=[])
        orch._executor.take_snapshot = AsyncMock()

        orch._market_data.fetch_universe = AsyncMock(
            return_value={
                t: pd.DataFrame(
                    {
                        "Open": closes[t],
                        "High": closes[t],
                        "Low": closes[t],
                        "Close": closes[t],
                        "Volume": [1e6] * len(closes),
                    },
                    index=closes.index,
                )
                for t in closes.columns
            }
        )
        orch._macro_data.get_latest_yield_curve = MagicMock(return_value=None)
        orch._macro_data.get_latest_vix = MagicMock(return_value=None)
        mock_response = {
            "reasoning": "hold",
            "trades": [],
            "_raw": "{}",
            "_tokens_used": 0,
            "_model": "test",
        }
        orch._strategy_b._agent.analyze = MagicMock(return_value=mock_response)
        orch._strategy_b._agent._client = MagicMock()

        with patch("src.orchestrator.is_market_open", return_value=True):
            await orch.run_daily(today=date(2026, 2, 5))

        call_args = orch._executor.take_snapshot.call_args_list
        assert len(call_args) >= 1
        for call in call_args:
            prices = call.kwargs.get("latest_prices") or call.args[2]
            # Both should use yfinance since Alpaca has no positions
            for t in ("AAPL", "MSFT"):
                assert abs(prices[t] - float(closes[t].iloc[-1])) < 0.01


# ── Step 8.5: _reconcile_equity ──────────────────────────────────────────


class TestReconcileEquity:
    async def test_reconcile_adjusts_cash_when_drift_positive(
        self,
        orch: Orchestrator,
        db: Database,
    ) -> None:
        """Tracked < Alpaca by $30 → cash increased."""
        # Set up tracked portfolios: A=33000, B=66000 → total 99000
        await db.upsert_portfolio("A", cash=10000, total_value=33000)
        await db.upsert_portfolio("B", cash=20000, total_value=66000)

        alloc = resolve_allocations(
            initial_equity=99000,
            portfolio_a_pct=33.33,
            portfolio_b_pct=66.67,
        )

        # Alpaca says equity = 99030 → drift = +30
        mock_client = MagicMock()
        mock_client.get_account.return_value = _make_alpaca_account(99030)
        orch._executor = MagicMock()
        orch._executor._client = mock_client

        drift = await orch._reconcile_equity(
            "default",
            run_portfolio_a=True,
            run_portfolio_b=True,
            allocations=alloc,
        )

        assert drift is not None
        assert abs(drift - 30.0) < 0.01

        # Verify cash was increased
        pa = await db.get_portfolio("A")
        pb = await db.get_portfolio("B")
        assert pa.cash > 10000
        assert pb.cash > 20000
        # Total adjustment should equal drift
        total_adj = (pa.total_value - 33000) + (pb.total_value - 66000)
        assert abs(total_adj - 30.0) < 0.01

    async def test_reconcile_adjusts_cash_when_drift_negative(
        self,
        orch: Orchestrator,
        db: Database,
    ) -> None:
        """Tracked > Alpaca by $25 → cash decreased."""
        await db.upsert_portfolio("A", cash=10000, total_value=33000)
        await db.upsert_portfolio("B", cash=20000, total_value=66000)

        alloc = resolve_allocations(
            initial_equity=99000,
            portfolio_a_pct=33.33,
            portfolio_b_pct=66.67,
        )

        # Alpaca says equity = 98975 → drift = -25
        mock_client = MagicMock()
        mock_client.get_account.return_value = _make_alpaca_account(98975)
        orch._executor = MagicMock()
        orch._executor._client = mock_client

        drift = await orch._reconcile_equity(
            "default",
            run_portfolio_a=True,
            run_portfolio_b=True,
            allocations=alloc,
        )

        assert drift is not None
        assert abs(drift - (-25.0)) < 0.01

        pa = await db.get_portfolio("A")
        pb = await db.get_portfolio("B")
        assert pa.cash < 10000
        assert pb.cash < 20000

    async def test_reconcile_skips_below_threshold(
        self,
        orch: Orchestrator,
        db: Database,
    ) -> None:
        """Drift < $10 → no reconciliation."""
        await db.upsert_portfolio("A", cash=10000, total_value=33000)
        await db.upsert_portfolio("B", cash=20000, total_value=66000)

        alloc = resolve_allocations(
            initial_equity=99000,
            portfolio_a_pct=33.33,
            portfolio_b_pct=66.67,
        )

        # Drift = +5 (below threshold)
        mock_client = MagicMock()
        mock_client.get_account.return_value = _make_alpaca_account(99005)
        orch._executor = MagicMock()
        orch._executor._client = mock_client

        drift = await orch._reconcile_equity(
            "default",
            run_portfolio_a=True,
            run_portfolio_b=True,
            allocations=alloc,
        )

        assert drift is None

        # Cash unchanged
        pa = await db.get_portfolio("A")
        assert pa.cash == 10000

    async def test_reconcile_skips_deposit_range(
        self,
        orch: Orchestrator,
        db: Database,
    ) -> None:
        """Drift > $50 positive → deferred to deposit detection."""
        await db.upsert_portfolio("A", cash=10000, total_value=33000)
        await db.upsert_portfolio("B", cash=20000, total_value=66000)

        alloc = resolve_allocations(
            initial_equity=99000,
            portfolio_a_pct=33.33,
            portfolio_b_pct=66.67,
        )

        # Drift = +100 (above deposit threshold)
        mock_client = MagicMock()
        mock_client.get_account.return_value = _make_alpaca_account(99100)
        orch._executor = MagicMock()
        orch._executor._client = mock_client

        drift = await orch._reconcile_equity(
            "default",
            run_portfolio_a=True,
            run_portfolio_b=True,
            allocations=alloc,
        )

        assert drift is None

    async def test_reconcile_skips_paper_trader(
        self,
        orch: Orchestrator,
        db: Database,
    ) -> None:
        """PaperTrader has no _client → skip reconciliation."""
        alloc = resolve_allocations(initial_equity=99000)

        # Default executor is PaperTrader (no _client attr)
        assert not hasattr(orch._executor, "_client")

        drift = await orch._reconcile_equity(
            "default",
            run_portfolio_a=True,
            run_portfolio_b=True,
            allocations=alloc,
        )

        assert drift is None

    async def test_reconcile_splits_proportionally(
        self,
        orch: Orchestrator,
        db: Database,
    ) -> None:
        """Both portfolios enabled → drift split by allocation pct."""
        await db.upsert_portfolio("A", cash=10000, total_value=33000)
        await db.upsert_portfolio("B", cash=20000, total_value=66000)

        # 33.33% / 66.67% split
        alloc = resolve_allocations(
            initial_equity=99000,
            portfolio_a_pct=33.33,
            portfolio_b_pct=66.67,
        )

        # Drift = +30
        mock_client = MagicMock()
        mock_client.get_account.return_value = _make_alpaca_account(99030)
        orch._executor = MagicMock()
        orch._executor._client = mock_client

        await orch._reconcile_equity(
            "default",
            run_portfolio_a=True,
            run_portfolio_b=True,
            allocations=alloc,
        )

        pa = await db.get_portfolio("A")
        pb = await db.get_portfolio("B")

        # A gets ~33.33% of 30 = ~10, B gets ~66.67% of 30 = ~20
        a_adj = pa.total_value - 33000
        b_adj = pb.total_value - 66000
        assert abs(a_adj - 10.0) < 0.5
        assert abs(b_adj - 20.0) < 0.5

    async def test_reconcile_single_portfolio(
        self,
        orch: Orchestrator,
        db: Database,
    ) -> None:
        """Only B enabled → entire drift goes to B."""
        await db.upsert_portfolio("B", cash=20000, total_value=66000)

        alloc = resolve_allocations(
            initial_equity=99000,
            portfolio_a_pct=33.33,
            portfolio_b_pct=66.67,
        )

        # Drift = +30 (only B tracked → broker 66030)
        mock_client = MagicMock()
        mock_client.get_account.return_value = _make_alpaca_account(66030)
        orch._executor = MagicMock()
        orch._executor._client = mock_client

        drift = await orch._reconcile_equity(
            "default",
            run_portfolio_a=False,
            run_portfolio_b=True,
            allocations=alloc,
        )

        assert drift is not None
        assert abs(drift - 30.0) < 0.01

        pb = await db.get_portfolio("B")
        assert abs(pb.total_value - 66030) < 0.01
        assert abs(pb.cash - 20030) < 0.01

    async def test_reconcile_large_negative_drift(
        self,
        orch: Orchestrator,
        db: Database,
    ) -> None:
        """Large negative drift (tracked > Alpaca by $200) is corrected.

        Negative drift has no deposit threshold guard.
        """
        await db.upsert_portfolio("A", cash=10000, total_value=33000)
        await db.upsert_portfolio("B", cash=20000, total_value=66000)

        alloc = resolve_allocations(
            initial_equity=99000,
            portfolio_a_pct=33.33,
            portfolio_b_pct=66.67,
        )

        # Drift = -200
        mock_client = MagicMock()
        mock_client.get_account.return_value = _make_alpaca_account(98800)
        orch._executor = MagicMock()
        orch._executor._client = mock_client

        drift = await orch._reconcile_equity(
            "default",
            run_portfolio_a=True,
            run_portfolio_b=True,
            allocations=alloc,
        )

        assert drift is not None
        assert abs(drift - (-200.0)) < 0.01

        pa = await db.get_portfolio("A")
        pb = await db.get_portfolio("B")
        total_adj = (pa.total_value - 33000) + (pb.total_value - 66000)
        assert abs(total_adj - (-200.0)) < 0.01

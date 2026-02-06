"""Tests for the daily pipeline orchestrator.

Uses mocked market data and mocked Claude agent — no external API calls.
"""

from datetime import date
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.agent.complexity_detector import ComplexityResult
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
        df = pd.DataFrame({
            "Open": close * 0.999,
            "High": close * 1.01,
            "Low": close * 0.99,
            "Close": close,
            "Volume": np.random.uniform(1e6, 1e8, days),
        }, index=dates)
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
    async def test_run_daily_portfolios_a_and_b(
        self, mock_yf, mock_macro_cls, orchestrator: Orchestrator
    ) -> None:
        """Test that portfolios A and B run through the pipeline with mocked data."""
        # Create synthetic data for A and B universe tickers
        tickers = [
            "XLK", "XLF", "XLV", "XLE", "XLI", "XLY", "XLP", "XLU", "XLB", "XLRE",
            "QQQ", "SMH", "XBI", "IWM", "EFA", "EEM", "TLT", "HYG", "GDX", "ARKK",
            "SH", "PSQ", "TBF", "GLD", "SLV", "USO", "IBIT",
            "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",
        ]
        fake_data = _make_market_data(tickers)

        # Mock the market data fetcher
        orchestrator._market_data.fetch_universe = AsyncMock(return_value=fake_data)

        # Mock macro data
        orchestrator._macro_data.get_latest_yield_curve = MagicMock(return_value=1.2)
        orchestrator._macro_data.get_latest_vix = MagicMock(return_value=18.0)

        # Mock the Claude agent to avoid API calls
        mock_response = {
            "regime_assessment": "Neutral test environment",
            "reasoning": "This is a test. Buying XLK for momentum.",
            "trades": [
                {"ticker": "XLK", "side": "BUY", "weight": 0.15, "reason": "test buy"},
            ],
            "risk_notes": "Test only",
            "_raw": "{}",
            "_tokens_used": 100,
            "_model": "test-model",
        }
        orchestrator._strategy_b._agent.analyze = MagicMock(return_value=mock_response)
        orchestrator._strategy_b._agent._client = MagicMock()

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
    async def test_handles_empty_market_data(
        self, mock_macro_cls, orchestrator: Orchestrator
    ) -> None:
        """Pipeline should handle empty market data gracefully."""
        orchestrator._market_data.fetch_universe = AsyncMock(return_value={})

        summary = await orchestrator.run_daily(today=date(2026, 2, 5))

        assert len(summary["errors"]) > 0
        assert "No market data" in summary["errors"][0]

    @patch("src.orchestrator.MacroDataFetcher")
    @patch("src.data.market_data.yf")
    async def test_snapshots_recorded(
        self, mock_yf, mock_macro_cls, orchestrator: Orchestrator
    ) -> None:
        """Verify that daily snapshots are taken for all portfolios."""
        tickers = ["XLK", "XLF", "QQQ", "GLD", "IBIT"]
        fake_data = _make_market_data(tickers)

        orchestrator._market_data.fetch_universe = AsyncMock(return_value=fake_data)
        orchestrator._macro_data.get_latest_yield_curve = MagicMock(return_value=0.5)
        orchestrator._macro_data.get_latest_vix = MagicMock(return_value=20.0)

        mock_response = {
            "regime_assessment": "test",
            "reasoning": "test",
            "trades": [],
            "risk_notes": "test",
            "_raw": "{}",
            "_tokens_used": 50,
            "_model": "test",
        }
        orchestrator._strategy_b._agent.analyze = MagicMock(return_value=mock_response)
        orchestrator._strategy_b._agent._client = MagicMock()

        await orchestrator.run_daily(today=date(2026, 2, 5))

        # Check snapshots exist for both portfolios
        for name in ("A", "B"):
            snapshots = await orchestrator._db.get_snapshots(name)
            assert len(snapshots) == 1
            assert snapshots[0].date == date(2026, 2, 5)
            assert snapshots[0].total_value > 0

    @patch("src.orchestrator.MacroDataFetcher")
    @patch("src.data.market_data.yf")
    async def test_agent_decision_persisted(
        self, mock_yf, mock_macro_cls, orchestrator: Orchestrator
    ) -> None:
        """Verify that Claude's decision is saved to the database."""
        tickers = ["XLK", "XLF", "QQQ", "GLD", "IBIT"]
        fake_data = _make_market_data(tickers)

        orchestrator._market_data.fetch_universe = AsyncMock(return_value=fake_data)
        orchestrator._macro_data.get_latest_yield_curve = MagicMock(return_value=1.0)
        orchestrator._macro_data.get_latest_vix = MagicMock(return_value=15.0)

        mock_response = {
            "regime_assessment": "Bullish test",
            "reasoning": "Testing persistence of agent decisions",
            "trades": [{"ticker": "XLK", "side": "BUY", "weight": 0.20, "reason": "test"}],
            "risk_notes": "none",
            "_raw": '{"test": true}',
            "_tokens_used": 200,
            "_model": "claude-sonnet-4-5-20250929",
        }
        orchestrator._strategy_b._agent.analyze = MagicMock(return_value=mock_response)
        orchestrator._strategy_b._agent._client = MagicMock()

        await orchestrator.run_daily(today=date(2026, 2, 5))

        # Check agent_decisions table
        from sqlalchemy import select
        from src.storage.models import AgentDecisionRow
        async with orchestrator._db.session() as s:
            result = await s.execute(select(AgentDecisionRow))
            decisions = result.scalars().all()
        assert len(decisions) == 1
        assert decisions[0].reasoning == "Testing persistence of agent decisions"
        assert decisions[0].tokens_used == 200


# ── Complexity-based model routing tests ────────────────────────────────────


class TestComplexityRouting:
    """Tests for complexity detection and model routing in the orchestrator."""

    def _setup_orchestrator(self, orchestrator: Orchestrator, tickers: list[str]) -> dict:
        """Common setup: mock market data, macro, and agent."""
        fake_data = _make_market_data(tickers)
        orchestrator._market_data.fetch_universe = AsyncMock(return_value=fake_data)
        orchestrator._macro_data.get_latest_yield_curve = MagicMock(return_value=1.0)
        orchestrator._macro_data.get_latest_vix = MagicMock(return_value=15.0)

        mock_response = {
            "regime_assessment": "Test",
            "reasoning": "Testing complexity routing",
            "trades": [],
            "risk_notes": "test",
            "_raw": "{}",
            "_tokens_used": 100,
            "_model": "claude-sonnet-4-5-20250929",
        }
        orchestrator._strategy_b._agent.analyze = MagicMock(return_value=mock_response)
        orchestrator._strategy_b._agent._client = MagicMock()
        return mock_response

    @patch("src.orchestrator.MacroDataFetcher")
    @patch("src.data.market_data.yf")
    async def test_low_complexity_uses_sonnet(
        self, mock_yf, mock_macro_cls, orchestrator: Orchestrator
    ) -> None:
        """Low complexity → no escalation, default Sonnet used."""
        tickers = ["XLK", "XLF", "QQQ", "GLD", "IBIT"]
        self._setup_orchestrator(orchestrator, tickers)

        # Set low VIX, no drawdown, same regime — low complexity
        orchestrator._macro_data.get_latest_vix = MagicMock(return_value=12.0)

        summary = await orchestrator.run_daily(today=date(2026, 2, 5))

        # Agent was called without model_override
        call_kwargs = orchestrator._strategy_b._agent.analyze.call_args
        assert call_kwargs[1].get("model_override") is None

    @patch("src.orchestrator.MacroDataFetcher")
    @patch("src.data.market_data.yf")
    async def test_high_complexity_no_telegram_defaults_sonnet(
        self, mock_yf, mock_macro_cls, orchestrator: Orchestrator
    ) -> None:
        """High complexity but no Telegram → defaults to Sonnet (no approval)."""
        tickers = ["XLK", "XLF", "QQQ", "GLD", "IBIT"]
        self._setup_orchestrator(orchestrator, tickers)

        # Force high complexity via detector mock
        orchestrator._complexity_detector.evaluate = MagicMock(
            return_value=ComplexityResult(score=70, should_escalate=True, signals=["VIX elevated at 35.0"])
        )
        # Ensure notifier is not configured
        orchestrator._notifier._token = ""
        orchestrator._notifier._chat_id = ""

        summary = await orchestrator.run_daily(today=date(2026, 2, 5))

        # Agent was called without model_override (no Telegram approval)
        call_kwargs = orchestrator._strategy_b._agent.analyze.call_args
        assert call_kwargs[1].get("model_override") is None

    @patch("src.orchestrator.MacroDataFetcher")
    @patch("src.data.market_data.yf")
    async def test_user_approves_opus(
        self, mock_yf, mock_macro_cls, orchestrator: Orchestrator
    ) -> None:
        """High complexity + Telegram approval = Opus used."""
        tickers = ["XLK", "XLF", "QQQ", "GLD", "IBIT"]
        mock_response = self._setup_orchestrator(orchestrator, tickers)
        mock_response["_model"] = "claude-opus-4-6"

        # Force high complexity via detector mock
        orchestrator._complexity_detector.evaluate = MagicMock(
            return_value=ComplexityResult(score=70, should_escalate=True, signals=["VIX elevated at 35.0"])
        )
        # Configure Telegram
        orchestrator._notifier._token = "test-token"
        orchestrator._notifier._chat_id = "12345"

        # Mock approval request + wait
        orchestrator._notifier.send_approval_request = AsyncMock(return_value=42)
        orchestrator._notifier.wait_for_approval = AsyncMock(return_value="opus")

        summary = await orchestrator.run_daily(today=date(2026, 2, 5))

        call_kwargs = orchestrator._strategy_b._agent.analyze.call_args
        assert call_kwargs[1].get("model_override") == "claude-opus-4-6"

    @patch("src.orchestrator.MacroDataFetcher")
    @patch("src.data.market_data.yf")
    async def test_user_chooses_skip(
        self, mock_yf, mock_macro_cls, orchestrator: Orchestrator
    ) -> None:
        """User chooses Skip → no trades from Portfolio B."""
        tickers = ["XLK", "XLF", "QQQ", "GLD", "IBIT"]
        self._setup_orchestrator(orchestrator, tickers)

        # Force high complexity via detector mock
        orchestrator._complexity_detector.evaluate = MagicMock(
            return_value=ComplexityResult(score=70, should_escalate=True, signals=["VIX elevated at 35.0"])
        )
        orchestrator._notifier._token = "test-token"
        orchestrator._notifier._chat_id = "12345"

        orchestrator._notifier.send_approval_request = AsyncMock(return_value=42)
        orchestrator._notifier.wait_for_approval = AsyncMock(return_value="skip")

        summary = await orchestrator.run_daily(today=date(2026, 2, 5))

        # Agent should NOT have been called
        orchestrator._strategy_b._agent.analyze.assert_not_called()
        assert summary["trades"]["B"] == 0

    @patch("src.orchestrator.MacroDataFetcher")
    @patch("src.data.market_data.yf")
    async def test_model_used_recorded_in_decision(
        self, mock_yf, mock_macro_cls, orchestrator: Orchestrator
    ) -> None:
        """Verify that model_used in AgentDecisionRow reflects the model actually used."""
        tickers = ["XLK", "XLF", "QQQ", "GLD", "IBIT"]
        mock_response = self._setup_orchestrator(orchestrator, tickers)
        mock_response["_model"] = "claude-opus-4-6"

        orchestrator._complexity_detector.evaluate = MagicMock(
            return_value=ComplexityResult(score=70, should_escalate=True, signals=["VIX elevated at 35.0"])
        )
        orchestrator._notifier._token = "test-token"
        orchestrator._notifier._chat_id = "12345"
        orchestrator._notifier.send_approval_request = AsyncMock(return_value=42)
        orchestrator._notifier.wait_for_approval = AsyncMock(return_value="opus")

        await orchestrator.run_daily(today=date(2026, 2, 5))

        from sqlalchemy import select
        from src.storage.models import AgentDecisionRow
        async with orchestrator._db.session() as s:
            result = await s.execute(select(AgentDecisionRow))
            decisions = result.scalars().all()
        assert len(decisions) == 1
        assert decisions[0].model_used == "claude-opus-4-6"

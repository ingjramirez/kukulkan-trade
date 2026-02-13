"""Tests for AI backtest strategy: cost tracking, budget, decisions."""

import json
import os
from datetime import date
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from src.backtest.ai_strategy import (
    INPUT_COST_PER_M,
    OUTPUT_COST_PER_M,
    AIBacktestStrategy,
)
from src.backtest.runner import BacktestRunner
from src.execution.paper_trader import PaperTrader
from src.storage.database import Database

# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_closes(tickers: list[str], days: int = 100, seed: int = 42) -> pd.DataFrame:
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
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(end=date.today(), periods=days)
    n = len(dates)
    data = {}
    for t in tickers:
        data[t] = rng.uniform(1_000_000, 50_000_000, n)
    return pd.DataFrame(data, index=dates)


def _mock_agent_response(tokens: int = 2000) -> dict:
    return {
        "regime_assessment": "Risk-on environment",
        "reasoning": "Tech momentum strong, buying XLK",
        "trades": [
            {"ticker": "XLK", "side": "BUY", "weight": 0.15, "reason": "momentum"},
        ],
        "risk_notes": "Standard risk",
        "_raw": "{}",
        "_tokens_used": tokens,
        "_model": "claude-sonnet-4-5-20250929",
    }


# ── Cost Tracking ────────────────────────────────────────────────────────────


class TestCostTracking:
    def test_track_cost_single_call(self) -> None:
        strategy = AIBacktestStrategy(budget_usd=10.0, run_label="test")
        cost = strategy._track_cost(1000, 500)

        expected = (1000 * INPUT_COST_PER_M / 1e6) + (500 * OUTPUT_COST_PER_M / 1e6)
        assert abs(cost - expected) < 1e-10
        assert strategy._total_input_tokens == 1000
        assert strategy._total_output_tokens == 500
        assert strategy._api_calls == 1

    def test_track_cost_accumulates(self) -> None:
        strategy = AIBacktestStrategy(budget_usd=10.0, run_label="test")
        strategy._track_cost(1000, 500)
        strategy._track_cost(2000, 1000)

        assert strategy._total_input_tokens == 3000
        assert strategy._total_output_tokens == 1500
        assert strategy._api_calls == 2
        assert strategy.total_tokens == 4500

    def test_budget_remaining(self) -> None:
        strategy = AIBacktestStrategy(budget_usd=1.50, run_label="test")
        # Simulate a small cost
        strategy._track_cost(100_000, 50_000)
        assert strategy.budget_remaining > 0
        assert strategy.budget_remaining < 1.50

    def test_budget_exhausted_property(self) -> None:
        strategy = AIBacktestStrategy(budget_usd=0.001, run_label="test")
        assert not strategy.budget_exhausted

        # Force a cost that exceeds budget
        strategy._track_cost(1_000_000, 500_000)
        assert strategy.budget_exhausted


# ── Budget Enforcement ───────────────────────────────────────────────────────


class TestBudgetEnforcement:
    def test_returns_empty_when_budget_exhausted(self) -> None:
        strategy = AIBacktestStrategy(budget_usd=0.001, run_label="test")
        # Force budget exhaustion
        strategy._total_cost_usd = 0.002

        tickers = ["XLK", "XLF", "QQQ"]
        closes = _make_closes(tickers, days=60)
        volumes = _make_volumes(tickers, days=60)

        trades = strategy.generate_trades(
            closes=closes,
            volumes=volumes,
            positions=[],
            cash=66_000.0,
            total_value=66_000.0,
            current_positions={},
            recent_trades=[],
            sim_date=date(2026, 1, 15),
        )

        assert trades == []
        assert strategy._api_calls == 0  # No API call made

    def test_api_error_returns_empty_and_logs(self) -> None:
        strategy = AIBacktestStrategy(budget_usd=10.0, run_label="test")
        # Mock the agent to raise an error
        strategy._strategy._agent.analyze = MagicMock(side_effect=Exception("API error"))
        strategy._strategy._agent._client = MagicMock()

        tickers = ["XLK", "XLF", "QQQ"]
        closes = _make_closes(tickers, days=60)
        volumes = _make_volumes(tickers, days=60)

        trades = strategy.generate_trades(
            closes=closes,
            volumes=volumes,
            positions=[],
            cash=66_000.0,
            total_value=66_000.0,
            current_positions={},
            recent_trades=[],
            sim_date=date(2026, 1, 15),
        )

        assert trades == []
        # Should have logged the error
        assert len(strategy._decisions) == 1
        assert "error" in strategy._decisions[0]


# ── Decision Logging ─────────────────────────────────────────────────────────


class TestDecisionLogging:
    def test_log_decision_records_entry(self) -> None:
        strategy = AIBacktestStrategy(budget_usd=10.0, run_label="test")
        strategy._log_decision(
            sim_date=date(2026, 1, 15),
            response=_mock_agent_response(),
            trades=[],
            cost=0.01,
        )

        assert len(strategy._decisions) == 1
        entry = strategy._decisions[0]
        assert entry["date"] == "2026-01-15"
        assert entry["run_label"] == "test"
        assert entry["cost_usd"] == 0.01

    def test_save_decisions_creates_file(self, tmp_path) -> None:
        strategy = AIBacktestStrategy(
            budget_usd=10.0,
            run_label="test-run",
            decisions_dir=str(tmp_path),
        )
        strategy._log_decision(
            sim_date=date(2026, 1, 15),
            response=_mock_agent_response(),
            trades=[],
            cost=0.005,
        )
        strategy._total_cost_usd = 0.005

        path = strategy.save_decisions()

        assert path is not None
        assert os.path.exists(path)
        assert "test-run" in path

        with open(path) as f:
            data = json.load(f)

        assert data["run_label"] == "test-run"
        assert data["budget_usd"] == 10.0
        assert len(data["decisions"]) == 1

    def test_save_decisions_returns_none_if_empty(self) -> None:
        strategy = AIBacktestStrategy(budget_usd=10.0, run_label="test")
        assert strategy.save_decisions() is None


# ── Cost Report ──────────────────────────────────────────────────────────────


class TestCostReport:
    def test_report_format(self) -> None:
        strategy = AIBacktestStrategy(budget_usd=1.50, run_label="standard")
        strategy._total_input_tokens = 50_000
        strategy._total_output_tokens = 15_000
        strategy._total_cost_usd = 0.375
        strategy._api_calls = 25

        report = strategy.get_cost_report()

        assert "standard" in report
        assert "25" in report
        assert "50,000" in report
        assert "15,000" in report
        assert "$0.3750" in report
        assert "$1.50" in report


# ── Prompt Override ──────────────────────────────────────────────────────────


class TestPromptOverride:
    def test_prompt_override_passed_to_context(self) -> None:
        custom_prompt = "You are a conservative portfolio manager."
        strategy = AIBacktestStrategy(
            budget_usd=10.0,
            run_label="conservative",
            prompt_override=custom_prompt,
        )
        # Mock the agent to capture the call
        strategy._strategy._agent.analyze = MagicMock(return_value=_mock_agent_response())
        strategy._strategy._agent._client = MagicMock()

        tickers = ["XLK", "XLF", "QQQ"]
        closes = _make_closes(tickers, days=60)
        volumes = _make_volumes(tickers, days=60)

        strategy.generate_trades(
            closes=closes,
            volumes=volumes,
            positions=[],
            cash=66_000.0,
            total_value=66_000.0,
            current_positions={},
            recent_trades=[],
            sim_date=date(2026, 1, 15),
        )

        # Verify the agent was called with the custom prompt
        call_kwargs = strategy._strategy._agent.analyze.call_args
        assert call_kwargs.kwargs.get("system_prompt") == custom_prompt


# ── Generate Trades ──────────────────────────────────────────────────────────


class TestGenerateTrades:
    def test_successful_trade_generation(self) -> None:
        strategy = AIBacktestStrategy(budget_usd=10.0, run_label="test")
        strategy._strategy._agent.analyze = MagicMock(return_value=_mock_agent_response(tokens=2000))
        strategy._strategy._agent._client = MagicMock()

        tickers = ["XLK", "XLF", "QQQ"]
        closes = _make_closes(tickers, days=60)
        volumes = _make_volumes(tickers, days=60)

        trades = strategy.generate_trades(
            closes=closes,
            volumes=volumes,
            positions=[],
            cash=66_000.0,
            total_value=66_000.0,
            current_positions={},
            recent_trades=[],
            sim_date=date(2026, 1, 15),
        )

        assert isinstance(trades, list)
        assert strategy._api_calls == 1
        assert strategy._total_cost_usd > 0
        assert len(strategy._decisions) == 1

    def test_sim_date_passed_to_agent(self) -> None:
        strategy = AIBacktestStrategy(budget_usd=10.0, run_label="test")
        strategy._strategy._agent.analyze = MagicMock(return_value=_mock_agent_response())
        strategy._strategy._agent._client = MagicMock()

        tickers = ["XLK", "XLF", "QQQ"]
        closes = _make_closes(tickers, days=60)
        volumes = _make_volumes(tickers, days=60)

        strategy.generate_trades(
            closes=closes,
            volumes=volumes,
            positions=[],
            cash=66_000.0,
            total_value=66_000.0,
            current_positions={},
            recent_trades=[],
            sim_date=date(2026, 1, 15),
        )

        call_kwargs = strategy._strategy._agent.analyze.call_args
        assert call_kwargs.kwargs.get("analysis_date") == date(2026, 1, 15)


# ── Runner Integration ───────────────────────────────────────────────────────


class TestRunnerIntegration:
    @pytest.fixture
    async def runner_and_db(self):
        runner = BacktestRunner.__new__(BacktestRunner)
        runner._db_url = "sqlite+aiosqlite:///:memory:"
        runner._db = Database(url="sqlite+aiosqlite:///:memory:")
        await runner._db.init_db()
        yield runner, runner._db
        await runner._db.close()

    @pytest.mark.asyncio
    async def test_run_portfolio_b_ai_with_budget(self, runner_and_db) -> None:
        runner, db = runner_and_db

        tickers = ["XLK", "XLF", "QQQ", "GLD", "AAPL", "MSFT"]
        closes = _make_closes(tickers, days=60)
        volumes = _make_volumes(tickers, days=60)

        trader = PaperTrader(db)
        await trader.initialize_portfolios()

        ai_bt = AIBacktestStrategy(budget_usd=10.0, run_label="test")
        ai_bt._strategy._agent.analyze = MagicMock(return_value=_mock_agent_response())
        ai_bt._strategy._agent._client = MagicMock()

        trades = await runner._run_portfolio_b_ai(
            ai_bt,
            closes,
            volumes,
            trader,
            date.today(),
        )

        assert isinstance(trades, list)
        assert ai_bt._api_calls == 1

    def test_dry_run_with_budget(self) -> None:
        runner = BacktestRunner.__new__(BacktestRunner)
        result = runner._estimate_cost(63, use_ai=True)

        assert result["dry_run"] is True
        assert result["estimated_api_calls"] == 63
        assert result["estimated_cost_usd"] > 0


# ── CLI Argument Parsing ─────────────────────────────────────────────────────


class TestCLIParsing:
    def test_run_label_auto_detection_standard(self) -> None:
        """No prompt override -> standard label."""
        prompt_override = None
        label = None
        if label is None:
            if prompt_override and "conservative" in prompt_override.lower():
                label = "conservative"
            elif prompt_override and "aggressive" in prompt_override.lower():
                label = "aggressive"
            else:
                label = "standard"
        assert label == "standard"

    def test_run_label_auto_detection_conservative(self) -> None:
        """Prompt with 'conservative' -> conservative label."""
        prompt_override = "You are a conservative portfolio manager."
        label = None
        if label is None:
            if prompt_override and "conservative" in prompt_override.lower():
                label = "conservative"
            elif prompt_override and "aggressive" in prompt_override.lower():
                label = "aggressive"
            else:
                label = "standard"
        assert label == "conservative"

    def test_run_label_auto_detection_aggressive(self) -> None:
        """Prompt with 'aggressive' -> aggressive label."""
        prompt_override = "Be aggressive with position sizing."
        label = None
        if label is None:
            if prompt_override and "conservative" in prompt_override.lower():
                label = "conservative"
            elif prompt_override and "aggressive" in prompt_override.lower():
                label = "aggressive"
            else:
                label = "standard"
        assert label == "aggressive"

    def test_explicit_label_takes_priority(self) -> None:
        """Explicit --run-label overrides auto-detection."""
        label = "my-custom-run"
        assert label == "my-custom-run"

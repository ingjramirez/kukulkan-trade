"""Tests for the agentic loop integration in the orchestrator.

Validates the agent loop branch (use_agent_loop=True), the single-shot fallback,
tool call log persistence, and accumulated action state merging.
"""

from datetime import date
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from cryptography.fernet import Fernet

from config.settings import settings
from src.agent.agent_runner import AgentRunResult, ToolCallLog
from src.agent.token_tracker import TokenTracker
from src.agent.tools.actions import ActionState
from src.storage.database import Database
from src.storage.models import (
    PortfolioRow,
    TenantRow,
)
from src.utils.crypto import encrypt_value

_TEST_KEY = Fernet.generate_key().decode()


@pytest.fixture(autouse=True)
def _set_encryption_key(monkeypatch):
    monkeypatch.setattr(settings, "tenant_encryption_key", _TEST_KEY)


@pytest.fixture
async def db():
    database = Database(url="sqlite+aiosqlite:///:memory:")
    await database.init_db()
    yield database
    await database.close()


def _make_tenant(tenant_id: str = "t-agent", use_agent_loop: bool = False) -> TenantRow:
    return TenantRow(
        id=tenant_id,
        name="AgentTest",
        alpaca_api_key_enc=encrypt_value("KEY"),
        alpaca_api_secret_enc=encrypt_value("SECRET"),
        telegram_bot_token_enc=encrypt_value("TOKEN"),
        telegram_chat_id_enc=encrypt_value("123"),
        strategy_mode="conservative",
        run_portfolio_a=False,
        run_portfolio_b=True,
        use_agent_loop=use_agent_loop,
    )


def _make_closes(tickers: list[str], days: int = 60) -> pd.DataFrame:
    np.random.seed(42)
    dates = pd.bdate_range(end="2026-02-13", periods=days)
    data = {}
    for t in tickers:
        data[t] = 100 + np.cumsum(np.random.normal(0.05, 1.0, days))
    return pd.DataFrame(data, index=dates)


def _mock_agent_response() -> dict:
    return {
        "regime_assessment": "Bull market",
        "reasoning": "Test reasoning",
        "trades": [{"ticker": "XLK", "side": "BUY", "weight": 0.15, "conviction": "high", "reason": "momentum"}],
        "risk_notes": "Acceptable risk",
        "memory_notes": [{"key": "test", "content": "observation"}],
        "watchlist_updates": [],
        "_raw": "{}",
        "_tokens_used": 500,
        "_model": "claude-sonnet-4-5-20250929",
    }


class TestDefaultTenantAgentLoop:
    """Fix #1: Default tenant with use_agent_loop=True should use agentic path."""

    async def test_default_tenant_with_agent_loop_enabled(self, db: Database) -> None:
        """Default tenant with use_agent_loop=True uses the agentic path."""
        # Create a "default" tenant row with use_agent_loop=True
        default_tenant = TenantRow(
            id="default",
            name="Default",
            alpaca_api_key_enc=encrypt_value("KEY"),
            alpaca_api_secret_enc=encrypt_value("SECRET"),
            telegram_bot_token_enc=encrypt_value("TOKEN"),
            telegram_chat_id_enc=encrypt_value("123"),
            strategy_mode="conservative",
            run_portfolio_a=True,
            run_portfolio_b=True,
            use_agent_loop=True,
        )
        await db.create_tenant(default_tenant)

        from src.orchestrator import Orchestrator

        orch = Orchestrator(db)
        async with db.session() as s:
            s.add(PortfolioRow(name="B", cash=66000.0, total_value=66000.0, tenant_id="default"))
            await s.commit()

        closes = _make_closes(["XLK", "SPY", "QQQ"])

        mock_result = AgentRunResult(
            response=_mock_agent_response(),
            tool_calls=[
                ToolCallLog(
                    turn=1,
                    tool_name="get_market_context",
                    tool_input={},
                    tool_output_preview="{}",
                    success=True,
                ),
            ],
            turns=2,
            token_tracker=TokenTracker(session_budget_usd=0.50),
        )
        mock_result.token_tracker.record("claude-sonnet-4-5-20250929", 500, 200, 1)

        with (
            patch("src.agent.agent_runner.AgentRunner") as mock_runner_cls,
            patch("src.agent.tools.portfolio.register_portfolio_tools"),
            patch("src.agent.tools.market.register_market_tools"),
            patch("src.agent.tools.news.register_news_tools"),
            patch("src.agent.tools.actions.register_action_tools"),
            patch("src.agent.tools.actions.ActionState", return_value=ActionState()),
            patch.object(orch._strategy_b, "agent_response_to_trades", return_value=[]),
            patch.object(orch._strategy_b, "save_decision", new_callable=AsyncMock),
            patch.object(
                orch,
                "_memory_manager",
                MagicMock(
                    save_short_term=AsyncMock(),
                    save_agent_notes=AsyncMock(),
                ),
            ),
            patch.object(orch, "_process_suggested_tickers", new_callable=AsyncMock),
            patch.object(orch, "_process_watchlist_updates", new_callable=AsyncMock),
            patch("src.orchestrator.ComplexityDetector") as mock_cx,
        ):
            mock_cx_instance = MagicMock()
            mock_cx_instance.evaluate.return_value = MagicMock(score=30, should_escalate=False)
            mock_cx.return_value = mock_cx_instance

            mock_runner = MagicMock()
            mock_runner.run = AsyncMock(return_value=mock_result)
            mock_runner.registry = MagicMock()
            mock_runner_cls.return_value = mock_runner

            trades, reasoning, tool_summary = await orch._run_portfolio_b(
                closes=closes,
                volumes=closes * 1e6,
                yield_curve=1.0,
                vix=15.0,
                today=date(2026, 2, 13),
                tenant_id="default",
            )

        # Agentic path was used
        assert tool_summary is not None
        assert tool_summary["tools_used"] == 1


class TestAgentLoopDisabled:
    """When use_agent_loop=False (default), the existing single-shot path is used."""

    async def test_default_tenant_uses_single_shot(self, db: Database) -> None:
        """Default tenant (no tenant row) uses the single-shot path."""
        from src.orchestrator import Orchestrator

        orch = Orchestrator(db)
        # Set up portfolio B
        async with db.session() as s:
            s.add(PortfolioRow(name="B", cash=66000.0, total_value=66000.0))
            await s.commit()

        closes = _make_closes(["XLK", "XLF", "QQQ", "SPY"])
        volumes = closes * 1e6

        mock_response = _mock_agent_response()

        with (
            patch.object(orch._strategy_b._agent, "analyze", return_value=mock_response),
            patch.object(orch._strategy_b, "agent_response_to_trades", return_value=[]),
            patch.object(orch._strategy_b, "save_decision", new_callable=AsyncMock),
            patch.object(
                orch,
                "_memory_manager",
                MagicMock(
                    save_short_term=AsyncMock(),
                    save_agent_notes=AsyncMock(),
                ),
            ),
            patch.object(orch, "_process_suggested_tickers", new_callable=AsyncMock),
            patch.object(orch, "_process_watchlist_updates", new_callable=AsyncMock),
            patch("src.orchestrator.ComplexityDetector") as mock_cx,
        ):
            mock_cx_instance = MagicMock()
            mock_cx_instance.evaluate.return_value = MagicMock(score=30, should_escalate=False)
            mock_cx.return_value = mock_cx_instance

            trades, reasoning, tool_summary = await orch._run_portfolio_b(
                closes=closes,
                volumes=volumes,
                yield_curve=1.0,
                vix=15.0,
                today=date(2026, 2, 13),
                news_context="Test news",
                session="Morning",
                tenant_id="default",
            )

        assert tool_summary is None  # Single-shot path
        assert reasoning == "Test reasoning"

    async def test_tenant_with_loop_disabled(self, db: Database) -> None:
        """Tenant with use_agent_loop=False uses single-shot."""
        tenant = _make_tenant(use_agent_loop=False)
        await db.create_tenant(tenant)

        from src.orchestrator import Orchestrator

        orch = Orchestrator(db)
        async with db.session() as s:
            s.add(PortfolioRow(name="B", cash=66000.0, total_value=66000.0, tenant_id=tenant.id))
            await s.commit()

        closes = _make_closes(["XLK", "SPY"])
        mock_response = _mock_agent_response()

        with (
            patch.object(orch._strategy_b._agent, "analyze", return_value=mock_response),
            patch.object(orch._strategy_b, "agent_response_to_trades", return_value=[]),
            patch.object(orch._strategy_b, "save_decision", new_callable=AsyncMock),
            patch.object(
                orch,
                "_memory_manager",
                MagicMock(
                    save_short_term=AsyncMock(),
                    save_agent_notes=AsyncMock(),
                ),
            ),
            patch.object(orch, "_process_suggested_tickers", new_callable=AsyncMock),
            patch.object(orch, "_process_watchlist_updates", new_callable=AsyncMock),
            patch("src.orchestrator.ComplexityDetector") as mock_cx,
        ):
            mock_cx_instance = MagicMock()
            mock_cx_instance.evaluate.return_value = MagicMock(score=30, should_escalate=False)
            mock_cx.return_value = mock_cx_instance

            trades, reasoning, tool_summary = await orch._run_portfolio_b(
                closes=closes,
                volumes=closes * 1e6,
                yield_curve=1.0,
                vix=15.0,
                today=date(2026, 2, 13),
                tenant_id=tenant.id,
            )

        assert tool_summary is None


class TestAgentLoopEnabled:
    """When use_agent_loop=True, the agentic loop path is used."""

    async def test_agentic_path_runs_agent_runner(self, db: Database) -> None:
        """Tenant with use_agent_loop=True uses AgentRunner."""
        tenant = _make_tenant(use_agent_loop=True)
        await db.create_tenant(tenant)

        from src.orchestrator import Orchestrator

        orch = Orchestrator(db)
        async with db.session() as s:
            s.add(PortfolioRow(name="B", cash=66000.0, total_value=66000.0, tenant_id=tenant.id))
            await s.commit()

        closes = _make_closes(["XLK", "SPY", "QQQ"])

        # Mock AgentRunner.run
        mock_result = AgentRunResult(
            response=_mock_agent_response(),
            tool_calls=[
                ToolCallLog(
                    turn=1,
                    tool_name="get_market_context",
                    tool_input={},
                    tool_output_preview="{}",
                    success=True,
                ),
                ToolCallLog(
                    turn=2,
                    tool_name="get_price_and_technicals",
                    tool_input={"ticker": "XLK"},
                    tool_output_preview="{}",
                    success=True,
                ),
            ],
            turns=3,
            token_tracker=TokenTracker(session_budget_usd=0.50),
        )
        mock_result.token_tracker.record("claude-sonnet-4-5-20250929", 500, 200, 1)

        with (
            patch("src.agent.agent_runner.AgentRunner") as mock_runner_cls,
            patch("src.agent.tools.portfolio.register_portfolio_tools"),
            patch("src.agent.tools.market.register_market_tools"),
            patch("src.agent.tools.news.register_news_tools"),
            patch("src.agent.tools.actions.register_action_tools"),
            patch("src.agent.tools.actions.ActionState", return_value=ActionState()),
            patch.object(orch._strategy_b, "agent_response_to_trades", return_value=[]),
            patch.object(orch._strategy_b, "save_decision", new_callable=AsyncMock),
            patch.object(
                orch,
                "_memory_manager",
                MagicMock(
                    save_short_term=AsyncMock(),
                    save_agent_notes=AsyncMock(),
                ),
            ),
            patch.object(orch, "_process_suggested_tickers", new_callable=AsyncMock),
            patch.object(orch, "_process_watchlist_updates", new_callable=AsyncMock),
            patch("src.orchestrator.ComplexityDetector") as mock_cx,
        ):
            mock_cx_instance = MagicMock()
            mock_cx_instance.evaluate.return_value = MagicMock(score=30, should_escalate=False)
            mock_cx.return_value = mock_cx_instance

            mock_runner = MagicMock()
            mock_runner.run = AsyncMock(return_value=mock_result)
            mock_runner.registry = MagicMock()
            mock_runner_cls.return_value = mock_runner

            trades, reasoning, tool_summary = await orch._run_portfolio_b(
                closes=closes,
                volumes=closes * 1e6,
                yield_curve=1.0,
                vix=15.0,
                today=date(2026, 2, 13),
                tenant_id=tenant.id,
            )

        assert tool_summary is not None
        assert tool_summary["tools_used"] == 2
        assert tool_summary["turns"] == 3
        assert tool_summary["cost_usd"] > 0
        assert reasoning == "Test reasoning"

    async def test_accumulated_actions_merged_into_response(self, db: Database) -> None:
        """Action tools' accumulated state is merged when response has no trades."""
        tenant = _make_tenant(use_agent_loop=True)
        await db.create_tenant(tenant)

        from src.orchestrator import Orchestrator

        orch = Orchestrator(db)
        async with db.session() as s:
            s.add(PortfolioRow(name="B", cash=66000.0, total_value=66000.0, tenant_id=tenant.id))
            await s.commit()

        closes = _make_closes(["XLK", "SPY"])

        # Response without trades (agent used propose_trades tool instead)
        response_no_trades = {
            "regime_assessment": "Bull",
            "reasoning": "Used tools for investigation",
            "trades": [],
            "risk_notes": "ok",
            "_tokens_used": 300,
            "_model": "test",
        }

        mock_result = AgentRunResult(
            response=response_no_trades,
            tool_calls=[],
            turns=2,
            token_tracker=TokenTracker(session_budget_usd=0.50),
        )

        # Create an ActionState with accumulated trades
        action_state = ActionState()
        action_state.proposed_trades.append(
            {"ticker": "XLK", "side": "BUY", "weight": 0.15, "conviction": "high", "reason": "tools said so"}
        )
        action_state.memory_notes.append({"key": "test-note", "content": "learned something"})

        with (
            patch("src.agent.agent_runner.AgentRunner") as mock_runner_cls,
            patch("src.agent.tools.portfolio.register_portfolio_tools"),
            patch("src.agent.tools.market.register_market_tools"),
            patch("src.agent.tools.news.register_news_tools"),
            patch("src.agent.tools.actions.register_action_tools"),
            patch("src.agent.tools.actions.ActionState", return_value=action_state),
            patch.object(orch._strategy_b, "agent_response_to_trades", return_value=[]) as mock_a2t,
            patch.object(orch._strategy_b, "save_decision", new_callable=AsyncMock),
            patch.object(
                orch,
                "_memory_manager",
                MagicMock(
                    save_short_term=AsyncMock(),
                    save_agent_notes=AsyncMock(),
                ),
            ),
            patch.object(orch, "_process_suggested_tickers", new_callable=AsyncMock),
            patch.object(orch, "_process_watchlist_updates", new_callable=AsyncMock),
            patch("src.orchestrator.ComplexityDetector") as mock_cx,
        ):
            mock_cx_instance = MagicMock()
            mock_cx_instance.evaluate.return_value = MagicMock(score=30, should_escalate=False)
            mock_cx.return_value = mock_cx_instance

            mock_runner = MagicMock()
            mock_runner.run = AsyncMock(return_value=mock_result)
            mock_runner.registry = MagicMock()
            mock_runner_cls.return_value = mock_runner

            trades, reasoning, tool_summary = await orch._run_portfolio_b(
                closes=closes,
                volumes=closes * 1e6,
                yield_curve=1.0,
                vix=15.0,
                today=date(2026, 2, 13),
                tenant_id=tenant.id,
            )

        # Verify accumulated trades were merged
        call_args = mock_a2t.call_args
        response_passed = call_args.kwargs.get("response") or call_args[1].get("response")
        assert len(response_passed["trades"]) == 1
        assert response_passed["trades"][0]["ticker"] == "XLK"
        assert len(response_passed["memory_notes"]) == 1

    async def test_tool_call_logs_saved(self, db: Database) -> None:
        """Tool call logs are persisted to the database."""
        tenant = _make_tenant(use_agent_loop=True)
        await db.create_tenant(tenant)

        from src.orchestrator import Orchestrator

        orch = Orchestrator(db)
        async with db.session() as s:
            s.add(PortfolioRow(name="B", cash=66000.0, total_value=66000.0, tenant_id=tenant.id))
            await s.commit()

        closes = _make_closes(["XLK", "SPY"])
        today = date(2026, 2, 13)

        mock_result = AgentRunResult(
            response=_mock_agent_response(),
            tool_calls=[
                ToolCallLog(
                    turn=1,
                    tool_name="get_market_context",
                    tool_input={},
                    tool_output_preview='{"regime":"BULL"}',
                    success=True,
                ),
                ToolCallLog(
                    turn=1,
                    tool_name="get_current_positions",
                    tool_input={},
                    tool_output_preview="[]",
                    success=True,
                ),
                ToolCallLog(
                    turn=2,
                    tool_name="propose_trades",
                    tool_input={"trades": []},
                    tool_output_preview='{"status":"ok"}',
                    success=True,
                ),
            ],
            turns=2,
            token_tracker=TokenTracker(session_budget_usd=0.50),
        )

        with (
            patch("src.agent.agent_runner.AgentRunner") as mock_runner_cls,
            patch("src.agent.tools.portfolio.register_portfolio_tools"),
            patch("src.agent.tools.market.register_market_tools"),
            patch("src.agent.tools.news.register_news_tools"),
            patch("src.agent.tools.actions.register_action_tools"),
            patch("src.agent.tools.actions.ActionState", return_value=ActionState()),
            patch.object(orch._strategy_b, "agent_response_to_trades", return_value=[]),
            patch.object(orch._strategy_b, "save_decision", new_callable=AsyncMock),
            patch.object(
                orch,
                "_memory_manager",
                MagicMock(
                    save_short_term=AsyncMock(),
                    save_agent_notes=AsyncMock(),
                ),
            ),
            patch.object(orch, "_process_suggested_tickers", new_callable=AsyncMock),
            patch.object(orch, "_process_watchlist_updates", new_callable=AsyncMock),
            patch("src.orchestrator.ComplexityDetector") as mock_cx,
        ):
            mock_cx_instance = MagicMock()
            mock_cx_instance.evaluate.return_value = MagicMock(score=30, should_escalate=False)
            mock_cx.return_value = mock_cx_instance

            mock_runner = MagicMock()
            mock_runner.run = AsyncMock(return_value=mock_result)
            mock_runner.registry = MagicMock()
            mock_runner_cls.return_value = mock_runner

            await orch._run_portfolio_b(
                closes=closes,
                volumes=closes * 1e6,
                yield_curve=1.0,
                vix=15.0,
                today=today,
                session="Morning",
                tenant_id=tenant.id,
            )

        # Verify logs were saved
        logs = await db.get_tool_call_logs(tenant_id=tenant.id, session_date=today)
        assert len(logs) == 3
        assert logs[0].tool_name in ("get_market_context", "get_current_positions", "propose_trades")
        assert all(log.session_label == "Morning" for log in logs)


class TestTwoPhaseFlow:
    """Fix #6: Two-phase seed→investigate flow."""

    async def test_two_phase_seed_then_investigate(self, db: Database) -> None:
        """Agentic mode calls analyze() first (seed), then AgentRunner.run() (investigate)."""
        tenant = _make_tenant(use_agent_loop=True)
        await db.create_tenant(tenant)

        from src.orchestrator import Orchestrator

        orch = Orchestrator(db)
        async with db.session() as s:
            s.add(PortfolioRow(name="B", cash=66000.0, total_value=66000.0, tenant_id=tenant.id))
            await s.commit()

        closes = _make_closes(["XLK", "SPY", "QQQ"])
        seed_response = _mock_agent_response()

        mock_result = AgentRunResult(
            response={"reasoning": "investigation confirmed", "trades": []},
            tool_calls=[
                ToolCallLog(
                    turn=1, tool_name="get_market_context", tool_input={}, tool_output_preview="{}", success=True
                ),
            ],
            turns=2,
            token_tracker=TokenTracker(session_budget_usd=0.50),
        )

        with (
            patch.object(orch._strategy_b._agent, "analyze", return_value=seed_response) as mock_analyze,
            patch("src.agent.agent_runner.AgentRunner") as mock_runner_cls,
            patch("src.agent.tools.portfolio.register_portfolio_tools"),
            patch("src.agent.tools.market.register_market_tools"),
            patch("src.agent.tools.news.register_news_tools"),
            patch("src.agent.tools.actions.register_action_tools"),
            patch("src.agent.tools.actions.ActionState", return_value=ActionState()),
            patch.object(orch._strategy_b, "agent_response_to_trades", return_value=[]),
            patch.object(orch._strategy_b, "save_decision", new_callable=AsyncMock),
            patch.object(
                orch,
                "_memory_manager",
                MagicMock(save_short_term=AsyncMock(), save_agent_notes=AsyncMock()),
            ),
            patch.object(orch, "_process_suggested_tickers", new_callable=AsyncMock),
            patch.object(orch, "_process_watchlist_updates", new_callable=AsyncMock),
            patch("src.orchestrator.ComplexityDetector") as mock_cx,
        ):
            mock_cx_instance = MagicMock()
            mock_cx_instance.evaluate.return_value = MagicMock(score=30, should_escalate=False)
            mock_cx.return_value = mock_cx_instance

            mock_runner = MagicMock()
            mock_runner.run = AsyncMock(return_value=mock_result)
            mock_runner.registry = MagicMock()
            mock_runner_cls.return_value = mock_runner

            trades, reasoning, tool_summary = await orch._run_portfolio_b(
                closes=closes,
                volumes=closes * 1e6,
                yield_curve=1.0,
                vix=15.0,
                today=date(2026, 2, 13),
                tenant_id=tenant.id,
            )

        # Phase 1: analyze() was called
        mock_analyze.assert_called_once()
        # Phase 2: AgentRunner.run() was called with investigation message
        mock_runner.run.assert_called_once()
        call_kwargs = mock_runner.run.call_args.kwargs
        assert "Initial Analysis" in call_kwargs["user_message"]
        assert "propose_trades" in call_kwargs["system_prompt"]
        assert tool_summary is not None

    async def test_investigation_overrides_seed_trades(self, db: Database) -> None:
        """ActionState trades from investigation override seed trades."""
        tenant = _make_tenant(use_agent_loop=True)
        await db.create_tenant(tenant)

        from src.orchestrator import Orchestrator

        orch = Orchestrator(db)
        async with db.session() as s:
            s.add(PortfolioRow(name="B", cash=66000.0, total_value=66000.0, tenant_id=tenant.id))
            await s.commit()

        closes = _make_closes(["XLK", "SPY", "QQQ"])

        seed_response = _mock_agent_response()  # Has XLK trade
        mock_result = AgentRunResult(
            response={"reasoning": "refined", "trades": []},
            tool_calls=[],
            turns=1,
            token_tracker=TokenTracker(session_budget_usd=0.50),
        )

        # ActionState with QQQ trade (overrides seed's XLK)
        action_state = ActionState()
        action_state.proposed_trades.append(
            {"ticker": "QQQ", "side": "BUY", "weight": 0.20, "conviction": "high", "reason": "investigation found QQQ"}
        )

        with (
            patch.object(orch._strategy_b._agent, "analyze", return_value=seed_response),
            patch("src.agent.agent_runner.AgentRunner") as mock_runner_cls,
            patch("src.agent.tools.portfolio.register_portfolio_tools"),
            patch("src.agent.tools.market.register_market_tools"),
            patch("src.agent.tools.news.register_news_tools"),
            patch("src.agent.tools.actions.register_action_tools"),
            patch("src.agent.tools.actions.ActionState", return_value=action_state),
            patch.object(orch._strategy_b, "agent_response_to_trades", return_value=[]) as mock_a2t,
            patch.object(orch._strategy_b, "save_decision", new_callable=AsyncMock),
            patch.object(
                orch,
                "_memory_manager",
                MagicMock(save_short_term=AsyncMock(), save_agent_notes=AsyncMock()),
            ),
            patch.object(orch, "_process_suggested_tickers", new_callable=AsyncMock),
            patch.object(orch, "_process_watchlist_updates", new_callable=AsyncMock),
            patch("src.orchestrator.ComplexityDetector") as mock_cx,
        ):
            mock_cx_instance = MagicMock()
            mock_cx_instance.evaluate.return_value = MagicMock(score=30, should_escalate=False)
            mock_cx.return_value = mock_cx_instance

            mock_runner = MagicMock()
            mock_runner.run = AsyncMock(return_value=mock_result)
            mock_runner.registry = MagicMock()
            mock_runner_cls.return_value = mock_runner

            await orch._run_portfolio_b(
                closes=closes,
                volumes=closes * 1e6,
                yield_curve=1.0,
                vix=15.0,
                today=date(2026, 2, 13),
                tenant_id=tenant.id,
            )

        # Investigation trades (QQQ) override seed trades (XLK)
        call_args = mock_a2t.call_args
        response_passed = call_args.kwargs.get("response") or call_args[1].get("response")
        assert len(response_passed["trades"]) == 1
        assert response_passed["trades"][0]["ticker"] == "QQQ"

    async def test_investigation_fallback_to_seed(self, db: Database) -> None:
        """When investigation produces no trades, seed trades are used."""
        tenant = _make_tenant(use_agent_loop=True)
        await db.create_tenant(tenant)

        from src.orchestrator import Orchestrator

        orch = Orchestrator(db)
        async with db.session() as s:
            s.add(PortfolioRow(name="B", cash=66000.0, total_value=66000.0, tenant_id=tenant.id))
            await s.commit()

        closes = _make_closes(["XLK", "SPY"])

        seed_response = _mock_agent_response()  # Has XLK trade
        mock_result = AgentRunResult(
            response={"reasoning": "confirmed seed", "trades": []},
            tool_calls=[],
            turns=1,
            token_tracker=TokenTracker(session_budget_usd=0.50),
        )

        # Empty ActionState (no investigation trades)
        action_state = ActionState()

        with (
            patch.object(orch._strategy_b._agent, "analyze", return_value=seed_response),
            patch("src.agent.agent_runner.AgentRunner") as mock_runner_cls,
            patch("src.agent.tools.portfolio.register_portfolio_tools"),
            patch("src.agent.tools.market.register_market_tools"),
            patch("src.agent.tools.news.register_news_tools"),
            patch("src.agent.tools.actions.register_action_tools"),
            patch("src.agent.tools.actions.ActionState", return_value=action_state),
            patch.object(orch._strategy_b, "agent_response_to_trades", return_value=[]) as mock_a2t,
            patch.object(orch._strategy_b, "save_decision", new_callable=AsyncMock),
            patch.object(
                orch,
                "_memory_manager",
                MagicMock(save_short_term=AsyncMock(), save_agent_notes=AsyncMock()),
            ),
            patch.object(orch, "_process_suggested_tickers", new_callable=AsyncMock),
            patch.object(orch, "_process_watchlist_updates", new_callable=AsyncMock),
            patch("src.orchestrator.ComplexityDetector") as mock_cx,
        ):
            mock_cx_instance = MagicMock()
            mock_cx_instance.evaluate.return_value = MagicMock(score=30, should_escalate=False)
            mock_cx.return_value = mock_cx_instance

            mock_runner = MagicMock()
            mock_runner.run = AsyncMock(return_value=mock_result)
            mock_runner.registry = MagicMock()
            mock_runner_cls.return_value = mock_runner

            await orch._run_portfolio_b(
                closes=closes,
                volumes=closes * 1e6,
                yield_curve=1.0,
                vix=15.0,
                today=date(2026, 2, 13),
                tenant_id=tenant.id,
            )

        # Seed trades (XLK) used as fallback
        call_args = mock_a2t.call_args
        response_passed = call_args.kwargs.get("response") or call_args[1].get("response")
        assert len(response_passed["trades"]) == 1
        assert response_passed["trades"][0]["ticker"] == "XLK"


class TestToolCallLogsPersistence:
    """Test the database CRUD for tool call logs."""

    async def test_save_and_retrieve_logs(self, db: Database) -> None:
        today = date(2026, 2, 13)
        logs = [
            {
                "turn": 1,
                "tool_name": "get_market_context",
                "tool_input": {},
                "tool_output_preview": "{}",
                "success": True,
                "error": None,
            },
            {
                "turn": 2,
                "tool_name": "propose_trades",
                "tool_input": {"trades": []},
                "tool_output_preview": "{}",
                "success": True,
                "error": None,
            },
        ]
        await db.save_tool_call_logs(logs, today, session_label="Morning", tenant_id="default")

        rows = await db.get_tool_call_logs(tenant_id="default", session_date=today)
        assert len(rows) == 2

    async def test_empty_logs_no_op(self, db: Database) -> None:
        await db.save_tool_call_logs([], date(2026, 2, 13))
        rows = await db.get_tool_call_logs()
        assert len(rows) == 0

    async def test_filter_by_date(self, db: Database) -> None:
        await db.save_tool_call_logs(
            [
                {
                    "turn": 1,
                    "tool_name": "t1",
                    "tool_input": {},
                    "tool_output_preview": "",
                    "success": True,
                    "error": None,
                }
            ],
            date(2026, 2, 12),
        )
        await db.save_tool_call_logs(
            [
                {
                    "turn": 1,
                    "tool_name": "t2",
                    "tool_input": {},
                    "tool_output_preview": "",
                    "success": True,
                    "error": None,
                }
            ],
            date(2026, 2, 13),
        )

        rows_12 = await db.get_tool_call_logs(session_date=date(2026, 2, 12))
        rows_13 = await db.get_tool_call_logs(session_date=date(2026, 2, 13))
        assert len(rows_12) == 1
        assert rows_12[0].tool_name == "t1"
        assert len(rows_13) == 1
        assert rows_13[0].tool_name == "t2"


class TestInfluencedDecision:
    """Fix #7: influenced_decision tracking in tool call logs."""

    async def test_action_tools_always_influential(self, db: Database) -> None:
        """Always-influential tools (propose_trades, get_market_context, etc.) are marked."""
        tenant = _make_tenant(use_agent_loop=True)
        await db.create_tenant(tenant)

        from src.orchestrator import Orchestrator
        from src.storage.models import TradeSchema

        orch = Orchestrator(db)
        async with db.session() as s:
            s.add(PortfolioRow(name="B", cash=66000.0, total_value=66000.0, tenant_id=tenant.id))
            await s.commit()

        closes = _make_closes(["XLK", "SPY"])
        today = date(2026, 2, 13)

        mock_result = AgentRunResult(
            response=_mock_agent_response(),
            tool_calls=[
                ToolCallLog(
                    turn=1, tool_name="get_market_context", tool_input={}, tool_output_preview="{}", success=True
                ),
                ToolCallLog(
                    turn=1, tool_name="get_current_positions", tool_input={}, tool_output_preview="[]", success=True
                ),
                ToolCallLog(
                    turn=2,
                    tool_name="propose_trades",
                    tool_input={"trades": []},
                    tool_output_preview="{}",
                    success=True,
                ),
            ],
            turns=2,
            token_tracker=TokenTracker(session_budget_usd=0.50),
        )

        # Return actual TradeSchema objects so traded_tickers is non-empty
        mock_trades = [
            TradeSchema(portfolio="B", ticker="XLK", side="BUY", shares=10, price=150.0, reason="test"),
        ]

        with (
            patch.object(orch._strategy_b._agent, "analyze", return_value=_mock_agent_response()),
            patch("src.agent.agent_runner.AgentRunner") as mock_runner_cls,
            patch("src.agent.tools.portfolio.register_portfolio_tools"),
            patch("src.agent.tools.market.register_market_tools"),
            patch("src.agent.tools.news.register_news_tools"),
            patch("src.agent.tools.actions.register_action_tools"),
            patch("src.agent.tools.actions.ActionState", return_value=ActionState()),
            patch.object(orch._strategy_b, "agent_response_to_trades", return_value=mock_trades),
            patch.object(orch._strategy_b, "save_decision", new_callable=AsyncMock),
            patch.object(
                orch,
                "_memory_manager",
                MagicMock(save_short_term=AsyncMock(), save_agent_notes=AsyncMock()),
            ),
            patch.object(orch, "_process_suggested_tickers", new_callable=AsyncMock),
            patch.object(orch, "_process_watchlist_updates", new_callable=AsyncMock),
            patch("src.orchestrator.ComplexityDetector") as mock_cx,
        ):
            mock_cx_instance = MagicMock()
            mock_cx_instance.evaluate.return_value = MagicMock(score=30, should_escalate=False)
            mock_cx.return_value = mock_cx_instance

            mock_runner = MagicMock()
            mock_runner.run = AsyncMock(return_value=mock_result)
            mock_runner.registry = MagicMock()
            mock_runner_cls.return_value = mock_runner

            await orch._run_portfolio_b(
                closes=closes,
                volumes=closes * 1e6,
                yield_curve=1.0,
                vix=15.0,
                today=today,
                session="Morning",
                tenant_id=tenant.id,
            )

        logs = await db.get_tool_call_logs(tenant_id=tenant.id, session_date=today)
        assert len(logs) == 3
        # All three are always-influential tools
        for log_row in logs:
            assert log_row.influenced_decision is True, f"{log_row.tool_name} should be influential"

    async def test_ticker_match_marks_influential(self, db: Database) -> None:
        """Tool calls whose input mentions a traded ticker are marked influential."""
        tenant = _make_tenant(use_agent_loop=True)
        await db.create_tenant(tenant)

        from src.orchestrator import Orchestrator
        from src.storage.models import TradeSchema

        orch = Orchestrator(db)
        async with db.session() as s:
            s.add(PortfolioRow(name="B", cash=66000.0, total_value=66000.0, tenant_id=tenant.id))
            await s.commit()

        closes = _make_closes(["XLK", "AAPL", "SPY"])
        today = date(2026, 2, 13)

        mock_result = AgentRunResult(
            response=_mock_agent_response(),
            tool_calls=[
                ToolCallLog(
                    turn=1,
                    tool_name="get_ticker_data",
                    tool_input={"ticker": "XLK"},
                    tool_output_preview="{}",
                    success=True,
                ),
                ToolCallLog(
                    turn=1,
                    tool_name="get_ticker_data",
                    tool_input={"ticker": "AAPL"},
                    tool_output_preview="{}",
                    success=True,
                ),
            ],
            turns=1,
            token_tracker=TokenTracker(session_budget_usd=0.50),
        )

        mock_trades = [
            TradeSchema(portfolio="B", ticker="XLK", side="BUY", shares=10, price=150.0, reason="test"),
        ]

        with (
            patch.object(orch._strategy_b._agent, "analyze", return_value=_mock_agent_response()),
            patch("src.agent.agent_runner.AgentRunner") as mock_runner_cls,
            patch("src.agent.tools.portfolio.register_portfolio_tools"),
            patch("src.agent.tools.market.register_market_tools"),
            patch("src.agent.tools.news.register_news_tools"),
            patch("src.agent.tools.actions.register_action_tools"),
            patch("src.agent.tools.actions.ActionState", return_value=ActionState()),
            patch.object(orch._strategy_b, "agent_response_to_trades", return_value=mock_trades),
            patch.object(orch._strategy_b, "save_decision", new_callable=AsyncMock),
            patch.object(
                orch,
                "_memory_manager",
                MagicMock(save_short_term=AsyncMock(), save_agent_notes=AsyncMock()),
            ),
            patch.object(orch, "_process_suggested_tickers", new_callable=AsyncMock),
            patch.object(orch, "_process_watchlist_updates", new_callable=AsyncMock),
            patch("src.orchestrator.ComplexityDetector") as mock_cx,
        ):
            mock_cx_instance = MagicMock()
            mock_cx_instance.evaluate.return_value = MagicMock(score=30, should_escalate=False)
            mock_cx.return_value = mock_cx_instance

            mock_runner = MagicMock()
            mock_runner.run = AsyncMock(return_value=mock_result)
            mock_runner.registry = MagicMock()
            mock_runner_cls.return_value = mock_runner

            await orch._run_portfolio_b(
                closes=closes,
                volumes=closes * 1e6,
                yield_curve=1.0,
                vix=15.0,
                today=today,
                session="Morning",
                tenant_id=tenant.id,
            )

        logs = await db.get_tool_call_logs(tenant_id=tenant.id, session_date=today)
        assert len(logs) == 2
        # XLK was traded — tool call mentioning XLK is influential
        xlk_log = next(r for r in logs if "XLK" in (r.tool_input or ""))
        assert xlk_log.influenced_decision is True
        # AAPL was not traded — tool call mentioning AAPL is NOT influential
        aapl_log = next(r for r in logs if "AAPL" in (r.tool_input or ""))
        assert aapl_log.influenced_decision is False

    async def test_unrelated_tools_not_influential(self, db: Database) -> None:
        """Tool calls unrelated to traded tickers and not always-influential are not marked."""
        tenant = _make_tenant(use_agent_loop=True)
        await db.create_tenant(tenant)

        from src.orchestrator import Orchestrator
        from src.storage.models import TradeSchema

        orch = Orchestrator(db)
        async with db.session() as s:
            s.add(PortfolioRow(name="B", cash=66000.0, total_value=66000.0, tenant_id=tenant.id))
            await s.commit()

        closes = _make_closes(["XLK", "SPY"])
        today = date(2026, 2, 13)

        mock_result = AgentRunResult(
            response=_mock_agent_response(),
            tool_calls=[
                ToolCallLog(
                    turn=1,
                    tool_name="get_news_context",
                    tool_input={"query": "general market outlook"},
                    tool_output_preview="{}",
                    success=True,
                ),
            ],
            turns=1,
            token_tracker=TokenTracker(session_budget_usd=0.50),
        )

        # No trades produced
        mock_trades: list[TradeSchema] = []

        with (
            patch.object(orch._strategy_b._agent, "analyze", return_value=_mock_agent_response()),
            patch("src.agent.agent_runner.AgentRunner") as mock_runner_cls,
            patch("src.agent.tools.portfolio.register_portfolio_tools"),
            patch("src.agent.tools.market.register_market_tools"),
            patch("src.agent.tools.news.register_news_tools"),
            patch("src.agent.tools.actions.register_action_tools"),
            patch("src.agent.tools.actions.ActionState", return_value=ActionState()),
            patch.object(orch._strategy_b, "agent_response_to_trades", return_value=mock_trades),
            patch.object(orch._strategy_b, "save_decision", new_callable=AsyncMock),
            patch.object(
                orch,
                "_memory_manager",
                MagicMock(save_short_term=AsyncMock(), save_agent_notes=AsyncMock()),
            ),
            patch.object(orch, "_process_suggested_tickers", new_callable=AsyncMock),
            patch.object(orch, "_process_watchlist_updates", new_callable=AsyncMock),
            patch("src.orchestrator.ComplexityDetector") as mock_cx,
        ):
            mock_cx_instance = MagicMock()
            mock_cx_instance.evaluate.return_value = MagicMock(score=30, should_escalate=False)
            mock_cx.return_value = mock_cx_instance

            mock_runner = MagicMock()
            mock_runner.run = AsyncMock(return_value=mock_result)
            mock_runner.registry = MagicMock()
            mock_runner_cls.return_value = mock_runner

            await orch._run_portfolio_b(
                closes=closes,
                volumes=closes * 1e6,
                yield_curve=1.0,
                vix=15.0,
                today=today,
                session="Morning",
                tenant_id=tenant.id,
            )

        logs = await db.get_tool_call_logs(tenant_id=tenant.id, session_date=today)
        assert len(logs) == 1
        assert logs[0].influenced_decision is False

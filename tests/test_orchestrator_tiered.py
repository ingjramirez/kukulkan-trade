"""Tests for orchestrator tiered model integration.

Tests the tiered runner wiring in _run_portfolio_b_persistent and the
budget-exhausted skip path.
"""

from datetime import date
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.agent.persistent_agent import PersistentRunResult
from src.agent.token_tracker import TokenTracker
from src.orchestrator import Orchestrator
from src.storage.database import Database
from src.storage.models import TenantRow


def _make_closes(tickers: list[str], days: int = 250) -> pd.DataFrame:
    np.random.seed(42)
    dates = pd.bdate_range(end="2026-02-05", periods=days)
    data = {}
    for i, t in enumerate(tickers):
        drift = 0.2 - i * 0.01
        data[t] = 100 + np.cumsum(np.random.normal(drift, 1.5, days))
    return pd.DataFrame(data, index=dates)


TICKERS = ["SPY", "NVDA", "AAPL", "XLK", "GLD"]


@pytest.fixture
async def db():
    database = Database(url="sqlite+aiosqlite:///:memory:")
    await database.init_db()
    yield database
    await database.close()


@pytest.fixture
async def orchestrator(db: Database):
    return Orchestrator(db)


def _make_tenant(
    use_persistent_agent: bool = True,
    use_tiered_models: bool = False,
) -> TenantRow:
    return TenantRow(
        id="t1",
        name="Test Tenant",
        dashboard_user="test",
        dashboard_password_enc="$2b$12$hash",
        run_portfolio_a=True,
        run_portfolio_b=True,
        use_persistent_agent=use_persistent_agent,
        use_tiered_models=use_tiered_models,
    )


def _persistent_result() -> PersistentRunResult:
    return PersistentRunResult(
        response={
            "regime_assessment": "BULL",
            "reasoning": "test",
            "trades": [],
            "risk_notes": "",
        },
        session_id="test-session",
        tool_calls=[],
        turns=2,
        token_tracker=TokenTracker(),
    )


class TestTieredBackwardCompat:
    """Verify enable_tiered=False produces unchanged behavior."""

    @pytest.mark.asyncio
    async def test_tiered_disabled_uses_normal_path(self, orchestrator, db):
        """When enable_tiered=False, no tiered runner is created."""
        tenant = _make_tenant(use_persistent_agent=True, use_tiered_models=False)
        closes = _make_closes(TICKERS)

        with (
            patch.object(orchestrator._db, "get_tenant", new_callable=AsyncMock, return_value=tenant),
            patch.object(orchestrator._db, "get_current_posture", new_callable=AsyncMock, return_value=None),
            patch.object(orchestrator._db, "get_latest_playbook", new_callable=AsyncMock, return_value=[]),
            patch.object(orchestrator._db, "get_latest_calibration", new_callable=AsyncMock, return_value=[]),
            patch("src.agent.agent_runner.AgentRunner") as mock_runner_cls,
            patch("src.agent.persistent_agent.PersistentAgent") as mock_persistent_cls,
            patch("src.agent.tools.portfolio.register_portfolio_tools"),
            patch("src.agent.tools.market.register_market_tools"),
            patch("src.agent.tools.news.register_news_tools"),
            patch("src.agent.tools.actions.register_action_tools"),
            patch("src.agent.tools.actions.ActionState"),
            patch("src.orchestrator.settings") as mock_settings,
        ):
            mock_settings.anthropic_api_key = "test-key"
            mock_settings.agent.agent_tool_model = "claude-sonnet-4-5-20250929"
            mock_settings.agent.agent_max_turns = 8
            mock_settings.agent.agent_session_budget = 0.50
            mock_settings.agent.enable_tiered = False
            mock_settings.agent.enable_cache = False

            mock_persistent_inst = MagicMock()
            mock_persistent_inst.run_session = AsyncMock(return_value=_persistent_result())
            mock_persistent_cls.return_value = mock_persistent_inst

            mock_runner_inst = MagicMock()
            mock_runner_inst.registry = MagicMock()
            mock_runner_inst._token_tracker = TokenTracker()
            mock_runner_cls.return_value = mock_runner_inst

            trades, reasoning, tool_summary = await orchestrator._run_portfolio_b_persistent(
                tenant_id="t1",
                trigger_type="morning",
                dynamic_prompt="test prompt",
                closes=closes,
                volumes=closes,
                vix=15.0,
                yield_curve=None,
                regime_str="BULL",
                news_context="",
                positions_for_agent=[],
                cash=50000.0,
                total_value=66000.0,
                recent_trades=[],
                model_override=None,
                portfolio_b_universe=TICKERS,
                allocations=None,
                today=date(2026, 2, 15),
                session="Morning",
                strategy_mode="conservative",
            )

            # Should have used normal path — check runner_kwargs passed to persistent
            call_kwargs = mock_persistent_inst.run_session.call_args
            runner_kwargs = call_kwargs.kwargs.get("runner_kwargs") or call_kwargs[1].get("runner_kwargs", {})
            assert "tiered_runner" not in runner_kwargs


class TestTieredEnabled:
    """Tests for tiered mode wiring."""

    @pytest.mark.asyncio
    async def test_daily_budget_exhausted_skips_session(self, orchestrator, db):
        """When daily budget is exhausted, return empty trades."""
        tenant = _make_tenant(use_persistent_agent=True, use_tiered_models=True)

        with (
            patch.object(orchestrator._db, "get_tenant", new_callable=AsyncMock, return_value=tenant),
            patch.object(orchestrator._db, "get_current_posture", new_callable=AsyncMock, return_value=None),
            patch.object(orchestrator._db, "get_latest_playbook", new_callable=AsyncMock, return_value=[]),
            patch.object(orchestrator._db, "get_latest_calibration", new_callable=AsyncMock, return_value=[]),
            patch.object(orchestrator._db, "get_daily_spend", new_callable=AsyncMock, return_value=5.0),
            patch.object(orchestrator._db, "get_monthly_spend", new_callable=AsyncMock, return_value=10.0),
            patch("src.agent.agent_runner.AgentRunner") as mock_runner_cls,
            patch("src.agent.tools.portfolio.register_portfolio_tools"),
            patch("src.agent.tools.market.register_market_tools"),
            patch("src.agent.tools.news.register_news_tools"),
            patch("src.agent.tools.actions.register_action_tools"),
            patch("src.agent.tools.actions.ActionState"),
            patch("src.orchestrator.settings") as mock_settings,
        ):
            mock_settings.anthropic_api_key = "test-key"
            mock_settings.agent.agent_tool_model = "claude-sonnet-4-5-20250929"
            mock_settings.agent.agent_max_turns = 8
            mock_settings.agent.agent_session_budget = 0.50
            mock_settings.agent.enable_tiered = True
            mock_settings.agent.enable_cache = False
            mock_settings.agent.daily_budget = 3.0  # daily_spend=5.0 > 3.0 → exhausted
            mock_settings.agent.monthly_budget = 75.0

            mock_runner_inst = MagicMock()
            mock_runner_inst.registry = MagicMock()
            mock_runner_inst._token_tracker = TokenTracker()
            mock_runner_cls.return_value = mock_runner_inst

            closes = _make_closes(TICKERS)
            trades, reasoning, tool_summary = await orchestrator._run_portfolio_b_persistent(
                tenant_id="t1",
                trigger_type="morning",
                dynamic_prompt="test prompt",
                closes=closes,
                volumes=closes,
                vix=15.0,
                yield_curve=None,
                regime_str="BULL",
                news_context="",
                positions_for_agent=[],
                cash=50000.0,
                total_value=66000.0,
                recent_trades=[],
                model_override=None,
                portfolio_b_universe=TICKERS,
                allocations=None,
                today=date(2026, 2, 15),
                session="Morning",
                strategy_mode="conservative",
            )

            assert trades == []
            assert "exhausted" in reasoning.lower()


class TestTelegramTieredInfo:
    """Test Telegram brief includes tiered info."""

    def test_session_profile_in_brief(self):
        from src.notifications.telegram_bot import format_daily_brief

        msg = format_daily_brief(
            brief_date=date(2026, 2, 15),
            regime="BULL",
            portfolio_a={"total_value": 33000, "daily_return_pct": 0.5, "top_ticker": "QQQ"},
            portfolio_b={"total_value": 66000, "daily_return_pct": 0.3, "reasoning": "Holding positions"},
            proposed_trades=[],
            agent_tool_summary={"tools_used": 5, "turns": 3, "cost_usd": 0.05, "session_profile": "full"},
        )
        assert "FULL" in msg

    def test_no_profile_if_not_tiered(self):
        from src.notifications.telegram_bot import format_daily_brief

        msg = format_daily_brief(
            brief_date=date(2026, 2, 15),
            regime="BULL",
            portfolio_a={"total_value": 33000, "daily_return_pct": 0.5, "top_ticker": "QQQ"},
            portfolio_b={"total_value": 66000, "daily_return_pct": 0.3, "reasoning": "Holding"},
            proposed_trades=[],
            agent_tool_summary={"tools_used": 3, "turns": 2, "cost_usd": 0.03},
        )
        assert "Profile" not in msg

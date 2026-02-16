"""Tests for orchestrator persistent agent integration.

Tests the _run_portfolio_b_persistent path in the orchestrator and
the three-level fallback logic (persistent → agentic → single-shot).
"""

from datetime import date
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.agent.complexity_detector import ComplexityResult
from src.orchestrator import Orchestrator
from src.storage.database import Database
from src.storage.models import TenantRow


def _make_closes(tickers: list[str], days: int = 250) -> pd.DataFrame:
    """Generate synthetic close prices DataFrame."""
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
    await database.ensure_tenant("t1")
    yield database
    await database.close()


@pytest.fixture
async def orchestrator(db: Database):
    orch = Orchestrator(db)
    yield orch


def _make_tenant(
    use_persistent_agent: bool = False,
    use_agent_loop: bool = False,
) -> TenantRow:
    """Build a TenantRow with specific agent flags."""
    tenant = TenantRow(
        id="t1",
        name="Test Tenant",
        dashboard_user="test",
        dashboard_password_enc="$2b$12$hash",
        run_portfolio_a=True,
        run_portfolio_b=True,
        use_agent_loop=use_agent_loop,
        use_persistent_agent=use_persistent_agent,
    )
    return tenant


class TestPersistentAgentFallback:
    """Tests for the three-level fallback logic."""

    async def test_persistent_flag_routes_to_persistent_path(self, orchestrator: Orchestrator, db: Database):
        """use_persistent_agent=True routes to _run_portfolio_b_persistent."""
        tenant = _make_tenant(use_persistent_agent=True)

        with (
            patch.object(orchestrator._db, "get_tenant", new_callable=AsyncMock, return_value=tenant),
            patch.object(
                orchestrator,
                "_run_portfolio_b_persistent",
                new_callable=AsyncMock,
                return_value=([], "test reasoning", None),
            ) as mock_persistent,
        ):
            # Patch complexity detector to avoid full pipeline
            orchestrator._complexity_detector.evaluate = MagicMock(
                return_value=ComplexityResult(score=0, signals=[], should_escalate=False)
            )

            closes = _make_closes(TICKERS)
            volumes = pd.DataFrame(
                np.random.uniform(1e6, 1e8, closes.shape), index=closes.index, columns=closes.columns
            )

            result = await orchestrator._run_portfolio_b(
                closes=closes,
                volumes=volumes,
                yield_curve=0.5,
                vix=15.0,
                today=date(2026, 2, 5),
                news_context="",
                session="Morning",
                regime_result=None,
                tenant_id="t1",
            )

            mock_persistent.assert_called_once()
            assert result == ([], "test reasoning", None)

    async def test_no_persistent_flag_does_not_call_persistent(self, orchestrator: Orchestrator, db: Database):
        """use_agent_loop=True (without persistent) does NOT route to persistent path."""
        tenant = _make_tenant(use_agent_loop=True, use_persistent_agent=False)

        with (
            patch.object(orchestrator._db, "get_tenant", new_callable=AsyncMock, return_value=tenant),
            patch.object(
                orchestrator,
                "_run_portfolio_b_persistent",
                new_callable=AsyncMock,
            ) as mock_persistent,
            patch.object(orchestrator._strategy_b, "_agent") as mock_agent,
        ):
            orchestrator._complexity_detector.evaluate = MagicMock(
                return_value=ComplexityResult(score=0, signals=[], should_escalate=False)
            )

            # Mock seed phase for agentic path
            mock_agent.analyze.return_value = {
                "regime_assessment": "BULL",
                "reasoning": "Test",
                "trades": [],
                "risk_notes": "",
                "_tokens_used": 500,
                "_model": "claude-sonnet-4-5-20250929",
            }

            # Patch the lazy AgentRunner import in the if-block
            from src.agent.agent_runner import AgentRunResult
            from src.agent.token_tracker import TokenTracker

            mock_runner = MagicMock()
            mock_runner.run = AsyncMock(
                return_value=AgentRunResult(
                    response={"trades": [], "reasoning": "Investigated"},
                    tool_calls=[],
                    turns=2,
                    token_tracker=TokenTracker(),
                    raw_messages=[],
                )
            )
            mock_runner.registry = MagicMock()

            with patch("src.agent.agent_runner.AgentRunner", return_value=mock_runner):
                closes = _make_closes(TICKERS)
                volumes = pd.DataFrame(
                    np.random.uniform(1e6, 1e8, closes.shape), index=closes.index, columns=closes.columns
                )

                await orchestrator._run_portfolio_b(
                    closes=closes,
                    volumes=volumes,
                    yield_curve=0.5,
                    vix=15.0,
                    today=date(2026, 2, 5),
                    news_context="",
                    session="Morning",
                    regime_result=None,
                    tenant_id="t1",
                )

            # Persistent path should NOT have been called
            mock_persistent.assert_not_called()
            # Seed phase should have been called (agentic path)
            mock_agent.analyze.assert_called_once()

    async def test_persistent_takes_priority_over_agentic(self, orchestrator: Orchestrator, db: Database):
        """When both flags are True, persistent takes priority."""
        tenant = _make_tenant(use_persistent_agent=True, use_agent_loop=True)

        with (
            patch.object(orchestrator._db, "get_tenant", new_callable=AsyncMock, return_value=tenant),
            patch.object(
                orchestrator,
                "_run_portfolio_b_persistent",
                new_callable=AsyncMock,
                return_value=([], "persistent", None),
            ) as mock_persistent,
        ):
            orchestrator._complexity_detector.evaluate = MagicMock(
                return_value=ComplexityResult(score=0, signals=[], should_escalate=False)
            )

            closes = _make_closes(TICKERS)
            volumes = pd.DataFrame(
                np.random.uniform(1e6, 1e8, closes.shape), index=closes.index, columns=closes.columns
            )

            result = await orchestrator._run_portfolio_b(
                closes=closes,
                volumes=volumes,
                yield_curve=0.5,
                vix=15.0,
                today=date(2026, 2, 5),
                news_context="",
                session="Morning",
                regime_result=None,
                tenant_id="t1",
            )

            mock_persistent.assert_called_once()
            assert result[1] == "persistent"


class TestRunPortfolioBPersistent:
    """Tests for _run_portfolio_b_persistent method."""

    async def test_persistent_method_returns_3_tuple(self, orchestrator: Orchestrator, db: Database):
        """_run_portfolio_b_persistent returns (trades, reasoning, tool_summary)."""
        from src.agent.persistent_agent import PersistentRunResult
        from src.agent.token_tracker import TokenTracker

        mock_result = PersistentRunResult(
            response={
                "regime_assessment": "BULL",
                "reasoning": "Persistent analysis done.",
                "trades": [],
                "risk_notes": "",
            },
            session_id="t1-morning-abc123",
            tool_calls=[],
            turns=2,
            token_tracker=TokenTracker(),
            tool_summary=None,
            compressed_count=0,
        )

        with (
            patch("src.agent.persistent_agent.PersistentAgent") as mock_pa_cls,
            patch("src.agent.agent_runner.AgentRunner") as mock_runner_cls,
            patch("src.agent.tools.portfolio.register_portfolio_tools"),
            patch("src.agent.tools.market.register_market_tools"),
            patch("src.agent.tools.news.register_news_tools"),
            patch("src.agent.tools.actions.register_action_tools"),
        ):
            mock_pa_instance = MagicMock()
            mock_pa_instance.run_session = AsyncMock(return_value=mock_result)
            mock_pa_cls.return_value = mock_pa_instance

            mock_runner_instance = MagicMock()
            mock_runner_instance.registry = MagicMock()
            mock_runner_cls.return_value = mock_runner_instance

            closes = _make_closes(TICKERS)
            volumes = pd.DataFrame(
                np.random.uniform(1e6, 1e8, closes.shape), index=closes.index, columns=closes.columns
            )

            result = await orchestrator._run_portfolio_b_persistent(
                tenant_id="t1",
                trigger_type="morning",
                dynamic_prompt="Base prompt.",
                closes=closes,
                volumes=volumes,
                vix=15.0,
                yield_curve=0.5,
                regime_str="BULL",
                news_context="",
                positions_for_agent=[],
                cash=25000.0,
                total_value=66000.0,
                recent_trades=[],
                model_override=None,
                portfolio_b_universe=TICKERS,
                allocations=None,
                today=date(2026, 2, 5),
                session="Morning",
                strategy_mode="conservative",
            )

            assert isinstance(result, tuple)
            assert len(result) == 3
            trades, reasoning, tool_summary = result
            assert reasoning == "Persistent analysis done."
            assert tool_summary == {}  # No tools used — empty dict (no trailing_stop_requests)

    async def test_persistent_method_registers_tools(self, orchestrator: Orchestrator, db: Database):
        """_run_portfolio_b_persistent registers all 4 tool sets."""
        from src.agent.persistent_agent import PersistentRunResult
        from src.agent.token_tracker import TokenTracker

        mock_result = PersistentRunResult(
            response={"trades": [], "reasoning": "Done", "risk_notes": ""},
            session_id="t1-morning-abc",
            token_tracker=TokenTracker(),
        )

        with (
            patch("src.agent.persistent_agent.PersistentAgent") as mock_pa_cls,
            patch("src.agent.agent_runner.AgentRunner") as mock_runner_cls,
            patch("src.agent.tools.portfolio.register_portfolio_tools") as mock_port,
            patch("src.agent.tools.market.register_market_tools") as mock_mkt,
            patch("src.agent.tools.news.register_news_tools") as mock_news,
            patch("src.agent.tools.actions.register_action_tools") as mock_act,
        ):
            mock_pa_instance = MagicMock()
            mock_pa_instance.run_session = AsyncMock(return_value=mock_result)
            mock_pa_cls.return_value = mock_pa_instance

            mock_runner_instance = MagicMock()
            mock_runner_instance.registry = MagicMock()
            mock_runner_cls.return_value = mock_runner_instance

            closes = _make_closes(TICKERS)
            volumes = pd.DataFrame(
                np.random.uniform(1e6, 1e8, closes.shape), index=closes.index, columns=closes.columns
            )

            await orchestrator._run_portfolio_b_persistent(
                tenant_id="t1",
                trigger_type="morning",
                dynamic_prompt="Base prompt.",
                closes=closes,
                volumes=volumes,
                vix=15.0,
                yield_curve=0.5,
                regime_str="BULL",
                news_context="",
                positions_for_agent=[],
                cash=25000.0,
                total_value=66000.0,
                recent_trades=[],
                model_override=None,
                portfolio_b_universe=TICKERS,
                allocations=None,
                today=date(2026, 2, 5),
                session="Morning",
                strategy_mode="conservative",
            )

            mock_port.assert_called_once()
            mock_mkt.assert_called_once()
            mock_news.assert_called_once()
            mock_act.assert_called_once()

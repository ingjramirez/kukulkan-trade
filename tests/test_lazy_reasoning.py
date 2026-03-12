"""Tests for lazy reasoning detection and humanization."""

from dataclasses import dataclass, field
from unittest.mock import AsyncMock, patch

from src.orchestrator import Orchestrator


class TestIsLazyReasoning:
    """_is_lazy_reasoning detects lazy/placeholder agent responses."""

    def test_empty_reasoning(self) -> None:
        assert Orchestrator._is_lazy_reasoning("") is True

    def test_short_reasoning(self) -> None:
        assert Orchestrator._is_lazy_reasoning("ok") is True

    def test_already_processed(self) -> None:
        assert Orchestrator._is_lazy_reasoning("Already processed this data.") is True

    def test_session_complete_long(self) -> None:
        """Long responses with lazy patterns are still detected."""
        text = (
            "Session complete — all morning tasks executed before this notification arrived. "
            "The delayed agent returned context.md from 09:31 EST (earlier than the current session). "
            "All positions reviewed and no changes needed at this time."
        )
        assert Orchestrator._is_lazy_reasoning(text) is True

    def test_stale_notification(self) -> None:
        text = "Stale notification — same context.md already processed. Session complete."
        assert Orchestrator._is_lazy_reasoning(text) is True

    def test_context_md_reference(self) -> None:
        text = "The context.md shows the same data I reviewed earlier today."
        assert Orchestrator._is_lazy_reasoning(text) is True

    def test_real_reasoning_not_flagged(self) -> None:
        text = (
            "Markets showing weakness with SPY down 0.8% on tariff concerns. "
            "Rotating out of META and MSFT into defensive sectors XLP and XLV. "
            "VIX elevated at 22.3 suggests continued volatility ahead."
        )
        assert Orchestrator._is_lazy_reasoning(text) is False

    def test_real_short_reasoning_not_flagged(self) -> None:
        text = "Holding all positions. Market consolidating near support."
        assert Orchestrator._is_lazy_reasoning(text) is False


@dataclass
class FakeInvokeResult:
    response: dict = field(default_factory=dict)
    tool_call_logs: list = field(default_factory=list)
    posture: str | None = None
    mcp_executed_trades: list = field(default_factory=list)


class TestHumanizeReasoning:
    """_humanize_reasoning calls Claude to rewrite lazy reasoning."""

    async def test_calls_claude_cli(self) -> None:
        trades = [{"side": "SELL", "ticker": "META", "weight": 0.05, "conviction": "high", "reason": "Tariff risk"}]
        result = FakeInvokeResult(
            response={"trades": trades},
            tool_call_logs=[
                {"tool_name": "get_market_overview", "output_preview": "SPY -0.8%, VIX 22.3", "success": True},
                {"tool_name": "get_portfolio_state", "output_preview": "5 positions, $42K cash", "success": True},
            ],
            posture="defensive",
        )
        with patch("src.agent.claude_invoker.claude_cli_call", new_callable=AsyncMock) as mock_cli:
            mock_cli.return_value = (
                "Markets under pressure from tariff fears. "
                "Shifting to defensive positioning."
            )
            brief = await Orchestrator._humanize_reasoning(result, [])
            mock_cli.assert_called_once()
            assert "tariff" in brief.lower()
            assert len(brief) > 30

    async def test_fallback_on_failure(self) -> None:
        result = FakeInvokeResult(
            tool_call_logs=[{"tool_name": "get_market_overview", "output_preview": "SPY flat", "success": True}],
            posture="neutral",
        )
        with patch("src.agent.claude_invoker.claude_cli_call", new_callable=AsyncMock) as mock_cli:
            mock_cli.side_effect = Exception("timeout")
            brief = await Orchestrator._humanize_reasoning(result, [])
            assert "tool calls" in brief.lower() or "tool activity" in brief.lower()

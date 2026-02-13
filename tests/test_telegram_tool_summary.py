"""Tests for agent tool summary in Telegram daily brief."""

from datetime import date

from src.notifications.telegram_bot import format_daily_brief
from src.storage.models import OrderSide, PortfolioName, TradeSchema


def _make_trade() -> TradeSchema:
    return TradeSchema(
        ticker="XLK",
        side=OrderSide.BUY,
        shares=10,
        price=200.0,
        total=2000.0,
        portfolio=PortfolioName.B,
        reason="test",
    )


class TestAgentToolSummary:
    def test_includes_tool_summary_when_present(self) -> None:
        tool_summary = {"tools_used": 5, "turns": 3, "cost_usd": 0.18}
        msg = format_daily_brief(
            brief_date=date(2026, 2, 13),
            regime="BULL",
            portfolio_a={"total_value": 34000, "daily_return_pct": 1.0, "top_ticker": "XLK"},
            portfolio_b={"total_value": 67000, "daily_return_pct": 0.5, "reasoning": "Test reasoning"},
            proposed_trades=[_make_trade()],
            run_portfolio_a=True,
            run_portfolio_b=True,
            agent_tool_summary=tool_summary,
        )
        assert "Investigation: 5 tools across 3 turns" in msg
        assert "$0.18" in msg

    def test_no_tool_summary_for_single_shot(self) -> None:
        msg = format_daily_brief(
            brief_date=date(2026, 2, 13),
            regime="BULL",
            portfolio_a={"total_value": 34000, "daily_return_pct": 1.0, "top_ticker": "XLK"},
            portfolio_b={"total_value": 67000, "daily_return_pct": 0.5, "reasoning": "Test reasoning"},
            proposed_trades=[],
            run_portfolio_a=True,
            run_portfolio_b=True,
            agent_tool_summary=None,
        )
        assert "Investigation" not in msg

    def test_tool_summary_not_shown_when_portfolio_b_disabled(self) -> None:
        tool_summary = {"tools_used": 3, "turns": 2, "cost_usd": 0.10}
        msg = format_daily_brief(
            brief_date=date(2026, 2, 13),
            regime="BULL",
            portfolio_a={"total_value": 34000, "daily_return_pct": 1.0, "top_ticker": "XLK"},
            portfolio_b={},
            proposed_trades=[],
            run_portfolio_a=True,
            run_portfolio_b=False,
            agent_tool_summary=tool_summary,
        )
        # Portfolio B section skipped entirely, so no tool summary
        assert "Investigation" not in msg

    def test_tool_summary_cost_formatting(self) -> None:
        tool_summary = {"tools_used": 12, "turns": 8, "cost_usd": 0.4523}
        msg = format_daily_brief(
            brief_date=date(2026, 2, 13),
            regime=None,
            portfolio_a={},
            portfolio_b={"total_value": 67000, "daily_return_pct": None, "reasoning": "test"},
            proposed_trades=[],
            run_portfolio_a=False,
            run_portfolio_b=True,
            agent_tool_summary=tool_summary,
        )
        assert "$0.45" in msg
        assert "12 tools across 8 turns" in msg

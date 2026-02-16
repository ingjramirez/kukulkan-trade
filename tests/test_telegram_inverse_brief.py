"""Tests for inverse exposure and hold alerts in daily brief formatting."""

from datetime import date

from src.notifications.telegram_bot import format_daily_brief


def _empty_portfolio() -> dict:
    return {"total_value": 50_000, "cash": 10_000, "reasoning": "test", "daily_return_pct": 0.5}


class TestDailyBriefInverseExposure:
    def test_includes_inverse_section(self) -> None:
        inverse_exposure = {
            "total_value": 3000.0,
            "total_pct": 3.0,
            "net_equity_pct": 87.0,
            "positions": [
                {"ticker": "SH", "value": 3000.0, "pct": 3.0, "equity_hedge": True},
            ],
        }
        msg = format_daily_brief(
            brief_date=date(2026, 2, 16),
            regime="CORRECTION",
            portfolio_a=_empty_portfolio(),
            portfolio_b=_empty_portfolio(),
            proposed_trades=[],
            inverse_exposure=inverse_exposure,
        )
        assert "Inverse Exposure" in msg
        assert "SH" in msg
        assert "equity hedge" in msg
        assert "Net equity exposure" in msg

    def test_omitted_when_no_inverse(self) -> None:
        msg = format_daily_brief(
            brief_date=date(2026, 2, 16),
            regime="BULL",
            portfolio_a=_empty_portfolio(),
            portfolio_b=_empty_portfolio(),
            proposed_trades=[],
            inverse_exposure=None,
        )
        assert "Inverse Exposure" not in msg

    def test_omitted_when_empty_positions(self) -> None:
        inverse_exposure = {
            "total_value": 0.0,
            "total_pct": 0.0,
            "net_equity_pct": 0.0,
            "positions": [],
        }
        msg = format_daily_brief(
            brief_date=date(2026, 2, 16),
            regime="BULL",
            portfolio_a=_empty_portfolio(),
            portfolio_b=_empty_portfolio(),
            proposed_trades=[],
            inverse_exposure=inverse_exposure,
        )
        assert "Inverse Exposure" not in msg


class TestDailyBriefHoldAlerts:
    def test_includes_hold_alerts(self) -> None:
        alerts = [
            {"ticker": "SH", "days_held": 5, "alert_level": "review", "message": "SH held 5d — review"},
        ]
        msg = format_daily_brief(
            brief_date=date(2026, 2, 16),
            regime="CORRECTION",
            portfolio_a=_empty_portfolio(),
            portfolio_b=_empty_portfolio(),
            proposed_trades=[],
            inverse_hold_alerts=alerts,
        )
        assert "Inverse Hold Alerts" in msg
        assert "SH held 5d" in msg

    def test_no_alerts_section_when_empty(self) -> None:
        msg = format_daily_brief(
            brief_date=date(2026, 2, 16),
            regime="BULL",
            portfolio_a=_empty_portfolio(),
            portfolio_b=_empty_portfolio(),
            proposed_trades=[],
            inverse_hold_alerts=None,
        )
        assert "Inverse Hold Alerts" not in msg

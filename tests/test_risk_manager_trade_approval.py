"""Tests for large trade approval threshold in RiskManager.check_pre_trade()."""

import pytest

from src.analysis.risk_manager import RiskManager
from src.storage.models import OrderSide, PortfolioName, TradeSchema


def _make_trade(ticker: str, side: str = "BUY", shares: float = 100, price: float = 15.0) -> TradeSchema:
    return TradeSchema(
        ticker=ticker,
        side=OrderSide(side),
        shares=shares,
        price=price,
        total=shares * price,
        portfolio=PortfolioName.B,
        reason="test",
    )


@pytest.fixture
def rm() -> RiskManager:
    return RiskManager()


class TestLargeTradeApproval:
    """Rule 4: Non-inverse BUYs > threshold% of portfolio flagged for approval."""

    def test_trade_above_10pct_requires_approval(self, rm: RiskManager, monkeypatch) -> None:
        """A trade worth 15% of portfolio should be flagged."""
        from config import settings as settings_mod

        monkeypatch.setattr(settings_mod.settings, "trade_approval_threshold_pct", 10.0)
        # Trade: 100 shares @ $150 = $15,000 on a $100,000 portfolio = 15%
        verdict = rm.check_pre_trade(
            trades=[_make_trade("AAPL", shares=100, price=150.0)],
            portfolio_name="B",
            current_positions={},
            latest_prices={"AAPL": 150.0},
            portfolio_value=100_000,
            cash=100_000,
        )
        assert len(verdict.requires_trade_approval) == 1
        trade, reason = verdict.requires_trade_approval[0]
        assert trade.ticker == "AAPL"
        assert "15.0%" in reason
        # Trade is still in allowed (orchestrator decides to remove if rejected)
        assert len(verdict.allowed) == 1

    def test_trade_below_10pct_no_approval(self, rm: RiskManager, monkeypatch) -> None:
        """A trade worth 5% of portfolio should NOT be flagged."""
        from config import settings as settings_mod

        monkeypatch.setattr(settings_mod.settings, "trade_approval_threshold_pct", 10.0)
        # Trade: 50 shares @ $100 = $5,000 on a $100,000 portfolio = 5%
        verdict = rm.check_pre_trade(
            trades=[_make_trade("AAPL", shares=50, price=100.0)],
            portfolio_name="B",
            current_positions={},
            latest_prices={"AAPL": 100.0},
            portfolio_value=100_000,
            cash=100_000,
        )
        assert len(verdict.requires_trade_approval) == 0
        assert len(verdict.allowed) == 1

    def test_trade_exactly_10pct_no_approval(self, rm: RiskManager, monkeypatch) -> None:
        """Boundary: exactly 10% should NOT trigger (> not >=)."""
        from config import settings as settings_mod

        monkeypatch.setattr(settings_mod.settings, "trade_approval_threshold_pct", 10.0)
        # Trade: 100 shares @ $100 = $10,000 on a $100,000 portfolio = 10%
        verdict = rm.check_pre_trade(
            trades=[_make_trade("AAPL", shares=100, price=100.0)],
            portfolio_name="B",
            current_positions={},
            latest_prices={"AAPL": 100.0},
            portfolio_value=100_000,
            cash=100_000,
        )
        assert len(verdict.requires_trade_approval) == 0

    def test_inverse_etf_skips_large_trade_check(self, rm: RiskManager, monkeypatch) -> None:
        """Inverse ETFs have their own approval flow — should NOT appear in requires_trade_approval."""
        from config import settings as settings_mod

        monkeypatch.setattr(settings_mod.settings, "trade_approval_threshold_pct", 5.0)
        # SH: 100 shares @ $15 = $1,500 on $100k portfolio = 1.5% (under single inverse 10% cap)
        # But trade_approval_threshold is 5%, so 1.5% wouldn't trigger anyway.
        # Use a scenario where it WOULD trigger for a regular stock but doesn't for inverse.
        # SH: 100 shares @ $15 = $1,500 on $20k portfolio = 7.5% (under 10% inverse cap, above 5% threshold)
        verdict = rm.check_pre_trade(
            trades=[_make_trade("SH", shares=100, price=15.0)],
            portfolio_name="B",
            current_positions={},
            latest_prices={"SH": 15.0},
            portfolio_value=20_000,
            cash=20_000,
            regime="CORRECTION",
            current_posture="defensive",
        )
        assert len(verdict.requires_trade_approval) == 0
        # SH should be in the inverse requires_approval list instead
        assert len(verdict.requires_approval) == 1

    def test_blocked_trade_never_reaches_approval_check(self, rm: RiskManager, monkeypatch) -> None:
        """If a trade is blocked by concentration limits, it shouldn't be flagged for approval."""
        from config import settings as settings_mod

        monkeypatch.setattr(settings_mod.settings, "trade_approval_threshold_pct", 10.0)
        # Trade: 400 shares @ $100 = $40,000 on a $100,000 portfolio = 40%
        # This exceeds the 35% max_single_position_pct and gets blocked
        verdict = rm.check_pre_trade(
            trades=[_make_trade("AAPL", shares=400, price=100.0)],
            portfolio_name="B",
            current_positions={},
            latest_prices={"AAPL": 100.0},
            portfolio_value=100_000,
            cash=100_000,
        )
        assert len(verdict.blocked) == 1
        assert len(verdict.requires_trade_approval) == 0

    def test_approval_reason_includes_pct_and_value(self, rm: RiskManager, monkeypatch) -> None:
        """Approval reason should contain the trade value and percentage."""
        from config import settings as settings_mod

        monkeypatch.setattr(settings_mod.settings, "trade_approval_threshold_pct", 10.0)
        verdict = rm.check_pre_trade(
            trades=[_make_trade("MSFT", shares=100, price=200.0)],
            portfolio_name="B",
            current_positions={},
            latest_prices={"MSFT": 200.0},
            portfolio_value=100_000,
            cash=100_000,
        )
        assert len(verdict.requires_trade_approval) == 1
        _, reason = verdict.requires_trade_approval[0]
        assert "$20,000" in reason
        assert "20.0%" in reason
        assert "threshold" in reason.lower()

    def test_sell_never_requires_approval(self, rm: RiskManager, monkeypatch) -> None:
        """SELLs should never trigger the large trade approval check."""
        from config import settings as settings_mod

        monkeypatch.setattr(settings_mod.settings, "trade_approval_threshold_pct", 10.0)
        verdict = rm.check_pre_trade(
            trades=[_make_trade("AAPL", side="SELL", shares=100, price=150.0)],
            portfolio_name="B",
            current_positions={"AAPL": 200},
            latest_prices={"AAPL": 150.0},
            portfolio_value=100_000,
            cash=70_000,
        )
        assert len(verdict.requires_trade_approval) == 0
        assert len(verdict.allowed) == 1

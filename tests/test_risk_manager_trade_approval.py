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
    """Rule 4: Non-inverse BUYs > threshold% flagged when trade_approval_enabled=True."""

    def test_trade_above_10pct_requires_approval_when_enabled(self, rm: RiskManager, monkeypatch) -> None:
        """A trade worth 15% of portfolio should be flagged when approval is enabled."""
        from config import settings as settings_mod

        monkeypatch.setattr(settings_mod.settings, "trade_approval_threshold_pct", 10.0)
        monkeypatch.setattr(settings_mod.settings, "trade_approval_enabled", True)
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
        assert len(verdict.allowed) == 1

    def test_trade_above_10pct_no_approval_when_disabled(self, rm: RiskManager, monkeypatch) -> None:
        """Large trade goes straight to allowed when approval is disabled (paper trading)."""
        from config import settings as settings_mod

        monkeypatch.setattr(settings_mod.settings, "trade_approval_threshold_pct", 10.0)
        monkeypatch.setattr(settings_mod.settings, "trade_approval_enabled", False)
        verdict = rm.check_pre_trade(
            trades=[_make_trade("AAPL", shares=100, price=150.0)],
            portfolio_name="B",
            current_positions={},
            latest_prices={"AAPL": 150.0},
            portfolio_value=100_000,
            cash=100_000,
        )
        assert len(verdict.requires_trade_approval) == 0
        assert len(verdict.allowed) == 1

    def test_trade_below_10pct_no_approval(self, rm: RiskManager, monkeypatch) -> None:
        """A trade worth 5% of portfolio should NOT be flagged."""
        from config import settings as settings_mod

        monkeypatch.setattr(settings_mod.settings, "trade_approval_threshold_pct", 10.0)
        monkeypatch.setattr(settings_mod.settings, "trade_approval_enabled", True)
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
        monkeypatch.setattr(settings_mod.settings, "trade_approval_enabled", True)
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
        """Inverse ETFs have their own flow — should NOT appear in requires_trade_approval."""
        from config import settings as settings_mod

        monkeypatch.setattr(settings_mod.settings, "trade_approval_threshold_pct", 5.0)
        monkeypatch.setattr(settings_mod.settings, "trade_approval_enabled", True)
        # SH: 100 shares @ $15 = $1,500 on $20k portfolio = 7.5% (above 5% threshold)
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
        # SH goes to allowed (no approval in paper trading mode)
        assert len(verdict.allowed) == 1

    def test_blocked_trade_never_reaches_approval_check(self, rm: RiskManager, monkeypatch) -> None:
        """If a trade is blocked by concentration limits, it shouldn't be flagged for approval."""
        from config import settings as settings_mod

        monkeypatch.setattr(settings_mod.settings, "trade_approval_threshold_pct", 10.0)
        monkeypatch.setattr(settings_mod.settings, "trade_approval_enabled", True)
        # Trade: 600 shares @ $100 = $60,000 on a $100,000 portfolio = 60%
        # This exceeds the 50% max_single_position_pct and gets blocked
        verdict = rm.check_pre_trade(
            trades=[_make_trade("AAPL", shares=600, price=100.0)],
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
        monkeypatch.setattr(settings_mod.settings, "trade_approval_enabled", True)
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
        monkeypatch.setattr(settings_mod.settings, "trade_approval_enabled", True)
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

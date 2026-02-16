"""Tests for inverse ETF risk rules in RiskManager.check_pre_trade()."""

import pytest

from src.analysis.risk_manager import (
    RiskManager,
)
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


class TestRegimeGate:
    """Rule 0a: equity hedges blocked unless regime in CORRECTION/CRISIS."""

    def test_blocked_in_bull(self, rm: RiskManager) -> None:
        verdict = rm.check_pre_trade(
            trades=[_make_trade("SH")],
            portfolio_name="B",
            current_positions={},
            latest_prices={"SH": 15.0},
            portfolio_value=100_000,
            cash=100_000,
            regime="BULL",
            current_posture="defensive",
        )
        assert len(verdict.blocked) == 1
        assert "regime" in verdict.blocked[0][1].lower()

    def test_blocked_in_consolidation(self, rm: RiskManager) -> None:
        verdict = rm.check_pre_trade(
            trades=[_make_trade("SH")],
            portfolio_name="B",
            current_positions={},
            latest_prices={"SH": 15.0},
            portfolio_value=100_000,
            cash=100_000,
            regime="CONSOLIDATION",
            current_posture="defensive",
        )
        assert len(verdict.blocked) == 1

    def test_allowed_in_correction(self, rm: RiskManager) -> None:
        verdict = rm.check_pre_trade(
            trades=[_make_trade("SH")],
            portfolio_name="B",
            current_positions={},
            latest_prices={"SH": 15.0},
            portfolio_value=100_000,
            cash=100_000,
            regime="CORRECTION",
            current_posture="defensive",
        )
        assert len(verdict.allowed) == 1

    def test_allowed_in_crisis(self, rm: RiskManager) -> None:
        verdict = rm.check_pre_trade(
            trades=[_make_trade("SH")],
            portfolio_name="B",
            current_positions={},
            latest_prices={"SH": 15.0},
            portfolio_value=100_000,
            cash=100_000,
            regime="CRISIS",
            current_posture="crisis",
        )
        assert len(verdict.allowed) == 1

    def test_regime_none_blocks(self, rm: RiskManager) -> None:
        verdict = rm.check_pre_trade(
            trades=[_make_trade("SH")],
            portfolio_name="B",
            current_positions={},
            latest_prices={"SH": 15.0},
            portfolio_value=100_000,
            cash=100_000,
            regime=None,
            current_posture="defensive",
        )
        assert len(verdict.blocked) == 1


class TestPostureGate:
    """Rule 0b: equity hedges blocked unless posture is defensive/crisis."""

    def test_blocked_in_balanced(self, rm: RiskManager) -> None:
        verdict = rm.check_pre_trade(
            trades=[_make_trade("SH")],
            portfolio_name="B",
            current_positions={},
            latest_prices={"SH": 15.0},
            portfolio_value=100_000,
            cash=100_000,
            regime="CORRECTION",
            current_posture="balanced",
        )
        assert len(verdict.blocked) == 1
        assert "posture" in verdict.blocked[0][1].lower()

    def test_blocked_in_aggressive(self, rm: RiskManager) -> None:
        verdict = rm.check_pre_trade(
            trades=[_make_trade("SH")],
            portfolio_name="B",
            current_positions={},
            latest_prices={"SH": 15.0},
            portfolio_value=100_000,
            cash=100_000,
            regime="CORRECTION",
            current_posture="aggressive",
        )
        assert len(verdict.blocked) == 1

    def test_posture_none_blocks(self, rm: RiskManager) -> None:
        verdict = rm.check_pre_trade(
            trades=[_make_trade("SH")],
            portfolio_name="B",
            current_positions={},
            latest_prices={"SH": 15.0},
            portfolio_value=100_000,
            cash=100_000,
            regime="CORRECTION",
            current_posture=None,
        )
        assert len(verdict.blocked) == 1

    def test_allowed_in_defensive(self, rm: RiskManager) -> None:
        verdict = rm.check_pre_trade(
            trades=[_make_trade("SH")],
            portfolio_name="B",
            current_positions={},
            latest_prices={"SH": 15.0},
            portfolio_value=100_000,
            cash=100_000,
            regime="CORRECTION",
            current_posture="defensive",
        )
        assert len(verdict.allowed) == 1


class TestTbfExemptFromGating:
    """TBF hedges interest rate risk — exempt from regime/posture gating."""

    def test_tbf_allowed_in_bull_balanced(self, rm: RiskManager) -> None:
        verdict = rm.check_pre_trade(
            trades=[_make_trade("TBF")],
            portfolio_name="B",
            current_positions={},
            latest_prices={"TBF": 18.0},
            portfolio_value=100_000,
            cash=100_000,
            regime="BULL",
            current_posture="balanced",
        )
        assert len(verdict.allowed) == 1
        assert verdict.allowed[0].ticker == "TBF"

    def test_tbf_allowed_any_regime(self, rm: RiskManager) -> None:
        for regime in ("BULL", "CONSOLIDATION", "CORRECTION", "CRISIS"):
            verdict = rm.check_pre_trade(
                trades=[_make_trade("TBF")],
                portfolio_name="B",
                current_positions={},
                latest_prices={"TBF": 18.0},
                portfolio_value=100_000,
                cash=100_000,
                regime=regime,
                current_posture="balanced",
            )
            assert len(verdict.allowed) == 1, f"TBF blocked in {regime}"


class TestSingleInverseLimit:
    """Rule 0c: max 10% single inverse position."""

    def test_blocked_over_10pct(self, rm: RiskManager) -> None:
        # 11% of portfolio
        verdict = rm.check_pre_trade(
            trades=[_make_trade("SH", shares=733, price=15.0)],  # $10,995 = ~11%
            portfolio_name="B",
            current_positions={},
            latest_prices={"SH": 15.0},
            portfolio_value=100_000,
            cash=100_000,
            regime="CORRECTION",
            current_posture="defensive",
        )
        assert len(verdict.blocked) == 1
        assert "inverse position" in verdict.blocked[0][1].lower()

    def test_allowed_under_10pct(self, rm: RiskManager) -> None:
        # ~9% of portfolio
        verdict = rm.check_pre_trade(
            trades=[_make_trade("SH", shares=600, price=15.0)],  # $9,000 = 9%
            portfolio_name="B",
            current_positions={},
            latest_prices={"SH": 15.0},
            portfolio_value=100_000,
            cash=100_000,
            regime="CORRECTION",
            current_posture="defensive",
        )
        assert len(verdict.allowed) == 1


class TestTotalInverseExposure:
    """Rule 0d: max 15% total inverse exposure."""

    def test_blocked_total_over_15pct(self, rm: RiskManager) -> None:
        # Already have SH at 10%, try adding PSQ at 6%
        verdict = rm.check_pre_trade(
            trades=[_make_trade("PSQ", shares=500, price=12.0)],  # $6,000 = 6%
            portfolio_name="B",
            current_positions={"SH": 666},  # ~$10,000
            latest_prices={"SH": 15.0, "PSQ": 12.0},
            portfolio_value=100_000,
            cash=90_000,
            regime="CORRECTION",
            current_posture="defensive",
        )
        assert len(verdict.blocked) == 1
        assert "total inverse" in verdict.blocked[0][1].lower()


class TestMaxInversePositions:
    """Rule 0e: max 2 inverse positions."""

    def test_blocked_third_position(self, rm: RiskManager) -> None:
        verdict = rm.check_pre_trade(
            trades=[_make_trade("RWM", shares=100, price=20.0)],
            portfolio_name="B",
            current_positions={"SH": 100, "PSQ": 100},
            latest_prices={"SH": 15.0, "PSQ": 12.0, "RWM": 20.0},
            portfolio_value=100_000,
            cash=95_000,
            regime="CORRECTION",
            current_posture="defensive",
        )
        assert len(verdict.blocked) == 1
        assert "inverse positions" in verdict.blocked[0][1].lower()

    def test_adding_to_existing_position_ok(self, rm: RiskManager) -> None:
        """Adding shares to an existing inverse position doesn't count as a new position."""
        verdict = rm.check_pre_trade(
            trades=[_make_trade("SH", shares=50, price=15.0)],  # Add to existing
            portfolio_name="B",
            current_positions={"SH": 100, "PSQ": 100},
            latest_prices={"SH": 15.0, "PSQ": 12.0},
            portfolio_value=100_000,
            cash=95_000,
            regime="CORRECTION",
            current_posture="defensive",
        )
        assert len(verdict.allowed) == 1


class TestRequiresApproval:
    """Inverse BUYs that pass all checks appear in requires_approval."""

    def test_passing_inverse_buy_in_requires_approval(self, rm: RiskManager) -> None:
        verdict = rm.check_pre_trade(
            trades=[_make_trade("SH", shares=100, price=15.0)],
            portfolio_name="B",
            current_positions={},
            latest_prices={"SH": 15.0},
            portfolio_value=100_000,
            cash=100_000,
            regime="CORRECTION",
            current_posture="defensive",
        )
        assert len(verdict.requires_approval) == 1
        assert verdict.requires_approval[0].ticker == "SH"

    def test_non_inverse_not_in_requires_approval(self, rm: RiskManager) -> None:
        verdict = rm.check_pre_trade(
            trades=[_make_trade("XLK", shares=10, price=200.0)],
            portfolio_name="B",
            current_positions={},
            latest_prices={"XLK": 200.0},
            portfolio_value=100_000,
            cash=100_000,
            regime="CORRECTION",
            current_posture="defensive",
        )
        assert len(verdict.requires_approval) == 0
        assert len(verdict.allowed) == 1


class TestSellsAlwaysPass:
    """SELL orders bypass inverse checks."""

    def test_sell_inverse_in_bull(self, rm: RiskManager) -> None:
        verdict = rm.check_pre_trade(
            trades=[_make_trade("SH", side="SELL", shares=100, price=15.0)],
            portfolio_name="B",
            current_positions={"SH": 100},
            latest_prices={"SH": 15.0},
            portfolio_value=100_000,
            cash=90_000,
            regime="BULL",
            current_posture="balanced",
        )
        assert len(verdict.allowed) == 1
        assert len(verdict.blocked) == 0


class TestExistingRulesStillApply:
    """Standard rules still run after inverse checks pass."""

    def test_sector_concentration_still_applies(self, rm: RiskManager) -> None:
        # SH passes inverse checks but sector concentration already maxed
        # We need a large position to trigger sector concentration
        # SH is in "Inverse" sector — with default 50% sector limit,
        # an existing PSQ position at 49% + new SH would exceed
        verdict = rm.check_pre_trade(
            trades=[_make_trade("XLK", shares=300, price=200.0)],  # $60K = 60% of portfolio
            portfolio_name="B",
            current_positions={"NVDA": 200},  # Also tech
            latest_prices={"XLK": 200.0, "NVDA": 120.0},
            portfolio_value=100_000,
            cash=76_000,
            regime="CORRECTION",
            current_posture="defensive",
        )
        # Should be blocked by sector concentration, not inverse rules
        assert len(verdict.blocked) == 1

    def test_non_inverse_trades_unaffected(self, rm: RiskManager) -> None:
        """Standard trades should work exactly as before with regime/posture params."""
        verdict = rm.check_pre_trade(
            trades=[_make_trade("XLK", shares=10, price=200.0)],
            portfolio_name="B",
            current_positions={},
            latest_prices={"XLK": 200.0},
            portfolio_value=100_000,
            cash=100_000,
            regime="BULL",
            current_posture="balanced",
        )
        assert len(verdict.allowed) == 1
        assert len(verdict.blocked) == 0

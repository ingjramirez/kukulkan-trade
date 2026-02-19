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


class TestPostureGateRemoved:
    """Posture gate removed for paper trading — all postures allow inverse."""

    def test_allowed_in_balanced(self, rm: RiskManager) -> None:
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
        assert len(verdict.allowed) == 1

    def test_allowed_in_aggressive(self, rm: RiskManager) -> None:
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
        assert len(verdict.allowed) == 1

    def test_posture_none_allowed(self, rm: RiskManager) -> None:
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
        assert len(verdict.allowed) == 1

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
    """Rule 0c: max 20% single inverse position."""

    def test_blocked_over_20pct(self, rm: RiskManager) -> None:
        # 21% of portfolio
        verdict = rm.check_pre_trade(
            trades=[_make_trade("SH", shares=1400, price=15.0)],  # $21,000 = 21%
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

    def test_allowed_under_20pct(self, rm: RiskManager) -> None:
        # ~19% of portfolio
        verdict = rm.check_pre_trade(
            trades=[_make_trade("SH", shares=1266, price=15.0)],  # $18,990 = ~19%
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
    """Rule 0d: max 30% total inverse exposure."""

    def test_blocked_total_over_30pct(self, rm: RiskManager) -> None:
        # Already have SH at 19%, try adding PSQ at 12% → 31% total
        verdict = rm.check_pre_trade(
            trades=[_make_trade("PSQ", shares=1000, price=12.0)],  # $12,000 = 12%
            portfolio_name="B",
            current_positions={"SH": 1266},  # ~$18,990 = ~19%
            latest_prices={"SH": 15.0, "PSQ": 12.0},
            portfolio_value=100_000,
            cash=69_000,
            regime="CORRECTION",
            current_posture="defensive",
        )
        assert len(verdict.blocked) == 1
        assert "total inverse" in verdict.blocked[0][1].lower()


class TestMaxInversePositions:
    """Rule 0e: max 4 inverse positions (matches universe size of 4 inverse ETFs)."""

    def test_all_four_inverse_allowed(self, rm: RiskManager) -> None:
        """With 3 existing inverse positions, a 4th is allowed (limit is 4)."""
        verdict = rm.check_pre_trade(
            trades=[_make_trade("TBF", shares=100, price=18.0)],
            portfolio_name="B",
            current_positions={"SH": 100, "PSQ": 100, "RWM": 100},
            latest_prices={"SH": 15.0, "PSQ": 12.0, "RWM": 20.0, "TBF": 18.0},
            portfolio_value=100_000,
            cash=93_000,
            regime="CORRECTION",
            current_posture="defensive",
        )
        assert len(verdict.allowed) == 1

    def test_max_inverse_positions_constant(self) -> None:
        """Verify the constant matches expected value."""
        from src.analysis.risk_manager import MAX_INVERSE_POSITIONS

        assert MAX_INVERSE_POSITIONS == 4

    def test_max_inverse_positions_equals_universe_size(self) -> None:
        """MAX_INVERSE_POSITIONS equals the inverse ETF universe size (tight limit)."""
        from config.universe import INVERSE_ETF_META
        from src.analysis.risk_manager import MAX_INVERSE_POSITIONS

        assert MAX_INVERSE_POSITIONS == len(INVERSE_ETF_META)

    def test_all_four_held_adding_shares_ok(self, rm: RiskManager) -> None:
        """With all 4 inverse ETFs held, adding to existing doesn't trigger the count limit."""
        verdict = rm.check_pre_trade(
            trades=[_make_trade("SH", shares=50, price=15.0)],
            portfolio_name="B",
            current_positions={"SH": 100, "PSQ": 100, "RWM": 100, "TBF": 100},
            latest_prices={"SH": 15.0, "PSQ": 12.0, "RWM": 20.0, "TBF": 18.0},
            portfolio_value=100_000,
            cash=93_000,
            regime="CORRECTION",
            current_posture="defensive",
        )
        assert len(verdict.allowed) == 1

    def test_adding_to_existing_position_ok(self, rm: RiskManager) -> None:
        """Adding shares to an existing inverse position doesn't count as a new position."""
        verdict = rm.check_pre_trade(
            trades=[_make_trade("SH", shares=50, price=15.0)],  # Add to existing
            portfolio_name="B",
            current_positions={"SH": 100, "PSQ": 100, "RWM": 100},
            latest_prices={"SH": 15.0, "PSQ": 12.0, "RWM": 20.0},
            portfolio_value=100_000,
            cash=93_000,
            regime="CORRECTION",
            current_posture="defensive",
        )
        assert len(verdict.allowed) == 1


class TestInverseNoApproval:
    """Inverse BUYs go straight to allowed (no approval in paper trading)."""

    def test_passing_inverse_buy_goes_to_allowed(self, rm: RiskManager) -> None:
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
        assert len(verdict.requires_approval) == 0
        assert len(verdict.allowed) == 1
        assert verdict.allowed[0].ticker == "SH"

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

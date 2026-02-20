"""Tests for paper trading freedom — guardrail relaxation."""

from config.risk_rules import RISK_RULES
from config.settings import settings
from src.agent.claude_agent import SYSTEM_PROMPT, build_system_prompt
from src.agent.posture import PostureLevel, PostureManager
from src.analysis.risk_manager import (
    HEDGE_ALLOWED_REGIMES,
    MAX_INVERSE_POSITIONS,
    MAX_SINGLE_INVERSE_PCT,
    MAX_TOTAL_INVERSE_PCT,
    RiskManager,
)
from src.analysis.weekly_improvement import (
    MAX_LEARNINGS_PER_WEEK,
    MAX_TICKER_EXCLUSIONS_PER_WEEK,
    TRAILING_STOP_MULTIPLIER_MAX,
    TRAILING_STOP_MULTIPLIER_MIN,
)
from src.storage.models import OrderSide, PortfolioName, TradeSchema


def _make_trade(ticker: str = "XLK", side: str = "BUY", shares: float = 10, price: float = 200.0) -> TradeSchema:
    return TradeSchema(
        ticker=ticker, side=OrderSide(side), shares=shares, price=price, portfolio=PortfolioName.B, reason="test"
    )


# ── Batch 1: Telegram Approval ──


class TestApprovalDisabled:
    def test_settings_approval_disabled_by_default(self) -> None:
        assert settings.trade_approval_enabled is False

    def test_inverse_buy_no_approval_flag(self) -> None:
        """Inverse BUY should not set requires_approval."""
        rm = RiskManager()
        trade = _make_trade("SH", "BUY", 5, 30.0)
        verdict = rm.check_pre_trade(
            trades=[trade],
            portfolio_name="B",
            current_positions={},
            latest_prices={"SH": 30.0},
            portfolio_value=100_000,
            cash=100_000,
            regime="CORRECTION",
            current_posture="defensive",
        )
        assert len(verdict.requires_approval) == 0
        assert len(verdict.allowed) == 1

    def test_large_trade_no_approval_when_disabled(self) -> None:
        """Large trade (>10% of portfolio) should not require approval when disabled."""
        rm = RiskManager()
        trade = _make_trade("AAPL", "BUY", 100, 180.0)  # $18K = 18% of $100K
        verdict = rm.check_pre_trade(
            trades=[trade],
            portfolio_name="B",
            current_positions={},
            latest_prices={"AAPL": 180.0},
            portfolio_value=100_000,
            cash=100_000,
        )
        assert len(verdict.requires_trade_approval) == 0
        assert len(verdict.allowed) == 1


# ── Batch 2: Circuit Breakers + Limits ──


class TestRelaxedLimits:
    def test_daily_circuit_breaker_at_15pct(self) -> None:
        assert RISK_RULES.daily_loss_limit_pct == 0.15

    def test_weekly_circuit_breaker_at_30pct(self) -> None:
        assert RISK_RULES.weekly_loss_limit_pct == 0.30

    def test_max_single_position_50pct(self) -> None:
        assert RISK_RULES.max_single_position_pct == 0.50

    def test_crypto_sector_cap_30pct(self) -> None:
        assert RISK_RULES.sector_concentration_overrides["Crypto"] == 0.30

    def test_hedge_sector_cap_30pct(self) -> None:
        assert RISK_RULES.sector_concentration_overrides["Hedge"] == 0.30

    def test_tech_weight_60pct(self) -> None:
        assert RISK_RULES.max_tech_weight == 0.60

    def test_11th_position_allowed(self) -> None:
        """With 10 existing positions, an 11th should still pass."""
        rm = RiskManager()
        existing = {f"TICK{i}": 100.0 for i in range(10)}
        prices = {f"TICK{i}": 100.0 for i in range(10)}
        prices["NVDA"] = 100.0
        trade = _make_trade("NVDA", "BUY", 10, 100.0)
        verdict = rm.check_pre_trade(
            trades=[trade],
            portfolio_name="B",
            current_positions=existing,
            latest_prices=prices,
            portfolio_value=100_000,
            cash=50_000,
        )
        assert len(verdict.allowed) == 1


# ── Batch 3: Inverse ETFs + Posture ──


class TestRelaxedInverse:
    def test_bear_regime_allows_inverse(self) -> None:
        assert "BEAR" in HEDGE_ALLOWED_REGIMES

    def test_inverse_no_posture_gate(self) -> None:
        """Inverse ETF should be allowed in balanced posture (posture gate removed)."""
        rm = RiskManager()
        trade = _make_trade("SH", "BUY", 5, 30.0)
        verdict = rm.check_pre_trade(
            trades=[trade],
            portfolio_name="B",
            current_positions={},
            latest_prices={"SH": 30.0},
            portfolio_value=100_000,
            cash=100_000,
            regime="CORRECTION",
            current_posture="balanced",  # Was blocked before
        )
        assert len(verdict.allowed) == 1
        assert len(verdict.blocked) == 0

    def test_max_single_inverse_20pct(self) -> None:
        assert MAX_SINGLE_INVERSE_PCT == 0.20

    def test_max_total_inverse_30pct(self) -> None:
        assert MAX_TOTAL_INVERSE_PCT == 0.30

    def test_max_4_inverse_positions(self) -> None:
        assert MAX_INVERSE_POSITIONS == 4


class TestUnlockedPosture:
    def test_aggressive_no_gate(self) -> None:
        """Aggressive posture should be granted without track record."""
        mgr = PostureManager()
        limits, effective = mgr.resolve_effective_limits(
            PostureLevel.AGGRESSIVE,
            total_trades=0,
            win_rate_pct=0.0,
            avg_alpha_vs_spy=None,
        )
        assert effective == PostureLevel.AGGRESSIVE

    def test_all_postures_valid(self) -> None:
        mgr = PostureManager()
        for level in PostureLevel:
            limits, effective = mgr.resolve_effective_limits(level)
            assert effective == level


# ── Batch 4: Strategy Directives ──


class TestStrategyFreedom:
    def test_conservative_directive_no_hard_caps(self) -> None:
        from src.agent.strategy_directives import CONSERVATIVE_DIRECTIVE

        assert "AT LEAST 40%" not in CONSERVATIVE_DIRECTIVE
        assert "no exceptions" not in CONSERVATIVE_DIRECTIVE
        assert "GUIDELINES" in CONSERVATIVE_DIRECTIVE or "guidelines" in CONSERVATIVE_DIRECTIVE.lower()

    def test_aggressive_directive_encourages_experimentation(self) -> None:
        from src.agent.strategy_directives import AGGRESSIVE_DIRECTIVE

        assert "paper trading" in AGGRESSIVE_DIRECTIVE.lower()
        assert "experiment" in AGGRESSIVE_DIRECTIVE.lower()

    def test_round_trip_not_restricted(self) -> None:
        prompt = build_system_prompt()
        assert "round-tripping" not in prompt
        assert "don't sell and rebuy" not in prompt


# ── Batch 5: System Prompt ──


class TestPromptFreedom:
    def test_prompt_mentions_paper_trading(self) -> None:
        prompt = build_system_prompt()
        assert "PAPER TRADING" in prompt

    def test_prompt_mentions_learning_velocity(self) -> None:
        prompt = build_system_prompt()
        assert "LEARNING VELOCITY" in prompt

    def test_prompt_no_hard_allocation_mandates(self) -> None:
        prompt = build_system_prompt()
        assert "at least 20% cash or inverse" not in prompt
        assert "Maximum 10 positions" not in prompt

    def test_prompt_positions_limit_is_20(self) -> None:
        assert "20 positions" in SYSTEM_PROMPT

    def test_prompt_position_max_is_50pct(self) -> None:
        assert "50%" in SYSTEM_PROMPT

    def test_prompt_inverse_mentions_bear(self) -> None:
        prompt = build_system_prompt()
        assert "BEAR" in prompt


# ── Batch 5b: Self-Improvement Bounds ──


class TestImprovementBounds:
    def test_trailing_stop_range_widened(self) -> None:
        assert TRAILING_STOP_MULTIPLIER_MIN == 0.5
        assert TRAILING_STOP_MULTIPLIER_MAX == 3.0

    def test_max_exclusions_widened(self) -> None:
        assert MAX_TICKER_EXCLUSIONS_PER_WEEK == 5

    def test_max_learnings_widened(self) -> None:
        assert MAX_LEARNINGS_PER_WEEK == 5

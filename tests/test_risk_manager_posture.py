"""Tests for posture_limits parameter on RiskManager.check_pre_trade."""

from config.risk_rules import RiskRules
from src.agent.posture import PostureLimits
from src.analysis.risk_manager import RiskManager
from src.storage.models import OrderSide, PortfolioName, TradeSchema

# ── Helpers ──────────────────────────────────────────────────────────────


def _make_trade(
    ticker: str,
    side: str,
    shares: float,
    price: float,
    portfolio: str = "B",
) -> TradeSchema:
    return TradeSchema(
        portfolio=PortfolioName(portfolio),
        ticker=ticker,
        side=OrderSide(side),
        shares=shares,
        price=price,
        reason="test",
    )


# ── Tests ────────────────────────────────────────────────────────────────


def test_posture_limits_none_unchanged() -> None:
    """With posture_limits=None, default 35% single-position limit applies.

    A 30% position should pass.
    """
    rm = RiskManager()
    # AAPL at $150 * 200 shares = $30,000 = 30% of $100,000
    trade = _make_trade("AAPL", "BUY", 200, 150.0)
    verdict = rm.check_pre_trade(
        trades=[trade],
        portfolio_name="B",
        current_positions={},
        latest_prices={"AAPL": 150.0},
        portfolio_value=100_000.0,
        cash=100_000.0,
        posture_limits=None,
    )
    assert len(verdict.allowed) == 1
    assert len(verdict.blocked) == 0


def test_defensive_posture_tightens_position_limit() -> None:
    """Defensive posture (max_single=0.25) blocks a 30% position."""
    rm = RiskManager()
    defensive = PostureLimits(
        max_single_position_pct=0.25,
        max_sector_concentration=0.35,
        max_equity_pct=0.50,
    )
    # AAPL at $150 * 200 shares = $30,000 = 30% of $100,000 — exceeds 25%
    trade = _make_trade("AAPL", "BUY", 200, 150.0)
    verdict = rm.check_pre_trade(
        trades=[trade],
        portfolio_name="B",
        current_positions={},
        latest_prices={"AAPL": 150.0},
        portfolio_value=100_000.0,
        cash=100_000.0,
        posture_limits=defensive,
    )
    assert len(verdict.blocked) == 1
    assert len(verdict.allowed) == 0
    assert "AAPL" in verdict.blocked[0][1]


def test_crisis_posture_blocks_large_trade() -> None:
    """Crisis posture (max_single=0.15) blocks even a 20% position."""
    rm = RiskManager()
    crisis = PostureLimits(
        max_single_position_pct=0.15,
        max_sector_concentration=0.25,
        max_equity_pct=0.30,
    )
    # AAPL at $100 * 200 shares = $20,000 = 20% of $100,000 — exceeds 15%
    trade = _make_trade("AAPL", "BUY", 200, 100.0)
    verdict = rm.check_pre_trade(
        trades=[trade],
        portfolio_name="B",
        current_positions={},
        latest_prices={"AAPL": 100.0},
        portfolio_value=100_000.0,
        cash=100_000.0,
        posture_limits=crisis,
    )
    assert len(verdict.blocked) == 1
    assert len(verdict.allowed) == 0


def test_posture_cannot_loosen_beyond_rules() -> None:
    """Posture limits cannot loosen beyond the RiskRules base limits.

    RiskRules has max_single=0.20, PostureLimits has 0.35.
    Effective should be min(0.20, 0.35) = 0.20. A 25% trade should be blocked.
    """
    rules = RiskRules(max_single_position_pct=0.20, max_sector_concentration=0.30)
    rm = RiskManager(rules=rules)
    loose_posture = PostureLimits(
        max_single_position_pct=0.35,
        max_sector_concentration=0.50,
        max_equity_pct=0.95,
    )
    # AAPL at $125 * 200 shares = $25,000 = 25% of $100,000 — exceeds min(0.20, 0.35) = 0.20
    trade = _make_trade("AAPL", "BUY", 200, 125.0)
    verdict = rm.check_pre_trade(
        trades=[trade],
        portfolio_name="B",
        current_positions={},
        latest_prices={"AAPL": 125.0},
        portfolio_value=100_000.0,
        cash=100_000.0,
        posture_limits=loose_posture,
    )
    assert len(verdict.blocked) == 1
    assert len(verdict.allowed) == 0


def test_sells_always_pass_with_posture() -> None:
    """SELL trades pass regardless of posture limits."""
    rm = RiskManager()
    crisis = PostureLimits(
        max_single_position_pct=0.15,
        max_sector_concentration=0.25,
        max_equity_pct=0.30,
    )
    trade = _make_trade("AAPL", "SELL", 200, 150.0)
    verdict = rm.check_pre_trade(
        trades=[trade],
        portfolio_name="B",
        current_positions={"AAPL": 200},
        latest_prices={"AAPL": 150.0},
        portfolio_value=100_000.0,
        cash=70_000.0,
        posture_limits=crisis,
    )
    assert len(verdict.allowed) == 1
    assert len(verdict.blocked) == 0

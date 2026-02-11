"""Strategy parameters for portfolios A and B."""

from dataclasses import dataclass


@dataclass(frozen=True)
class PortfolioAConfig:
    """Aggressive Momentum — top 1 ETF, daily rebalance."""

    name: str = "Aggressive Momentum"
    allocation_usd: float = 33_000.0
    top_n: int = 1
    momentum_lookback_days: int = 63  # ~3 months
    momentum_skip_days: int = 5  # skip last week (mean reversion filter)
    rebalance_frequency: str = "daily"
    use_stop_loss: bool = False


@dataclass(frozen=True)
class PortfolioBConfig:
    """AI Full Autonomy — Claude decides everything."""

    name: str = "AI Full Autonomy"
    allocation_usd: float = 66_000.0
    model: str = "claude-opus-4-6"
    max_positions: int = 10
    max_single_position_pct: float = 0.30  # no single position > 30%
    rebalance_frequency: str = "daily"
    # TODO: If keeping Opus as default, remove the escalation flow entirely
    # (escalation_model, escalation_threshold, approval_timeout_seconds,
    # ComplexityDetector, _request_model_approval in orchestrator.py,
    # and the Telegram approval keyboard).
    escalation_model: str = "claude-opus-4-6"
    escalation_threshold: int = 50
    approval_timeout_seconds: int = 300


PORTFOLIO_A = PortfolioAConfig()
PORTFOLIO_B = PortfolioBConfig()

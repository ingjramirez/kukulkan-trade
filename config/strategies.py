"""Strategy parameters for portfolios A, B, and C."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class PortfolioAConfig:
    """Aggressive Momentum — top 1 ETF, daily rebalance."""

    name: str = "Aggressive Momentum"
    allocation_usd: float = 33_333.0
    top_n: int = 1
    momentum_lookback_days: int = 63  # ~3 months
    momentum_skip_days: int = 5  # skip last week (mean reversion filter)
    rebalance_frequency: str = "daily"
    use_stop_loss: bool = False


@dataclass(frozen=True)
class PortfolioBConfig:
    """Sector Rotation + Macro + Contrarian — 7-factor composite score."""

    name: str = "Sector Rotation"
    allocation_usd: float = 33_333.0
    top_n: int = 3  # hold top 3 positions
    rebalance_frequency: str = "weekly"

    # Composite score weights (must sum to 1.0)
    weight_momentum: float = 0.20
    weight_rsi_contrarian: float = 0.15
    weight_macro_regime: float = 0.15
    weight_volume_breakout: float = 0.10
    weight_value_tilt: float = 0.15
    weight_crowding: float = 0.10
    weight_btc_risk: float = 0.15

    # Regime detection
    regimes: list[str] = field(
        default_factory=lambda: ["BULL", "ROTATION", "NEUTRAL", "BEAR"]
    )

    # Risk rules
    max_tech_allocation: float = 0.40  # anti-tech-bubble
    min_defensive_in_bear: float = 0.30


@dataclass(frozen=True)
class PortfolioCConfig:
    """AI Full Autonomy — Claude decides everything."""

    name: str = "AI Full Autonomy"
    allocation_usd: float = 33_333.0
    model: str = "claude-sonnet-4-5-20250929"
    max_positions: int = 10
    max_single_position_pct: float = 0.30  # no single position > 30%
    rebalance_frequency: str = "daily"


PORTFOLIO_A = PortfolioAConfig()
PORTFOLIO_B = PortfolioBConfig()
PORTFOLIO_C = PortfolioCConfig()

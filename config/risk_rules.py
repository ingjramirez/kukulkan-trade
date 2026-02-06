"""Global risk rules applied across all portfolios."""

from dataclasses import dataclass


@dataclass(frozen=True)
class RiskRules:
    """Hard risk limits that override strategy decisions."""

    # Position limits
    max_single_position_pct: float = 0.35  # max 35% in any single ticker
    max_sector_concentration: float = 0.50  # max 50% in one sector

    # Drawdown circuit breakers
    daily_loss_limit_pct: float = 0.05  # halt trading if portfolio drops 5% in a day
    weekly_loss_limit_pct: float = 0.10  # halt trading if portfolio drops 10% in a week

    # Anti-tech-bubble rules (Portfolio B)
    tech_etfs: tuple[str, ...] = ("XLK", "QQQ", "SMH", "ARKK")
    max_tech_weight: float = 0.40

    # Defensive assets
    defensive_tickers: tuple[str, ...] = ("XLU", "XLP", "TLT", "GLD", "SH")

    # BTC risk signal thresholds
    btc_proxy: str = "IBIT"
    btc_crash_threshold_pct: float = -0.20  # BTC down 20% = risk-off signal

    # Minimum cash buffer (not currently enforced in paper trading)
    min_cash_pct: float = 0.02


RISK_RULES = RiskRules()

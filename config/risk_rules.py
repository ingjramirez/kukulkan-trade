"""Global risk rules applied across all portfolios."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class RiskRules:
    """Hard risk limits that override strategy decisions."""

    # Position limits
    max_single_position_pct: float = 0.35  # max 35% in any single ticker
    max_sector_concentration: float = 0.50  # max 50% in one sector (default fallback)

    # Per-sector concentration overrides (sector name -> max fraction)
    # Sectors not listed here fall back to max_sector_concentration.
    sector_concentration_overrides: dict[str, float] = field(
        default_factory=lambda: {
            "Fixed Income": 0.50,
            "International": 0.25,
            "Dividend/Value": 0.25,
            "Hedge": 0.05,
            "Thematic": 0.10,
            "Commodities": 0.20,
            "Real Estate": 0.15,
            "Crypto": 0.05,
        }
    )

    # Drawdown circuit breakers
    daily_loss_limit_pct: float = 0.05  # halt trading if portfolio drops 5% in a day
    weekly_loss_limit_pct: float = 0.10  # halt trading if portfolio drops 10% in a week

    # Anti-tech-bubble rules (Portfolio B)
    tech_etfs: tuple[str, ...] = ("XLK", "QQQ", "SMH", "ARKK")
    max_tech_weight: float = 0.40

    # Defensive assets
    defensive_tickers: tuple[str, ...] = (
        "XLU",
        "XLP",
        "TLT",
        "GLD",
        "SH",
        "BIL",
        "SHY",
        "IEF",
        "AGG",
        "VTIP",
    )

    # BTC risk signal thresholds
    btc_proxy: str = "IBIT"
    btc_crash_threshold_pct: float = -0.20  # BTC down 20% = risk-off signal

    # Minimum cash buffer (not currently enforced in paper trading)
    min_cash_pct: float = 0.02


RISK_RULES = RiskRules()

# Trailing stop percentages by strategy mode and conviction level.
# Lower trail = tighter stop = less drawdown tolerance.
TRAIL_PCT: dict[str, dict[str, float]] = {
    "conservative": {"high": 0.05, "medium": 0.07, "low": 0.10},
    "standard": {"high": 0.07, "medium": 0.10, "low": 0.12},
    "aggressive": {"high": 0.10, "medium": 0.12, "low": 0.15},
}

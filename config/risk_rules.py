"""Global risk rules applied across all portfolios."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class RiskRules:
    """Hard risk limits that override strategy decisions."""

    # Position limits (relaxed for paper trading — learning > preservation)
    max_single_position_pct: float = 0.50  # max 50% in any single ticker
    max_sector_concentration: float = 0.50  # max 50% in one sector (default fallback)

    # Per-sector concentration overrides (sector name -> max fraction)
    # Sectors not listed here fall back to max_sector_concentration.
    sector_concentration_overrides: dict[str, float] = field(
        default_factory=lambda: {
            "Fixed Income": 0.50,
            "International": 0.40,
            "Dividend/Value": 0.40,
            "Hedge": 0.30,
            "Thematic": 0.30,
            "Commodities": 0.30,
            "Real Estate": 0.30,
            "Crypto": 0.30,
        }
    )

    # Drawdown circuit breakers (widened for paper trading)
    daily_loss_limit_pct: float = 0.15  # halt trading if portfolio drops 15% in a day
    weekly_loss_limit_pct: float = 0.30  # halt trading if portfolio drops 30% in a week

    # Anti-tech-bubble rules (Portfolio B)
    tech_etfs: tuple[str, ...] = ("XLK", "QQQ", "SMH", "ARKK")
    max_tech_weight: float = 0.60

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
    btc_ticker: str = "BTC-USD"  # Real BTC for crash detection
    btc_proxy: str = "IBIT"  # ETF proxy for correlation
    btc_crash_threshold_pct: float = -0.20  # BTC down 20% = risk-off signal

    # Max simultaneous positions (paper trading — permissive)
    max_positions: int = 20

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

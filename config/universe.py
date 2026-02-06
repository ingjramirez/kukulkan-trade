"""Ticker universe organized by category and portfolio access.

Portfolio A (Aggressive Momentum): SECTOR_ETFS + THEMATIC_ETFS only
Portfolio B (AI Full Autonomy): Full ETF universe + individual stocks
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.storage.database import Database

# --- Core Sector ETFs (SPDR Select Sector) ---
SECTOR_ETFS: list[str] = [
    "XLK",  # Technology
    "XLF",  # Financials
    "XLV",  # Health Care
    "XLE",  # Energy
    "XLI",  # Industrials
    "XLY",  # Consumer Discretionary
    "XLP",  # Consumer Staples
    "XLU",  # Utilities
    "XLB",  # Materials
    "XLRE",  # Real Estate
]

# --- Thematic / Factor ETFs ---
THEMATIC_ETFS: list[str] = [
    "QQQ",  # Nasdaq 100
    "SMH",  # Semiconductors
    "XBI",  # Biotech
    "IWM",  # Russell 2000 (small caps)
    "EFA",  # International developed
    "EEM",  # Emerging markets
    "TLT",  # 20+ Year Treasury
    "HYG",  # High yield corporate bonds
    "GDX",  # Gold miners
    "ARKK",  # ARK Innovation
]

# --- Inverse / Hedge ETFs ---
INVERSE_ETFS: list[str] = [
    "SH",  # Short S&P 500
    "PSQ",  # Short QQQ
    "TBF",  # Short 20+ Year Treasury
]

# --- Commodities ---
COMMODITY_ETFS: list[str] = [
    "GLD",  # Gold
    "SLV",  # Silver
    "USO",  # Oil
]

# --- Individual Stocks ---
INDIVIDUAL_STOCKS: list[str] = [
    "AAPL",
    "MSFT",
    "GOOGL",
    "AMZN",
    "NVDA",
    "META",
    "TSLA",
    "JPM",
    "V",
    "UNH",
    "XOM",
    "LLY",
    "JNJ",
    "PG",
    "HD",
    "COST",
    "NFLX",
    "AMD",
    "CRM",
    "BA",
]

# --- Crypto proxy ---
CRYPTO: list[str] = [
    "IBIT",  # iShares Bitcoin Trust
]

# --- Portfolio universes ---
PORTFOLIO_A_UNIVERSE: list[str] = SECTOR_ETFS + THEMATIC_ETFS
PORTFOLIO_B_UNIVERSE: list[str] = (
    SECTOR_ETFS + THEMATIC_ETFS + INVERSE_ETFS + COMMODITY_ETFS + INDIVIDUAL_STOCKS + CRYPTO
)

# Full universe (for data fetching)
FULL_UNIVERSE: list[str] = sorted(set(PORTFOLIO_B_UNIVERSE))

# --- Sector classification for risk management ---
SECTOR_MAP: dict[str, str] = {
    "XLK": "Technology", "QQQ": "Technology", "SMH": "Technology",
    "ARKK": "Technology", "AAPL": "Technology", "MSFT": "Technology",
    "GOOGL": "Technology", "NVDA": "Technology", "META": "Technology",
    "NFLX": "Technology", "AMD": "Technology", "CRM": "Technology",
    "XLF": "Financials", "JPM": "Financials", "V": "Financials",
    "XLV": "Health Care", "UNH": "Health Care", "LLY": "Health Care",
    "JNJ": "Health Care", "XBI": "Health Care",
    "XLE": "Energy", "XOM": "Energy",
    "XLI": "Industrials", "BA": "Industrials",
    "XLY": "Consumer Discretionary", "AMZN": "Consumer Discretionary",
    "TSLA": "Consumer Discretionary", "HD": "Consumer Discretionary",
    "XLP": "Consumer Staples", "PG": "Consumer Staples", "COST": "Consumer Staples",
    "XLU": "Utilities", "XLB": "Materials", "GDX": "Materials",
    "XLRE": "Real Estate",
    "IWM": "Broad Market", "EFA": "International", "EEM": "International",
    "TLT": "Fixed Income", "HYG": "Fixed Income",
    "SH": "Inverse", "PSQ": "Inverse", "TBF": "Inverse",
    "GLD": "Commodities", "SLV": "Commodities", "USO": "Commodities",
    "IBIT": "Crypto",
}


async def get_dynamic_universe(db: "Database") -> list[str]:
    """Get the full universe including approved dynamic tickers.

    Merges the static FULL_UNIVERSE with any approved discovered tickers
    from the database.

    Args:
        db: Database instance for querying discovered tickers.

    Returns:
        Sorted, deduplicated list of all active tickers.
    """

    approved = await db.get_approved_tickers()
    dynamic = [r.ticker for r in approved]
    return sorted(set(FULL_UNIVERSE + dynamic))

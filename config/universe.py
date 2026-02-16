"""Ticker universe organized by category and portfolio access.

Portfolio A (Aggressive Momentum): SECTOR_ETFS + THEMATIC_ETFS only
Portfolio B (AI Full Autonomy): Full ETF universe + individual stocks
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.storage.database import Database


class InstrumentType(str, Enum):
    """Classification of tradeable instruments."""

    STOCK = "stock"
    ETF = "etf"
    INVERSE_ETF = "inverse_etf"
    CRYPTO_PROXY = "crypto_proxy"


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
INVERSE_ETF_META: dict[str, dict] = {
    "SH": {"benchmark": "SPY", "leverage": 1, "description": "Short S&P 500", "equity_hedge": True},
    "PSQ": {"benchmark": "QQQ", "leverage": 1, "description": "Short Nasdaq 100", "equity_hedge": True},
    "RWM": {"benchmark": "IWM", "leverage": 1, "description": "Short Russell 2000", "equity_hedge": True},
    "TBF": {"benchmark": "TLT", "leverage": 1, "description": "Short 20+ Year Treasury", "equity_hedge": False},
}
INVERSE_ETFS: list[str] = list(INVERSE_ETF_META.keys())

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

# --- Fixed Income ETFs (bond ladder) ---
FIXED_INCOME_ETFS: list[str] = ["BIL", "SHY", "IEF", "AGG", "VTIP"]
# Note: TLT already in THEMATIC_ETFS, HYG already in THEMATIC_ETFS

# --- International ETFs ---
INTERNATIONAL_ETFS: list[str] = ["VEA", "VWO", "IXUS", "FXI", "KWEB", "INDA", "VGK"]
# Note: EFA already in THEMATIC_ETFS, EEM already in THEMATIC_ETFS

# --- Real Estate ETFs ---
REAL_ESTATE_ETFS: list[str] = ["VNQ", "SCHH"]
# Note: XLRE already in SECTOR_ETFS

# --- Dividend / Value ETFs ---
DIVIDEND_VALUE_ETFS: list[str] = ["SCHD", "VTV", "DVY"]

# --- Additional Individual Stocks ---
ADDITIONAL_STOCKS: list[str] = ["AVGO", "KO"]
# Note: UNH, PG, XOM already in INDIVIDUAL_STOCKS

# --- New Thematic ETFs ---
THEMATIC_NEW: list[str] = ["ICLN", "DBC", "PDBC", "VIXY"]
# Commodities (DBC, PDBC), Clean Energy (ICLN), Volatility Hedge (VIXY)

# --- Benchmark tickers (for regime classification & SPY benchmarking) ---
BENCHMARK_TICKERS: list[str] = ["SPY"]

# --- Portfolio universes ---
PORTFOLIO_A_UNIVERSE: list[str] = SECTOR_ETFS + THEMATIC_ETFS
PORTFOLIO_B_UNIVERSE: list[str] = (
    SECTOR_ETFS
    + THEMATIC_ETFS
    + INVERSE_ETFS
    + COMMODITY_ETFS
    + INDIVIDUAL_STOCKS
    + CRYPTO
    + FIXED_INCOME_ETFS
    + INTERNATIONAL_ETFS
    + REAL_ESTATE_ETFS
    + DIVIDEND_VALUE_ETFS
    + ADDITIONAL_STOCKS
    + THEMATIC_NEW
)

# Full universe (for data fetching)
FULL_UNIVERSE: list[str] = sorted(set(PORTFOLIO_B_UNIVERSE))

# --- Sector classification for risk management ---
SECTOR_MAP: dict[str, str] = {
    "XLK": "Technology",
    "QQQ": "Technology",
    "SMH": "Technology",
    "ARKK": "Technology",
    "AAPL": "Technology",
    "MSFT": "Technology",
    "GOOGL": "Technology",
    "NVDA": "Technology",
    "META": "Technology",
    "NFLX": "Technology",
    "AMD": "Technology",
    "CRM": "Technology",
    "XLF": "Financials",
    "JPM": "Financials",
    "V": "Financials",
    "XLV": "Health Care",
    "UNH": "Health Care",
    "LLY": "Health Care",
    "JNJ": "Health Care",
    "XBI": "Health Care",
    "XLE": "Energy",
    "XOM": "Energy",
    "XLI": "Industrials",
    "BA": "Industrials",
    "XLY": "Consumer Discretionary",
    "AMZN": "Consumer Discretionary",
    "TSLA": "Consumer Discretionary",
    "HD": "Consumer Discretionary",
    "XLP": "Consumer Staples",
    "PG": "Consumer Staples",
    "COST": "Consumer Staples",
    "XLU": "Utilities",
    "XLB": "Materials",
    "GDX": "Materials",
    "XLRE": "Real Estate",
    "IWM": "Broad Market",
    "EFA": "International",
    "EEM": "International",
    "TLT": "Fixed Income",
    "HYG": "Fixed Income",
    "SH": "Inverse",
    "PSQ": "Inverse",
    "RWM": "Inverse",
    "TBF": "Inverse",
    "GLD": "Commodities",
    "SLV": "Commodities",
    "USO": "Commodities",
    "IBIT": "Crypto",
    # Fixed Income (new)
    "BIL": "Fixed Income",
    "SHY": "Fixed Income",
    "IEF": "Fixed Income",
    "AGG": "Fixed Income",
    "VTIP": "Fixed Income",
    # International (new)
    "VEA": "International",
    "VWO": "International",
    "IXUS": "International",
    "FXI": "International",
    "KWEB": "International",
    "INDA": "International",
    "VGK": "International",
    # Real Estate (new)
    "VNQ": "Real Estate",
    "SCHH": "Real Estate",
    # Dividend/Value (new)
    "SCHD": "Dividend/Value",
    "VTV": "Dividend/Value",
    "DVY": "Dividend/Value",
    # Technology (new)
    "AVGO": "Technology",
    # Consumer Staples (new)
    "KO": "Consumer Staples",
    # Commodities (new)
    "DBC": "Commodities",
    "PDBC": "Commodities",
    # Thematic (new)
    "ICLN": "Thematic",
    # Hedge (new)
    "VIXY": "Hedge",
}


# --- Sector → ETF benchmark map (for outcome tracking) ---
SECTOR_ETF_MAP: dict[str, str] = {
    "Technology": "XLK",
    "Financials": "XLF",
    "Health Care": "XLV",
    "Energy": "XLE",
    "Industrials": "XLI",
    "Consumer Discretionary": "XLY",
    "Consumer Staples": "XLP",
    "Utilities": "XLU",
    "Materials": "XLB",
    "Real Estate": "XLRE",
    "Broad Market": "IWM",
    "International": "EFA",
    "Fixed Income": "TLT",
    "Inverse": "SH",
    "Commodities": "GLD",
    "Crypto": "IBIT",
    "Dividend/Value": "SCHD",
    "Thematic": "ICLN",
    "Hedge": "VIXY",
}


def classify_instrument(ticker: str) -> InstrumentType:
    """Classify a ticker into its instrument type.

    Args:
        ticker: Ticker symbol.

    Returns:
        InstrumentType enum value.
    """
    if ticker in INVERSE_ETF_META:
        return InstrumentType.INVERSE_ETF
    if ticker in CRYPTO:
        return InstrumentType.CRYPTO_PROXY
    if ticker in INDIVIDUAL_STOCKS or ticker in ADDITIONAL_STOCKS:
        return InstrumentType.STOCK
    return InstrumentType.ETF


def is_equity_hedge(ticker: str) -> bool:
    """Check if a ticker is an equity hedge inverse ETF.

    TBF hedges interest rate risk (not equity) so returns False.
    SH/PSQ/RWM hedge equity risk so return True.

    Args:
        ticker: Ticker symbol.

    Returns:
        True if the ticker is an equity-hedging inverse ETF.
    """
    meta = INVERSE_ETF_META.get(ticker)
    if meta is None:
        return False
    return meta["equity_hedge"]


async def get_dynamic_universe(db: "Database") -> list[str]:
    """Get the full universe including approved dynamic tickers from all tenants.

    Merges the static FULL_UNIVERSE with any approved discovered tickers
    from the database (across all tenants — market data is global).

    Args:
        db: Database instance for querying discovered tickers.

    Returns:
        Sorted, deduplicated list of all active tickers.
    """
    approved = await db.get_all_approved_tickers_all_tenants()
    dynamic = [r.ticker for r in approved]
    return sorted(set(FULL_UNIVERSE + dynamic + BENCHMARK_TICKERS))

"""Ticker format mapping between data sources.

Canonical format: yfinance style (BTC-USD) — used in DB, universe, everywhere internal.
Alpaca format: BTC/USD — used only when sending orders to Alpaca.
"""

from __future__ import annotations

# Canonical (yfinance) -> Alpaca format
CRYPTO_TICKER_MAP: dict[str, str] = {
    "BTC-USD": "BTC/USD",
    # Future: "ETH-USD": "ETH/USD", "SOL-USD": "SOL/USD"
}

# Reverse: Alpaca -> Canonical
ALPACA_TO_CANONICAL: dict[str, str] = {v: k for k, v in CRYPTO_TICKER_MAP.items()}


def to_alpaca_format(ticker: str) -> str:
    """Convert canonical ticker to Alpaca format for order submission."""
    return CRYPTO_TICKER_MAP.get(ticker, ticker)


def to_canonical_format(ticker: str) -> str:
    """Convert Alpaca ticker to canonical format for DB/internal use."""
    return ALPACA_TO_CANONICAL.get(ticker, ticker)


def is_crypto_ticker(ticker: str) -> bool:
    """Check if a ticker is a crypto asset (not a proxy ETF like IBIT)."""
    return ticker in CRYPTO_TICKER_MAP or ticker in ALPACA_TO_CANONICAL

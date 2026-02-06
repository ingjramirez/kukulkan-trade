"""Unified news data models used across all news sources.

NewsArticle: standardized article from any source (Alpaca, Finnhub, yfinance).
NewsCluster: group of articles about the same event.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class NewsArticle:
    """Single news article from any source."""

    headline: str
    summary: str
    source: str  # "alpaca", "finnhub", "yfinance"
    publisher: str  # "Benzinga", "Reuters", etc.
    tickers: list[str]
    published_at: datetime | None = None
    url: str = ""
    sentiment: float | None = None  # -1.0 to 1.0 if available


@dataclass
class NewsCluster:
    """Group of articles about the same event."""

    representative: NewsArticle
    source_count: int = 1
    signal: str = "INFO"  # POS, NEG, MACRO, EVENT, INFO
    score: int = 0
    all_tickers: list[str] = field(default_factory=list)

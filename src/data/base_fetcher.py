"""Abstract base class for all news fetchers.

All fetchers are sync (no async) to match the existing pattern.
The aggregator calls them in sequence from its sync fetch_all().
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod

from src.data.news_article import NewsArticle

# Common uppercase words that could false-match ticker symbols in news text
_TICKER_STOPWORDS: set[str] = {
    "CEO", "IPO", "GDP", "USA", "FBI", "SEC", "ETF", "API", "CPI", "PMI", "NYSE", "FOMC",
}


class BaseNewsFetcher(ABC):
    """Base class for news source fetchers."""

    source_name: str = "unknown"
    region: str = "us"

    @abstractmethod
    def fetch(self, tickers: list[str] | None = None) -> list[NewsArticle]:
        """Fetch articles from this source.

        Args:
            tickers: Optional list of ticker symbols to filter for.

        Returns:
            List of NewsArticle objects.
        """

    def _extract_tickers(self, text: str, universe: set[str]) -> list[str]:
        """Extract ticker symbols from text by matching against a universe set.

        Looks for $TICKER cashtag patterns and uppercase words that match
        known tickers in the universe.

        Args:
            text: Text to scan for ticker mentions.
            universe: Set of valid ticker symbols to match against.

        Returns:
            Deduplicated list of matched ticker symbols.
        """
        found: set[str] = set()

        # $TICKER cashtag pattern
        for match in re.findall(r"\$([A-Z]{1,5})", text):
            if match in universe:
                found.add(match)

        # Uppercase words that match universe tickers (min 2 chars to avoid noise)
        for word in re.findall(r"\b([A-Z]{2,5})\b", text):
            if word in universe and word not in _TICKER_STOPWORDS:
                found.add(word)

        return sorted(found)

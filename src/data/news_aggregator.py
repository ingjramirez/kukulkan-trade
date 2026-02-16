"""News aggregator combining Alpaca, Finnhub, and yfinance sources.

Priority order: Alpaca (highest quality) → Finnhub → yfinance (fallback).
Deduplicates by headline word overlap (>50% = same story).
"""

from __future__ import annotations

from datetime import datetime

import structlog
import yfinance as yf

from src.data.alpaca_news import AlpacaNewsFetcher
from src.data.finnhub_news import FinnhubNewsFetcher
from src.data.news_article import NewsArticle
from src.utils.retry import retry_news_api

log = structlog.get_logger()

# Words to ignore when comparing headlines for dedup
_STOP_WORDS = frozenset(
    {
        "the",
        "a",
        "an",
        "is",
        "are",
        "was",
        "were",
        "has",
        "have",
        "had",
        "that",
        "this",
        "with",
        "for",
        "its",
        "says",
        "in",
        "of",
        "on",
        "to",
        "at",
        "by",
        "and",
        "or",
        "but",
        "be",
        "been",
        "will",
        "from",
        "as",
    }
)


def _headline_words(headline: str) -> set[str]:
    """Extract significant words from a headline (lowercase, no stop words)."""
    return {w for w in headline.lower().split() if w not in _STOP_WORDS and len(w) > 1}


def _headlines_overlap(h1: str, h2: str, threshold: float = 0.50) -> bool:
    """Check if two headlines are about the same story.

    Args:
        h1: First headline.
        h2: Second headline.
        threshold: Minimum word overlap ratio to consider same story.

    Returns:
        True if the headlines overlap above the threshold.
    """
    words1 = _headline_words(h1)
    words2 = _headline_words(h2)
    if not words1 or not words2:
        return False
    smaller = min(len(words1), len(words2))
    if smaller == 0:
        return False
    overlap = len(words1 & words2)
    return (overlap / smaller) >= threshold


class NewsAggregator:
    """Fetches news from all sources in priority order with deduplication."""

    def __init__(
        self,
        alpaca_fetcher: AlpacaNewsFetcher | None = None,
        finnhub_fetcher: FinnhubNewsFetcher | None = None,
    ) -> None:
        self._alpaca = alpaca_fetcher or AlpacaNewsFetcher()
        self._finnhub = finnhub_fetcher or FinnhubNewsFetcher()

    def fetch_all(
        self,
        tickers: list[str],
        max_articles: int = 100,
    ) -> list[NewsArticle]:
        """Fetch from all sources in priority order, deduplicate.

        Order: 1. Alpaca → 2. Finnhub → 3. yfinance fallback.
        Deduplicates by headline word overlap.

        Args:
            tickers: List of ticker symbols to fetch news for.
            max_articles: Maximum total articles to return.

        Returns:
            Deduplicated list of NewsArticle objects.
        """
        all_articles: list[NewsArticle] = []

        # Layer 1: Alpaca
        alpaca_articles = self._alpaca.fetch(tickers)
        all_articles.extend(alpaca_articles)

        # Layer 2: Finnhub
        finnhub_articles = self._finnhub.fetch(tickers)
        all_articles.extend(finnhub_articles)

        # Layer 3: yfinance fallback (only if layers 1+2 got fewer than 10)
        if len(all_articles) < 10:
            yf_articles = self._fetch_yfinance(tickers)
            all_articles.extend(yf_articles)

        # Deduplicate
        deduped = self._deduplicate(all_articles)

        log.info(
            "news_aggregated",
            raw=len(all_articles),
            deduped=len(deduped),
            alpaca=len(alpaca_articles),
            finnhub=len(finnhub_articles),
        )

        return deduped[:max_articles]

    @staticmethod
    @retry_news_api
    def _fetch_yf_ticker_news(ticker: str) -> list[dict]:
        """Fetch news for a single ticker from yfinance (with retry)."""
        t = yf.Ticker(ticker)
        return t.news or []

    def _fetch_yfinance(self, tickers: list[str]) -> list[NewsArticle]:
        """Fetch from yfinance as a fallback source.

        Args:
            tickers: Ticker symbols to fetch.

        Returns:
            List of NewsArticle objects.
        """
        articles: list[NewsArticle] = []
        for ticker in tickers[:15]:
            try:
                news = self._fetch_yf_ticker_news(ticker)
                for item in news[:3]:
                    title = item.get("title", "")
                    if not title:
                        continue
                    published = None
                    ts = item.get("providerPublishTime")
                    if ts:
                        try:
                            published = datetime.fromtimestamp(ts)
                        except (ValueError, OSError):
                            pass
                    articles.append(
                        NewsArticle(
                            headline=title,
                            summary="",
                            source="yfinance",
                            publisher=item.get("publisher", ""),
                            tickers=[ticker],
                            published_at=published,
                            url=item.get("link", ""),
                        )
                    )
            except Exception as e:
                log.debug("yfinance_news_failed", ticker=ticker, error=str(e))
        return articles

    def _deduplicate(self, articles: list[NewsArticle]) -> list[NewsArticle]:
        """Remove duplicate stories by headline word overlap.

        Keeps the first occurrence (higher-priority source).
        Also tracks how many sources reported the same story.

        Args:
            articles: Raw list from all sources.

        Returns:
            Deduplicated list.
        """
        kept: list[NewsArticle] = []
        for article in articles:
            if not article.headline:
                continue
            is_dup = False
            for existing in kept:
                if _headlines_overlap(article.headline, existing.headline):
                    # Merge tickers from duplicate
                    merged = set(existing.tickers + article.tickers)
                    existing.tickers = list(merged)
                    is_dup = True
                    break
            if not is_dup:
                kept.append(article)
        return kept

"""Alpaca News API fetcher (Benzinga-sourced headlines).

Uses the alpaca-py library already installed. Requires ALPACA_API_KEY
and ALPACA_SECRET_KEY set in .env (same keys used for trading).
"""

from __future__ import annotations

from datetime import datetime

import structlog

from config.settings import settings
from src.data.news_article import NewsArticle

log = structlog.get_logger()


class AlpacaNewsFetcher:
    """Fetches news from Alpaca's data API (Benzinga)."""

    def __init__(
        self,
        api_key: str | None = None,
        secret_key: str | None = None,
    ) -> None:
        self._api_key = api_key or settings.alpaca.api_key
        self._secret_key = secret_key or settings.alpaca.secret_key
        self._client = None

    def _get_client(self):
        """Lazy-init Alpaca REST client for news."""
        if self._client is None:
            from alpaca.data.historical.news import NewsClient
            from alpaca.data.requests import NewsRequest  # noqa: F401

            self._client = NewsClient(
                api_key=self._api_key,
                secret_key=self._secret_key,
            )
        return self._client

    def fetch(
        self,
        tickers: list[str],
        limit: int = 50,
    ) -> list[NewsArticle]:
        """Fetch news articles from Alpaca News API.

        Args:
            tickers: List of ticker symbols to query.
            limit: Maximum articles to return.

        Returns:
            List of NewsArticle objects.
        """
        if not self._api_key or not self._secret_key:
            log.debug("alpaca_news_skipped_no_keys")
            return []

        try:
            from alpaca.data.requests import NewsRequest

            client = self._get_client()
            request = NewsRequest(
                symbols=",".join(tickers[:30]),
                limit=min(limit, 50),
            )
            response = client.get_news(request)

            articles: list[NewsArticle] = []
            for item in response.data.get("news", []):
                published = None
                if hasattr(item, "created_at") and item.created_at:
                    published = (
                        item.created_at
                        if isinstance(item.created_at, datetime)
                        else datetime.fromisoformat(str(item.created_at))
                    )

                article_tickers = []
                if hasattr(item, "symbols") and item.symbols:
                    article_tickers = [s for s in item.symbols if s]

                articles.append(
                    NewsArticle(
                        headline=item.headline or "",
                        summary=getattr(item, "summary", "") or "",
                        source="alpaca",
                        publisher=getattr(item, "source", "Benzinga") or "Benzinga",
                        tickers=article_tickers,
                        published_at=published,
                        url=getattr(item, "url", "") or "",
                    )
                )

            log.info("alpaca_news_fetched", count=len(articles))
            return articles

        except Exception as e:
            log.warning("alpaca_news_fetch_failed", error=str(e))
            return []

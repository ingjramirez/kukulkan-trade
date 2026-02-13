"""Finnhub news fetcher for company and market-wide news.

Free tier: 60 calls/min. Returns headline, summary, source, url, datetime.
Requires FINNHUB_API_KEY environment variable.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta

import structlog

from config.settings import settings
from src.data.news_article import NewsArticle

log = structlog.get_logger()


class FinnhubNewsFetcher:
    """Fetches news from Finnhub's company-news and general-news endpoints."""

    def __init__(self, api_key: str | None = None) -> None:
        self._api_key = api_key or settings.finnhub.api_key
        self._client = None

    def _get_client(self):
        """Lazy-init Finnhub client."""
        if self._client is None:
            import finnhub

            self._client = finnhub.Client(api_key=self._api_key)
        return self._client

    def fetch(
        self,
        tickers: list[str],
        days_back: int = 2,
        max_per_ticker: int = 5,
    ) -> list[NewsArticle]:
        """Fetch company news for given tickers + general market news.

        Args:
            tickers: List of ticker symbols.
            days_back: How many days of history to fetch.
            max_per_ticker: Max articles per ticker query.

        Returns:
            List of NewsArticle objects.
        """
        if not self._api_key:
            log.debug("finnhub_news_skipped_no_key")
            return []

        articles: list[NewsArticle] = []
        today = date.today()
        from_date = (today - timedelta(days=days_back)).isoformat()
        to_date = today.isoformat()

        try:
            client = self._get_client()

            # Company news per ticker
            for ticker in tickers[:20]:  # Respect rate limits
                try:
                    raw = client.company_news(ticker, _from=from_date, to=to_date)
                    if not raw:
                        continue
                    for item in raw[:max_per_ticker]:
                        published = None
                        if item.get("datetime"):
                            try:
                                published = datetime.fromtimestamp(item["datetime"])
                            except (ValueError, OSError):
                                pass

                        articles.append(
                            NewsArticle(
                                headline=item.get("headline", ""),
                                summary=item.get("summary", ""),
                                source="finnhub",
                                publisher=item.get("source", ""),
                                tickers=[ticker] + item.get("related", "").split(","),
                                published_at=published,
                                url=item.get("url", ""),
                            )
                        )
                except Exception as e:
                    log.debug("finnhub_company_news_failed", ticker=ticker, error=str(e))

            # General market news
            try:
                general = client.general_news("general", min_id=0)
                for item in (general or [])[:10]:
                    published = None
                    if item.get("datetime"):
                        try:
                            published = datetime.fromtimestamp(item["datetime"])
                        except (ValueError, OSError):
                            pass

                    related = item.get("related", "")
                    article_tickers = [t.strip() for t in related.split(",") if t.strip()]

                    articles.append(
                        NewsArticle(
                            headline=item.get("headline", ""),
                            summary=item.get("summary", ""),
                            source="finnhub",
                            publisher=item.get("source", ""),
                            tickers=article_tickers,
                            published_at=published,
                            url=item.get("url", ""),
                        )
                    )
            except Exception as e:
                log.debug("finnhub_general_news_failed", error=str(e))

            log.info("finnhub_news_fetched", count=len(articles))

        except Exception as e:
            log.warning("finnhub_news_fetch_failed", error=str(e))

        return articles

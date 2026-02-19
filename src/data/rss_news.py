"""RSS news fetcher for global news sources (Reuters, Nikkei Asia, SCMP).

Uses feedparser (BSD license, sync). Configurable per source.
Graceful fallback on bad feeds — logs warning, returns empty list.
"""

from __future__ import annotations

from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

import feedparser
import structlog

from config.universe import FULL_UNIVERSE
from src.data.base_fetcher import BaseNewsFetcher
from src.data.news_article import NewsArticle

log = structlog.get_logger()

MAX_ENTRIES_PER_FEED = 15


class RSSNewsFetcher(BaseNewsFetcher):
    """Fetches news from one or more RSS feeds."""

    def __init__(
        self,
        source_name: str,
        feed_urls: list[str],
        region: str = "global",
        source_language: str = "en",
    ) -> None:
        self.source_name = source_name
        self.region = region
        self._feed_urls = feed_urls
        self._source_language = source_language
        self._universe = set(FULL_UNIVERSE)

    def fetch(self, tickers: list[str] | None = None) -> list[NewsArticle]:
        """Fetch articles from configured RSS feeds.

        Args:
            tickers: Optional ticker filter (unused — we extract from text).

        Returns:
            List of NewsArticle objects.
        """
        articles: list[NewsArticle] = []

        for url in self._feed_urls:
            try:
                feed = feedparser.parse(url)
                if feed.bozo and not feed.entries:
                    log.warning("rss_feed_parse_error", source=self.source_name, url=url)
                    continue

                for entry in feed.entries[:MAX_ENTRIES_PER_FEED]:
                    title = entry.get("title", "")
                    if not title:
                        continue

                    summary = entry.get("summary", "") or entry.get("description", "")
                    # Strip HTML tags from summary
                    summary = _strip_html(summary)[:300]

                    # Extract tickers from title + summary
                    found_tickers = self._extract_tickers(f"{title} {summary}", self._universe)

                    # Parse publication date
                    published = _parse_date(entry)

                    articles.append(
                        NewsArticle(
                            headline=title,
                            summary=summary,
                            source=self.source_name,
                            publisher=feed.feed.get("title", self.source_name),
                            tickers=found_tickers,
                            published_at=published,
                            url=entry.get("link", ""),
                            region=self.region,
                            source_language=self._source_language,
                            metadata={"feed_url": url},
                        )
                    )

            except Exception as e:
                log.warning("rss_fetch_failed", source=self.source_name, url=url, error=str(e))

        log.info("rss_news_fetched", source=self.source_name, count=len(articles))
        return articles


def _strip_html(text: str) -> str:
    """Remove HTML tags from text."""
    import re

    return re.sub(r"<[^>]+>", "", text).strip()


def _parse_date(entry: dict) -> datetime | None:
    """Parse publication date from RSS entry."""
    # Try string-based date first
    published_str = entry.get("published") or entry.get("updated")
    if published_str:
        try:
            return parsedate_to_datetime(published_str)
        except (ValueError, TypeError):
            pass

    # Fallback: try struct_time from feedparser
    struct_time = entry.get("published_parsed") or entry.get("updated_parsed")
    if struct_time:
        try:
            from calendar import timegm

            return datetime.fromtimestamp(timegm(struct_time), tz=timezone.utc)
        except (ValueError, OSError):
            pass

    return None


def create_default_rss_fetchers() -> list[RSSNewsFetcher]:
    """Create preconfigured RSS fetchers for global news sources.

    Returns:
        List of RSSNewsFetcher instances for Reuters, Nikkei Asia, SCMP.
    """
    return [
        RSSNewsFetcher(
            source_name="reuters",
            feed_urls=[
                "https://www.reutersagency.com/feed/?best-topics=business-finance&post_type=best",
            ],
            region="global",
            source_language="en",
        ),
        RSSNewsFetcher(
            source_name="nikkei_asia",
            feed_urls=[
                "https://asia.nikkei.com/rss",
            ],
            region="asia",
            source_language="en",
        ),
        RSSNewsFetcher(
            source_name="scmp",
            feed_urls=[
                "https://www.scmp.com/rss/91/feed",
            ],
            region="china",
            source_language="en",
        ),
    ]

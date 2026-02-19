"""Tests for RSS news fetcher."""

from unittest.mock import MagicMock, patch

from src.data.base_fetcher import BaseNewsFetcher
from src.data.rss_news import (
    RSSNewsFetcher,
    _parse_date,
    _strip_html,
    create_default_rss_fetchers,
)


def test_inherits_base_fetcher():
    fetcher = RSSNewsFetcher(source_name="test", feed_urls=["https://example.com/rss"])
    assert isinstance(fetcher, BaseNewsFetcher)
    assert fetcher.source_name == "test"
    assert fetcher.region == "global"


def test_strip_html():
    assert _strip_html("<p>Hello <b>world</b></p>") == "Hello world"
    assert _strip_html("No tags here") == "No tags here"
    assert _strip_html("") == ""


def test_parse_date_rfc2822():
    entry = {"published": "Mon, 19 Feb 2026 12:00:00 GMT"}
    result = _parse_date(entry)
    assert result is not None
    assert result.year == 2026
    assert result.month == 2


def test_parse_date_struct_time():
    import time

    entry = {"published_parsed": time.strptime("2026-02-19", "%Y-%m-%d")}
    result = _parse_date(entry)
    assert result is not None


def test_parse_date_returns_none():
    assert _parse_date({}) is None
    assert _parse_date({"published": "not-a-date"}) is None


def _make_feed(entries: list[dict], feed_title: str = "Test Feed") -> MagicMock:
    """Create a mock feedparser result."""
    feed = MagicMock()
    feed.bozo = False
    feed.entries = []
    for entry in entries:
        mock_entry = MagicMock()
        mock_entry.get = entry.get
        mock_entry.__getitem__ = entry.__getitem__
        feed.entries.append(entry)
    feed.feed = {"title": feed_title}
    return feed


@patch("src.data.rss_news.feedparser.parse")
def test_fetch_returns_articles(mock_parse):
    mock_parse.return_value = _make_feed(
        [
            {
                "title": "NVDA supply chain restart in Asia",
                "summary": "NVIDIA partners resume production at TSM facilities",
                "link": "https://example.com/article1",
                "published": "Mon, 19 Feb 2026 12:00:00 GMT",
            }
        ],
        feed_title="Reuters Business",
    )

    fetcher = RSSNewsFetcher(
        source_name="reuters",
        feed_urls=["https://example.com/rss"],
        region="global",
    )
    articles = fetcher.fetch()

    assert len(articles) == 1
    assert articles[0].source == "reuters"
    assert articles[0].region == "global"
    assert articles[0].headline == "NVDA supply chain restart in Asia"


@patch("src.data.rss_news.feedparser.parse")
def test_fetch_extracts_tickers(mock_parse):
    mock_parse.return_value = _make_feed(
        [
            {
                "title": "AAPL and MSFT lead tech rally",
                "summary": "",
                "link": "",
            }
        ]
    )

    fetcher = RSSNewsFetcher(source_name="test", feed_urls=["https://example.com/rss"])
    articles = fetcher.fetch()
    assert "AAPL" in articles[0].tickers
    assert "MSFT" in articles[0].tickers


@patch("src.data.rss_news.feedparser.parse")
def test_fetch_handles_bad_feed(mock_parse):
    mock_feed = MagicMock()
    mock_feed.bozo = True
    mock_feed.entries = []
    mock_parse.return_value = mock_feed

    fetcher = RSSNewsFetcher(source_name="bad", feed_urls=["https://example.com/bad"])
    articles = fetcher.fetch()
    assert articles == []


@patch("src.data.rss_news.feedparser.parse")
def test_fetch_strips_html_from_summary(mock_parse):
    mock_parse.return_value = _make_feed(
        [
            {
                "title": "Market update",
                "summary": "<p>Tech stocks <b>surge</b> on earnings</p>",
                "link": "",
            }
        ]
    )

    fetcher = RSSNewsFetcher(source_name="test", feed_urls=["https://example.com/rss"])
    articles = fetcher.fetch()
    assert "<" not in articles[0].summary


def test_create_default_rss_fetchers():
    fetchers = create_default_rss_fetchers()
    assert len(fetchers) == 3

    names = {f.source_name for f in fetchers}
    assert "reuters" in names
    assert "nikkei_asia" in names
    assert "scmp" in names

    regions = {f.region for f in fetchers}
    assert "global" in regions
    assert "asia" in regions
    assert "china" in regions


@patch("src.data.rss_news.feedparser.parse")
def test_metadata_includes_feed_url(mock_parse):
    mock_parse.return_value = _make_feed(
        [{"title": "Test article", "summary": "", "link": ""}]
    )

    fetcher = RSSNewsFetcher(
        source_name="test",
        feed_urls=["https://example.com/rss"],
        region="asia",
        source_language="ja",
    )
    articles = fetcher.fetch()
    assert articles[0].metadata["feed_url"] == "https://example.com/rss"
    assert articles[0].source_language == "ja"
    assert articles[0].region == "asia"

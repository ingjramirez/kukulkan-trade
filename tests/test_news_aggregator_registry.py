"""Tests for NewsAggregator registry pattern."""

from unittest.mock import MagicMock, patch

from src.data.base_fetcher import BaseNewsFetcher
from src.data.news_aggregator import NewsAggregator
from src.data.news_article import NewsArticle


class FakeNewsFetcher(BaseNewsFetcher):
    source_name = "fake_rss"
    region = "asia"

    def __init__(self, articles: list[NewsArticle] | None = None) -> None:
        self._articles = articles or []

    def fetch(self, tickers: list[str] | None = None) -> list[NewsArticle]:
        return self._articles


class FailingFetcher(BaseNewsFetcher):
    source_name = "failing"
    region = "us"

    def fetch(self, tickers: list[str] | None = None) -> list[NewsArticle]:
        raise RuntimeError("API down")


def _make_article(headline: str, source: str = "test", tickers: list[str] | None = None) -> NewsArticle:
    return NewsArticle(
        headline=headline,
        summary="",
        source=source,
        publisher="TestPub",
        tickers=tickers or [],
    )


def test_register_fetcher():
    agg = NewsAggregator(
        alpaca_fetcher=MagicMock(fetch=MagicMock(return_value=[])),
        finnhub_fetcher=MagicMock(fetch=MagicMock(return_value=[])),
    )
    fetcher = FakeNewsFetcher()
    agg.register(fetcher)
    assert "fake_rss" in agg.registered_sources


@patch("src.data.news_aggregator.NewsAggregator._fetch_yfinance", return_value=[])
def test_fetch_all_includes_registered_sources(mock_yf):
    alpaca = MagicMock(
        fetch=MagicMock(return_value=[_make_article("AAPL hits record high on strong earnings", "alpaca")])
    )
    finnhub = MagicMock(
        fetch=MagicMock(return_value=[_make_article("Federal Reserve holds interest rates steady", "finnhub")])
    )
    agg = NewsAggregator(alpaca_fetcher=alpaca, finnhub_fetcher=finnhub)

    rss_article = _make_article("TSM supply chain restart boosts semiconductor outlook", "rss", tickers=["TSM"])
    agg.register(FakeNewsFetcher([rss_article]))

    result = agg.fetch_all(["AAPL"])
    headlines = [a.headline for a in result]
    assert "TSM supply chain restart boosts semiconductor outlook" in headlines
    assert len(result) == 3


@patch("src.data.news_aggregator.NewsAggregator._fetch_yfinance", return_value=[])
def test_failing_fetcher_doesnt_crash_aggregator(mock_yf):
    alpaca = MagicMock(
        fetch=MagicMock(return_value=[_make_article("AAPL hits record high on strong earnings", "alpaca")])
    )
    finnhub = MagicMock(fetch=MagicMock(return_value=[]))
    agg = NewsAggregator(alpaca_fetcher=alpaca, finnhub_fetcher=finnhub)
    agg.register(FailingFetcher())

    # Should not raise — failing fetcher is caught gracefully
    result = agg.fetch_all(["AAPL"])
    assert len(result) >= 1


@patch("src.data.news_aggregator.NewsAggregator._fetch_yfinance", return_value=[])
def test_dedup_across_sources(mock_yf):
    alpaca = MagicMock(
        fetch=MagicMock(return_value=[_make_article("NVDA beats earnings expectations", "alpaca", ["NVDA"])])
    )
    finnhub = MagicMock(
        fetch=MagicMock(return_value=[_make_article("NVDA beats earnings expectations strongly", "finnhub", ["NVDA"])])
    )
    agg = NewsAggregator(alpaca_fetcher=alpaca, finnhub_fetcher=finnhub)

    result = agg.fetch_all(["NVDA"])
    # Should dedup — nearly identical headlines
    assert len(result) == 1

"""Tests for BaseNewsFetcher ABC and NewsArticle extensions."""

from src.data.base_fetcher import BaseNewsFetcher
from src.data.news_article import NewsArticle


class ConcreteFetcher(BaseNewsFetcher):
    """Concrete implementation for testing."""

    source_name = "test_source"
    region = "global"

    def __init__(self, articles: list[NewsArticle] | None = None) -> None:
        self._articles = articles or []

    def fetch(self, tickers: list[str] | None = None) -> list[NewsArticle]:
        return self._articles


def test_concrete_fetcher_returns_articles():
    articles = [
        NewsArticle(headline="Test", summary="", source="test", publisher="Test", tickers=["AAPL"]),
    ]
    fetcher = ConcreteFetcher(articles)
    assert fetcher.fetch() == articles
    assert fetcher.source_name == "test_source"
    assert fetcher.region == "global"


def test_extract_tickers_cashtag():
    fetcher = ConcreteFetcher()
    universe = {"AAPL", "TSLA", "MSFT", "NVDA"}
    text = "Big day for $AAPL and $TSLA — both rallying hard"
    result = fetcher._extract_tickers(text, universe)
    assert "AAPL" in result
    assert "TSLA" in result


def test_extract_tickers_uppercase_words():
    fetcher = ConcreteFetcher()
    universe = {"NVDA", "AMD", "INTC"}
    text = "NVDA and AMD are leading the semiconductor rally today"
    result = fetcher._extract_tickers(text, universe)
    assert "NVDA" in result
    assert "AMD" in result


def test_extract_tickers_filters_to_universe():
    fetcher = ConcreteFetcher()
    universe = {"AAPL"}
    text = "$AAPL $FAKE and XYZ are mentioned"
    result = fetcher._extract_tickers(text, universe)
    assert result == ["AAPL"]


def test_extract_tickers_empty_text():
    fetcher = ConcreteFetcher()
    assert fetcher._extract_tickers("", {"AAPL"}) == []


def test_news_article_new_fields_defaults():
    article = NewsArticle(headline="Test", summary="", source="test", publisher="P", tickers=[])
    assert article.region == "us"
    assert article.source_language == "en"
    assert article.metadata == {}


def test_news_article_new_fields_custom():
    article = NewsArticle(
        headline="Asia news",
        summary="",
        source="nikkei",
        publisher="Nikkei",
        tickers=[],
        region="asia",
        source_language="ja",
        metadata={"feed": "nikkei_asia"},
    )
    assert article.region == "asia"
    assert article.source_language == "ja"
    assert article.metadata == {"feed": "nikkei_asia"}


def test_alpaca_fetcher_inherits_base():
    from src.data.alpaca_news import AlpacaNewsFetcher

    fetcher = AlpacaNewsFetcher(api_key="test", secret_key="test")
    assert isinstance(fetcher, BaseNewsFetcher)
    assert fetcher.source_name == "alpaca"
    assert fetcher.region == "us"


def test_finnhub_fetcher_inherits_base():
    from src.data.finnhub_news import FinnhubNewsFetcher

    fetcher = FinnhubNewsFetcher(api_key="test")
    assert isinstance(fetcher, BaseNewsFetcher)
    assert fetcher.source_name == "finnhub"
    assert fetcher.region == "us"

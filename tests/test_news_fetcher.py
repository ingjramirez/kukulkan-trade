"""Tests for the news fetcher and ChromaDB integration.

Uses mocked yfinance and mocked ChromaDB — no external API calls.
"""

from datetime import datetime
from unittest.mock import MagicMock, patch

from src.data.news_fetcher import NewsFetcher, _article_id

# ── Fixtures ─────────────────────────────────────────────────────────────────


def _mock_vector_store():
    """Create a mock VectorStore."""
    vs = MagicMock()
    vs.add_news = MagicMock()
    vs.search_similar = MagicMock(return_value={
        "documents": [["NVDA beats earnings expectations", "Fed holds rates steady"]],
        "metadatas": [[
            {"ticker": "NVDA", "publisher": "Reuters"},
            {"ticker": "SPY", "publisher": "Bloomberg"},
        ]],
        "distances": [[0.15, 0.32]],
    })
    vs.count = MagicMock(return_value=10)
    return vs


def _mock_yfinance_news():
    """Create mock yfinance news data."""
    return [
        {
            "title": "AAPL hits all-time high on strong iPhone sales",
            "link": "https://example.com/aapl",
            "publisher": "Reuters",
            "providerPublishTime": 1738700000,
        },
        {
            "title": "Tech sector rally continues amid AI optimism",
            "link": "https://example.com/tech",
            "publisher": "Bloomberg",
            "providerPublishTime": 1738690000,
        },
        {
            "title": "Market volatility spikes on trade war fears",
            "link": "https://example.com/vol",
            "publisher": "CNBC",
            "providerPublishTime": 1738680000,
        },
    ]


# ── Article ID ───────────────────────────────────────────────────────────────


class TestArticleId:
    def test_deterministic(self) -> None:
        article = {"ticker": "AAPL", "title": "Test headline"}
        id1 = _article_id(article)
        id2 = _article_id(article)
        assert id1 == id2

    def test_different_for_different_articles(self) -> None:
        a1 = {"ticker": "AAPL", "title": "Headline A"}
        a2 = {"ticker": "AAPL", "title": "Headline B"}
        assert _article_id(a1) != _article_id(a2)

    def test_different_tickers_same_title(self) -> None:
        a1 = {"ticker": "AAPL", "title": "Same headline"}
        a2 = {"ticker": "MSFT", "title": "Same headline"}
        assert _article_id(a1) != _article_id(a2)

    def test_returns_hex_string(self) -> None:
        article = {"ticker": "XLK", "title": "Test"}
        result = _article_id(article)
        assert len(result) == 32  # MD5 hex digest
        assert all(c in "0123456789abcdef" for c in result)


# ── Fetch News ───────────────────────────────────────────────────────────────


class TestFetchNews:
    @patch("src.data.news_fetcher.yf")
    def test_fetches_articles(self, mock_yf) -> None:
        mock_ticker = MagicMock()
        mock_ticker.news = _mock_yfinance_news()
        mock_yf.Ticker.return_value = mock_ticker

        fetcher = NewsFetcher(vector_store=_mock_vector_store())
        articles = fetcher.fetch_news(["AAPL"], max_per_ticker=5)

        assert len(articles) == 3
        assert articles[0]["ticker"] == "AAPL"
        assert "all-time high" in articles[0]["title"]

    @patch("src.data.news_fetcher.yf")
    def test_deduplicates_by_title(self, mock_yf) -> None:
        dup_news = [
            {
                "title": "Same headline",
                "link": "https://a.com",
                "publisher": "A",
                "providerPublishTime": 1738700000,
            },
            {
                "title": "Same headline",
                "link": "https://b.com",
                "publisher": "B",
                "providerPublishTime": 1738700000,
            },
        ]
        mock_ticker = MagicMock()
        mock_ticker.news = dup_news
        mock_yf.Ticker.return_value = mock_ticker

        fetcher = NewsFetcher(vector_store=_mock_vector_store())
        articles = fetcher.fetch_news(["AAPL"])

        assert len(articles) == 1

    @patch("src.data.news_fetcher.yf")
    def test_handles_empty_news(self, mock_yf) -> None:
        mock_ticker = MagicMock()
        mock_ticker.news = []
        mock_yf.Ticker.return_value = mock_ticker

        fetcher = NewsFetcher(vector_store=_mock_vector_store())
        articles = fetcher.fetch_news(["AAPL"])

        assert articles == []

    @patch("src.data.news_fetcher.yf")
    def test_handles_none_news(self, mock_yf) -> None:
        mock_ticker = MagicMock()
        mock_ticker.news = None
        mock_yf.Ticker.return_value = mock_ticker

        fetcher = NewsFetcher(vector_store=_mock_vector_store())
        articles = fetcher.fetch_news(["AAPL"])

        assert articles == []

    @patch("src.data.news_fetcher.yf")
    def test_respects_max_per_ticker(self, mock_yf) -> None:
        mock_ticker = MagicMock()
        mock_ticker.news = _mock_yfinance_news()
        mock_yf.Ticker.return_value = mock_ticker

        fetcher = NewsFetcher(vector_store=_mock_vector_store())
        articles = fetcher.fetch_news(["AAPL"], max_per_ticker=2)

        assert len(articles) == 2

    @patch("src.data.news_fetcher.yf")
    def test_multiple_tickers(self, mock_yf) -> None:
        mock_ticker = MagicMock()
        mock_ticker.news = [
            {
                "title": "Unique headline for ticker",
                "link": "",
                "publisher": "Test",
                "providerPublishTime": None,
            },
        ]
        mock_yf.Ticker.return_value = mock_ticker

        fetcher = NewsFetcher(vector_store=_mock_vector_store())
        # Same headline from different tickers — only first kept (dedup by title)
        articles = fetcher.fetch_news(["AAPL", "MSFT"])

        assert len(articles) == 1  # deduped across tickers

    @patch("src.data.news_fetcher.yf")
    def test_handles_yfinance_error(self, mock_yf) -> None:
        mock_yf.Ticker.side_effect = Exception("API error")

        fetcher = NewsFetcher(vector_store=_mock_vector_store())
        articles = fetcher.fetch_news(["AAPL"])

        assert articles == []

    @patch("src.data.news_fetcher.yf")
    def test_published_timestamp_converted(self, mock_yf) -> None:
        mock_ticker = MagicMock()
        mock_ticker.news = [
            {"title": "Test", "link": "", "publisher": "", "providerPublishTime": 1738700000},
        ]
        mock_yf.Ticker.return_value = mock_ticker

        fetcher = NewsFetcher(vector_store=_mock_vector_store())
        articles = fetcher.fetch_news(["AAPL"])

        assert isinstance(articles[0]["published"], datetime)


# ── Store Articles ───────────────────────────────────────────────────────────


class TestStoreArticles:
    def test_stores_in_chromadb(self) -> None:
        vs = _mock_vector_store()
        fetcher = NewsFetcher(vector_store=vs)

        articles = [
            {
                "ticker": "AAPL",
                "title": "Test headline",
                "link": "https://a.com",
                "publisher": "Reuters",
            },
        ]
        rows = fetcher.store_articles(articles)

        vs.add_news.assert_called_once()
        assert len(rows) == 1
        assert rows[0].ticker == "AAPL"
        assert rows[0].headline == "Test headline"
        assert rows[0].source == "Reuters"

    def test_returns_news_log_rows(self) -> None:
        vs = _mock_vector_store()
        fetcher = NewsFetcher(vector_store=vs)

        articles = [
            {
                "ticker": "MSFT",
                "title": "MSFT earnings",
                "link": "https://b.com",
                "publisher": "Bloomberg",
                "published": datetime(2026, 2, 5),
            },
            {
                "ticker": "NVDA",
                "title": "NVDA GPU launch",
                "link": "https://c.com",
                "publisher": "CNBC",
            },
        ]
        rows = fetcher.store_articles(articles)

        assert len(rows) == 2
        assert rows[0].published_at == datetime(2026, 2, 5)
        assert rows[1].published_at is None  # no timestamp

    def test_handles_chromadb_error(self) -> None:
        vs = _mock_vector_store()
        vs.add_news.side_effect = Exception("ChromaDB unavailable")
        fetcher = NewsFetcher(vector_store=vs)

        articles = [{"ticker": "XLK", "title": "Test", "link": "", "publisher": ""}]
        rows = fetcher.store_articles(articles)

        # Still returns the row even if ChromaDB fails
        assert len(rows) == 1


# ── Search Relevant ──────────────────────────────────────────────────────────


class TestSearchRelevant:
    def test_returns_articles(self) -> None:
        vs = _mock_vector_store()
        fetcher = NewsFetcher(vector_store=vs)

        results = fetcher.search_relevant("tech earnings")

        assert len(results) == 2
        assert results[0]["title"] == "NVDA beats earnings expectations"
        assert results[0]["ticker"] == "NVDA"
        assert results[0]["distance"] == 0.15

    def test_empty_results(self) -> None:
        vs = _mock_vector_store()
        vs.search_similar.return_value = {"documents": [[]], "metadatas": [[]], "distances": [[]]}
        fetcher = NewsFetcher(vector_store=vs)

        results = fetcher.search_relevant("obscure query")
        assert results == []

    def test_handles_none_results(self) -> None:
        vs = _mock_vector_store()
        vs.search_similar.return_value = {}
        fetcher = NewsFetcher(vector_store=vs)

        results = fetcher.search_relevant("query")
        assert results == []


# ── News Context ─────────────────────────────────────────────────────────────


class TestGetTargetedContext:
    def test_returns_per_ticker_results(self) -> None:
        vs = _mock_vector_store()
        fetcher = NewsFetcher(vector_store=vs)

        context = fetcher.get_targeted_context(["NVDA", "SPY"])

        assert "NVDA" in context
        assert "beats earnings" in context

    def test_deduplicates_by_title(self) -> None:
        """Same title from different ticker searches should be deduped."""
        vs = _mock_vector_store()
        fetcher = NewsFetcher(vector_store=vs)

        context = fetcher.get_targeted_context(["NVDA", "SPY"])
        lines = [ln for ln in context.strip().split("\n") if ln.strip()]
        titles = [ln.split("]")[1].strip() if "]" in ln else ln for ln in lines]
        assert len(titles) == len(set(titles))

    def test_empty_tickers_returns_empty(self) -> None:
        vs = _mock_vector_store()
        fetcher = NewsFetcher(vector_store=vs)

        context = fetcher.get_targeted_context([])
        assert context == ""

    def test_handles_search_error(self) -> None:
        vs = _mock_vector_store()
        vs.search_similar.side_effect = Exception("ChromaDB down")
        fetcher = NewsFetcher(vector_store=vs)

        context = fetcher.get_targeted_context(["NVDA"])
        assert context == ""


class TestGetNewsContext:
    def test_formats_context_string(self) -> None:
        vs = _mock_vector_store()
        fetcher = NewsFetcher(vector_store=vs)

        context = fetcher.get_news_context(["NVDA", "SPY"])

        assert "NVDA" in context
        assert "beats earnings expectations" in context
        assert "Fed holds rates steady" in context
        assert "Reuters" in context

    def test_no_news_returns_placeholder(self) -> None:
        vs = _mock_vector_store()
        vs.search_similar.return_value = {}
        fetcher = NewsFetcher(vector_store=vs)

        context = fetcher.get_news_context(["AAPL"])

        assert "no recent news" in context

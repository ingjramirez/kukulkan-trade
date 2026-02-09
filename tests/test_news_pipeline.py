"""Tests for the multi-source news pipeline: aggregator, compactor, source fetchers.

All external APIs are mocked — no network calls.
"""

from datetime import datetime
from unittest.mock import MagicMock, patch

from src.data.alpaca_news import AlpacaNewsFetcher
from src.data.finnhub_news import FinnhubNewsFetcher
from src.data.news_aggregator import (
    NewsAggregator,
    _headline_words,
    _headlines_overlap,
)
from src.data.news_article import NewsArticle, NewsCluster
from src.data.news_compactor import (
    NewsCompactor,
    classify_signal,
    compress_headline,
)

# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_article(
    headline: str = "Test headline",
    tickers: list[str] | None = None,
    source: str = "alpaca",
    publisher: str = "Benzinga",
    sentiment: float | None = None,
) -> NewsArticle:
    return NewsArticle(
        headline=headline,
        summary="Summary of " + headline,
        source=source,
        publisher=publisher,
        tickers=tickers or ["AAPL"],
        published_at=datetime(2026, 2, 6, 12, 0),
        url="https://example.com/article",
        sentiment=sentiment,
    )


# ── AlpacaNewsFetcher Tests ─────────────────────────────────────────────────


class TestAlpacaNewsFetcher:
    def test_returns_articles_from_api(self) -> None:
        fetcher = AlpacaNewsFetcher(api_key="test-key", secret_key="test-secret")

        mock_news_item = MagicMock()
        mock_news_item.headline = "NVDA beats Q4 earnings expectations"
        mock_news_item.summary = "Record data center revenue"
        mock_news_item.source = "Benzinga"
        mock_news_item.symbols = ["NVDA"]
        mock_news_item.created_at = datetime(2026, 2, 6, 10, 0)
        mock_news_item.url = "https://example.com/nvda"

        mock_response = MagicMock()
        mock_response.data = [mock_news_item]

        mock_client = MagicMock()
        mock_client.get_news.return_value = mock_response
        fetcher._client = mock_client

        articles = fetcher.fetch(["NVDA"])

        assert len(articles) == 1
        assert articles[0].headline == "NVDA beats Q4 earnings expectations"
        assert articles[0].source == "alpaca"
        assert "NVDA" in articles[0].tickers

    def test_returns_empty_without_keys(self) -> None:
        fetcher = AlpacaNewsFetcher(api_key="", secret_key="")
        articles = fetcher.fetch(["AAPL"])
        assert articles == []

    def test_handles_api_error(self) -> None:
        fetcher = AlpacaNewsFetcher(api_key="key", secret_key="secret")
        mock_client = MagicMock()
        mock_client.get_news.side_effect = Exception("API error")
        fetcher._client = mock_client

        articles = fetcher.fetch(["AAPL"])
        assert articles == []


# ── FinnhubNewsFetcher Tests ────────────────────────────────────────────────


class TestFinnhubNewsFetcher:
    def test_returns_articles_from_api(self) -> None:
        fetcher = FinnhubNewsFetcher(api_key="test-key")

        mock_client = MagicMock()
        mock_client.company_news.return_value = [
            {
                "headline": "AAPL iPhone revenue surges 12%",
                "summary": "Strong demand in services segment",
                "source": "Reuters",
                "related": "AAPL",
                "datetime": 1738800000,
                "url": "https://example.com/aapl",
            },
        ]
        mock_client.general_news.return_value = [
            {
                "headline": "Fed holds rates steady signals no June cut",
                "summary": "Federal Reserve holds rates",
                "source": "Bloomberg",
                "related": "",
                "datetime": 1738790000,
                "url": "https://example.com/fed",
            },
        ]
        fetcher._client = mock_client

        articles = fetcher.fetch(["AAPL"])

        assert len(articles) == 2
        assert articles[0].source == "finnhub"
        assert articles[0].publisher == "Reuters"
        assert articles[1].headline == "Fed holds rates steady signals no June cut"

    def test_returns_empty_without_key(self) -> None:
        fetcher = FinnhubNewsFetcher(api_key="")
        articles = fetcher.fetch(["AAPL"])
        assert articles == []

    def test_handles_api_error(self) -> None:
        fetcher = FinnhubNewsFetcher(api_key="key")
        mock_client = MagicMock()
        mock_client.company_news.side_effect = Exception("API error")
        mock_client.general_news.side_effect = Exception("API error")
        fetcher._client = mock_client

        articles = fetcher.fetch(["AAPL"])
        assert articles == []


# ── NewsAggregator Tests ────────────────────────────────────────────────────


class TestNewsAggregatorDedup:
    def test_headline_words_removes_stop_words(self) -> None:
        words = _headline_words("The quick fox is jumping over the lazy dog")
        assert "the" not in words
        assert "is" not in words
        assert "quick" in words
        assert "fox" in words

    def test_headlines_overlap_detects_same_story(self) -> None:
        h1 = "NVDA beats Q4 earnings expectations revenue record"
        h2 = "NVDA Q4 earnings beat expectations strong revenue"
        assert _headlines_overlap(h1, h2) is True

    def test_headlines_overlap_different_stories(self) -> None:
        h1 = "NVDA beats Q4 earnings expectations"
        h2 = "Fed holds interest rates steady in March"
        assert _headlines_overlap(h1, h2) is False

    def test_deduplication_removes_same_event(self) -> None:
        alpaca = MagicMock()
        alpaca.fetch.return_value = [
            _make_article("NVDA beats Q4 earnings strong revenue record", ["NVDA"]),
        ]
        finnhub = MagicMock()
        finnhub.fetch.return_value = [
            _make_article(
                "NVDA Q4 earnings beat expectations revenue record high",
                ["NVDA"],
                source="finnhub",
            ),
        ]

        agg = NewsAggregator(alpaca_fetcher=alpaca, finnhub_fetcher=finnhub)
        articles = agg.fetch_all(["NVDA"])

        # Should be deduped to 1
        assert len(articles) == 1
        assert articles[0].source == "alpaca"  # higher priority source kept

    def test_merges_tickers_on_dedup(self) -> None:
        alpaca = MagicMock()
        alpaca.fetch.return_value = [
            _make_article("Tech sector rally continues strong momentum gains", ["XLK"]),
        ]
        finnhub = MagicMock()
        finnhub.fetch.return_value = [
            _make_article(
                "Tech sector rally continues strong momentum gains higher",
                ["QQQ"],
                source="finnhub",
            ),
        ]

        agg = NewsAggregator(alpaca_fetcher=alpaca, finnhub_fetcher=finnhub)
        articles = agg.fetch_all(["XLK"])

        assert len(articles) == 1
        assert "XLK" in articles[0].tickers
        assert "QQQ" in articles[0].tickers

    @patch("src.data.news_aggregator.yf")
    def test_fallback_to_yfinance(self, mock_yf) -> None:
        alpaca = MagicMock()
        alpaca.fetch.return_value = []
        finnhub = MagicMock()
        finnhub.fetch.return_value = []

        mock_ticker = MagicMock()
        mock_ticker.news = [
            {
                "title": "AAPL hits new high",
                "link": "https://a.com",
                "publisher": "Reuters",
                "providerPublishTime": 1738700000,
            },
        ]
        mock_yf.Ticker.return_value = mock_ticker

        agg = NewsAggregator(alpaca_fetcher=alpaca, finnhub_fetcher=finnhub)
        articles = agg.fetch_all(["AAPL"])

        assert len(articles) == 1
        assert articles[0].source == "yfinance"

    def test_no_fallback_when_enough_articles(self) -> None:
        alpaca = MagicMock()
        alpaca.fetch.return_value = [
            _make_article(f"Article {i}", ["AAPL"]) for i in range(15)
        ]
        finnhub = MagicMock()
        finnhub.fetch.return_value = []

        agg = NewsAggregator(alpaca_fetcher=alpaca, finnhub_fetcher=finnhub)
        # Patch yfinance to track if it was called
        with patch("src.data.news_aggregator.yf") as mock_yf:
            agg.fetch_all(["AAPL"])
            mock_yf.Ticker.assert_not_called()


# ── NewsCompactor Tests ─────────────────────────────────────────────────────


class TestSignalClassification:
    def test_positive_signal(self) -> None:
        assert classify_signal("NVDA beats earnings expectations") == "POS"

    def test_negative_signal(self) -> None:
        assert classify_signal("AAPL misses revenue target warns on guidance") == "NEG"

    def test_macro_signal(self) -> None:
        assert classify_signal("Fed holds rates steady no June cut expected") == "MACRO"

    def test_event_signal(self) -> None:
        assert classify_signal("Company X announces acquisition of startup") == "EVENT"

    def test_info_signal_default(self) -> None:
        assert classify_signal("Company releases new product lineup") == "INFO"

    def test_sentiment_overrides(self) -> None:
        assert classify_signal("Neutral headline about company", sentiment=0.5) == "POS"
        assert classify_signal("Neutral headline about company", sentiment=-0.5) == "NEG"

    def test_tariff_macro(self) -> None:
        assert classify_signal("New tariff on China imports sparks concern") == "MACRO"

    def test_inflation_macro(self) -> None:
        assert classify_signal("Inflation data comes in hot above expectations") == "MACRO"


class TestHeadlineCompression:
    def test_removes_noise_words(self) -> None:
        result = compress_headline("The quick fox is jumping over the lazy dog")
        assert "The" not in result
        assert "is" not in result
        assert "quick" in result

    def test_caps_at_max_words(self) -> None:
        long = " ".join(f"word{i}" for i in range(30))
        result = compress_headline(long, max_words=10)
        assert len(result.split()) == 10

    def test_preserves_content_words(self) -> None:
        result = compress_headline("NVDA Q4 earnings beat expectations revenue record")
        assert "NVDA" in result
        assert "earnings" in result
        assert "record" in result

    def test_empty_after_filter_uses_original(self) -> None:
        result = compress_headline("the is a")
        # Should fallback to original words
        assert len(result.split()) > 0


class TestNewsCompactorPipeline:
    def test_full_pipeline_produces_output(self) -> None:
        articles = [
            _make_article("NVDA beats Q4 earnings expectations record revenue", ["NVDA"]),
            _make_article("Fed holds rates steady no June cut", ["SPY"]),
            _make_article("AAPL iPhone revenue surges 12 percent", ["AAPL"]),
        ]
        compactor = NewsCompactor()
        result = compactor.compact(
            articles, held_tickers=["NVDA", "AAPL"], top_movers=["TSLA"],
        )

        assert "TICKER|SIGNAL|EVENT|#SRC" in result
        assert "NVDA" in result
        assert "|POS|" in result or "|MACRO|" in result

    def test_filter_keeps_held_tickers(self) -> None:
        articles = [
            _make_article("NVDA earnings beat", ["NVDA"]),
            _make_article("Some random company news", ["RANDOM"]),
        ]
        compactor = NewsCompactor()
        result = compactor.compact(articles, held_tickers=["NVDA"])

        assert "NVDA" in result
        assert "RANDOM" not in result

    def test_filter_keeps_macro_events(self) -> None:
        articles = [
            _make_article("Fed raises interest rates unexpectedly", []),
            _make_article("Random startup raises funds", ["STARTUP"]),
        ]
        compactor = NewsCompactor()
        result = compactor.compact(articles, held_tickers=[], top_movers=[])

        assert "MACRO" in result or "Fed" in result

    def test_clustering_groups_same_event(self) -> None:
        articles = [
            _make_article("NVDA beats Q4 earnings strong revenue record", ["NVDA"]),
            _make_article(
                "NVDA Q4 earnings beat expectations record revenue",
                ["NVDA"],
                source="finnhub",
            ),
            _make_article(
                "NVDA quarterly earnings beat record revenue",
                ["NVDA"],
                source="yfinance",
            ),
        ]
        compactor = NewsCompactor()
        result = compactor.compact(articles, held_tickers=["NVDA"])

        # Should be clustered into 1 event with 3 sources
        lines = [ln for ln in result.split("\n") if ln and not ln.startswith("TICKER|")]
        assert len(lines) == 1
        assert "|3" in lines[0]  # 3 sources

    def test_ranking_prioritizes_held_tickers(self) -> None:
        articles = [
            _make_article("Random market news update today", ["SPY"]),
            _make_article("NVDA earnings beat record revenue strong", ["NVDA"]),
        ]
        compactor = NewsCompactor()
        result = compactor.compact(
            articles, held_tickers=["NVDA"], top_movers=["SPY"],
        )

        lines = [ln for ln in result.split("\n") if ln and not ln.startswith("TICKER|")]
        # NVDA should come first (held ticker + earnings keyword)
        assert lines[0].startswith("NVDA")

    def test_compaction_under_600_tokens(self) -> None:
        """Compact output should be much shorter than raw headlines."""
        articles = [
            _make_article(f"Article {i} about market news today", [f"TICK{i}"])
            for i in range(20)
        ]
        compactor = NewsCompactor(max_clusters=8)
        # All articles are about unknown tickers and no macro keywords,
        # so the filter should remove them
        result = compactor.compact(articles, held_tickers=[], top_movers=[])

        # With no relevant articles, output should be empty or very short
        # Let's test with relevant ones
        relevant_articles = [
            _make_article("NVDA beats expectations revenue strong", ["NVDA"]),
            _make_article("AAPL iPhone sales surge higher record", ["AAPL"]),
            _make_article("Fed holds rates steady", ["SPY"]),
            _make_article("XLK tech sector drops fear tariff", ["XLK"]),
            _make_article("TSLA deliveries beat estimates", ["TSLA"]),
            _make_article("GLD gold rallies on uncertainty", ["GLD"]),
            _make_article("MSFT cloud revenue growth strong", ["MSFT"]),
            _make_article("IBIT bitcoin drops below 60K outflows", ["IBIT"]),
        ]
        result = compactor.compact(
            relevant_articles,
            held_tickers=["NVDA", "AAPL", "XLK"],
            top_movers=["TSLA", "GLD", "MSFT", "IBIT"],
        )

        # Rough token estimate: ~4 tokens per word, result should be under 600 tokens
        word_count = len(result.split())
        assert word_count < 150  # ~600 tokens / 4 words per token

    def test_empty_articles_returns_empty(self) -> None:
        compactor = NewsCompactor()
        result = compactor.compact([], held_tickers=["NVDA"])
        assert result == ""

    def test_max_clusters_cap(self) -> None:
        articles = [
            _make_article(f"Unique headline number {i} about earnings", [f"T{i}"])
            for i in range(20)
        ]
        compactor = NewsCompactor(max_clusters=5)
        result = compactor.compact(
            articles,
            held_tickers=[f"T{i}" for i in range(20)],
        )

        lines = [ln for ln in result.split("\n") if ln and not ln.startswith("TICKER|")]
        assert len(lines) <= 5


class TestPickDisplayTicker:
    def test_prefers_individual_stocks(self) -> None:
        compactor = NewsCompactor()
        cluster = NewsCluster(
            representative=_make_article("Test", ["XLK", "NVDA"]),
            all_tickers=["XLK", "NVDA"],
        )
        ticker = compactor._pick_display_ticker(cluster)
        assert ticker == "NVDA"  # Individual stock over ETF

    def test_falls_back_to_etf(self) -> None:
        compactor = NewsCompactor()
        cluster = NewsCluster(
            representative=_make_article("Test", ["XLK"]),
            all_tickers=["XLK"],
        )
        ticker = compactor._pick_display_ticker(cluster)
        assert ticker == "XLK"

    def test_macro_signal_returns_macro(self) -> None:
        compactor = NewsCompactor()
        cluster = NewsCluster(
            representative=_make_article("Fed holds rates", []),
            all_tickers=[],
            signal="MACRO",
        )
        ticker = compactor._pick_display_ticker(cluster)
        assert ticker == "MACRO"

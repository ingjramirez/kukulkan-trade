"""Tests for the NewsCompactor discovery feature and existing pipeline."""

from src.data.news_article import NewsArticle
from src.data.news_compactor import NewsCompactor


def _make_article(
    headline: str,
    tickers: list[str],
    summary: str = "",
    sentiment: float | None = None,
) -> NewsArticle:
    """Helper to create a NewsArticle for testing."""
    return NewsArticle(
        headline=headline,
        summary=summary or headline,
        source="test",
        publisher="TestPub",
        tickers=tickers,
        sentiment=sentiment,
    )


# ── Universe / discovery sets used across tests ──────────────────────────────

UNIVERSE = {"AAPL", "MSFT", "GOOG", "AMZN", "NVDA", "SPY", "QQQ", "GDX"}


class TestDiscoveryFiltering:
    """Articles with non-universe tickers appear in the discovery section."""

    def test_non_universe_articles_in_discovery(self) -> None:
        """Articles mentioning only non-universe tickers surface in discovery."""
        articles = [
            _make_article("AAPL beats earnings estimates", ["AAPL"]),
            _make_article("Palantir Q4 earnings beat estimates", ["PLTR"]),
            _make_article("ARM announces AI chip partnership", ["ARM"]),
        ]
        compactor = NewsCompactor()
        result = compactor.compact(
            articles=articles,
            held_tickers=["AAPL"],
            universe_tickers=UNIVERSE,
        )

        assert "== DISCOVERY (not in universe) ==" in result
        assert "PLTR" in result
        assert "ARM" in result

    def test_universe_articles_not_in_discovery(self) -> None:
        """Articles about universe tickers should NOT leak into discovery."""
        articles = [
            _make_article("AAPL surges on strong iPhone sales", ["AAPL"]),
            _make_article("MSFT beats cloud revenue expectations", ["MSFT"]),
        ]
        compactor = NewsCompactor()
        result = compactor.compact(
            articles=articles,
            held_tickers=["AAPL"],
            top_movers=["MSFT"],
            universe_tickers=UNIVERSE,
        )

        assert "== DISCOVERY (not in universe) ==" not in result
        assert "AAPL" in result
        assert "MSFT" in result

    def test_empty_discovery_when_no_non_universe(self) -> None:
        """No discovery section when all articles are universe/macro."""
        articles = [
            _make_article("NVDA rally on AI demand", ["NVDA"]),
            _make_article("Fed holds rates steady", []),
        ]
        compactor = NewsCompactor()
        result = compactor.compact(
            articles=articles,
            held_tickers=["NVDA"],
            universe_tickers=UNIVERSE,
        )

        assert "== DISCOVERY (not in universe) ==" not in result

    def test_max_3_discovery_items(self) -> None:
        """Discovery section capped at 3 items."""
        articles = [
            _make_article("PLTR earnings beat", ["PLTR"]),
            _make_article("UBER revenue surge", ["UBER"]),
            _make_article("ARM chip deal announced", ["ARM"]),
            _make_article("SNOW cloud growth strong", ["SNOW"]),
            _make_article("COIN crypto rally gains", ["COIN"]),
        ]
        compactor = NewsCompactor()
        result = compactor.compact(
            articles=articles,
            held_tickers=[],
            universe_tickers=UNIVERSE,
        )

        discovery_section = result.split("== DISCOVERY (not in universe) ==")[1]
        discovery_lines = [
            line for line in discovery_section.strip().split("\n") if line.strip()
        ]
        assert len(discovery_lines) <= 3

    def test_discovery_picks_non_universe_ticker(self) -> None:
        """When article has both universe and non-universe tickers, pick non-universe."""
        articles = [
            _make_article("PLTR and AAPL form partnership", ["PLTR", "AAPL"]),
        ]
        compactor = NewsCompactor()
        compactor.compact(
            articles=articles,
            held_tickers=["AAPL"],
            universe_tickers=UNIVERSE,
        )

        # Article matches held ticker AAPL, so it goes to relevant, not discovery.
        # Test with an article that doesn't match held/movers but has mixed tickers.
        compactor2 = NewsCompactor()
        articles2 = [
            _make_article("PLTR and GOOG partnership announced", ["PLTR", "GOOG"]),
        ]
        result2 = compactor2.compact(
            articles=articles2,
            held_tickers=[],
            top_movers=[],
            universe_tickers=UNIVERSE,
        )

        # GOOG is in universe, PLTR is not → discovery should pick PLTR
        assert "== DISCOVERY (not in universe) ==" in result2
        assert "PLTR" in result2

    def test_no_universe_means_no_discovery(self) -> None:
        """Without universe_tickers, discovery section is not produced."""
        articles = [
            _make_article("PLTR beats earnings", ["PLTR"]),
        ]
        compactor = NewsCompactor()
        result = compactor.compact(
            articles=articles,
            held_tickers=[],
        )

        # No universe_tickers passed, so no discovery filtering
        assert "== DISCOVERY (not in universe) ==" not in result


class TestDiscoveryRanking:
    """Discovery items ranked by actionability keywords."""

    def test_earnings_keyword_ranked_higher(self) -> None:
        """Articles with earnings keywords score higher in discovery."""
        articles = [
            _make_article("PLTR stock price moves slightly", ["PLTR"]),
            _make_article("UBER quarterly earnings beat estimates", ["UBER"]),
        ]
        compactor = NewsCompactor()
        result = compactor.compact(
            articles=articles,
            held_tickers=[],
            universe_tickers=UNIVERSE,
        )

        discovery_section = result.split("== DISCOVERY (not in universe) ==")[1]
        lines = [ln for ln in discovery_section.strip().split("\n") if ln.strip()]
        # UBER (earnings keyword) should appear before PLTR
        assert lines[0].startswith("UBER")

    def test_multi_source_ranked_higher(self) -> None:
        """Articles with multiple sources score higher."""
        articles = [
            _make_article("PLTR minor news update", ["PLTR"]),
            # Two similar articles about UBER → cluster with source_count=2
            _make_article("UBER revenue surge strong demand", ["UBER"]),
            _make_article("UBER revenue surge ride-hailing demand", ["UBER"]),
        ]
        compactor = NewsCompactor()
        result = compactor.compact(
            articles=articles,
            held_tickers=[],
            universe_tickers=UNIVERSE,
        )

        discovery_section = result.split("== DISCOVERY (not in universe) ==")[1]
        lines = [ln for ln in discovery_section.strip().split("\n") if ln.strip()]
        # UBER (2 sources → +10) should rank above PLTR (1 source → +5)
        assert lines[0].startswith("UBER")


class TestCompactorBackwardCompatibility:
    """Existing behavior unchanged when universe_tickers is not provided."""

    def test_compact_without_universe(self) -> None:
        """compact() works identically when universe_tickers is omitted."""
        articles = [
            _make_article("AAPL beats earnings expectations", ["AAPL"]),
            _make_article("Fed holds rates steady amid inflation", []),
        ]
        compactor = NewsCompactor()
        result = compactor.compact(
            articles=articles,
            held_tickers=["AAPL"],
        )

        assert "TICKER|SIGNAL|EVENT|#SRC" in result
        assert "AAPL" in result
        assert "== DISCOVERY (not in universe) ==" not in result

    def test_empty_articles_returns_empty(self) -> None:
        """Empty input produces empty output."""
        compactor = NewsCompactor()
        result = compactor.compact(articles=[], universe_tickers=UNIVERSE)
        assert result == ""

    def test_macro_articles_not_in_discovery(self) -> None:
        """Macro articles go to relevant, not discovery, even with no tickers."""
        articles = [
            _make_article("Fed raises rates to fight inflation", []),
        ]
        compactor = NewsCompactor()
        result = compactor.compact(
            articles=articles,
            held_tickers=[],
            universe_tickers=UNIVERSE,
        )

        # Macro goes to relevant section
        assert "MACRO" in result
        assert "== DISCOVERY (not in universe) ==" not in result

    def test_discovery_only_output(self) -> None:
        """When no relevant articles but discovery exists, only discovery section."""
        articles = [
            _make_article("PLTR earnings beat estimates", ["PLTR"]),
        ]
        compactor = NewsCompactor()
        result = compactor.compact(
            articles=articles,
            held_tickers=[],
            universe_tickers=UNIVERSE,
        )

        # No relevant articles (PLTR not in held/movers, not macro)
        assert "== DISCOVERY (not in universe) ==" in result
        assert "PLTR" in result
        # Header line should NOT appear since no relevant clusters
        lines = result.split("\n")
        assert lines[0] == ""  # starts with blank line before discovery header

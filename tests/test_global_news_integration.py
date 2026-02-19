"""Integration tests for global news sources + sentiment."""

from src.agent.context_manager import ContextManager
from src.data.fear_greed import format_for_context
from src.data.news_article import NewsArticle
from src.data.news_compactor import NewsCompactor


def test_compactor_adds_region_tags():
    articles = [
        NewsArticle(
            headline="TSM supply chain restart boosts chip outlook",
            summary="",
            source="nikkei_asia",
            publisher="Nikkei Asia",
            tickers=["TSM"],
            region="asia",
        ),
        NewsArticle(
            headline="AAPL hits record high on earnings beat",
            summary="",
            source="alpaca",
            publisher="Benzinga",
            tickers=["AAPL"],
            region="us",
        ),
    ]
    compactor = NewsCompactor()
    result = compactor.compact(articles, held_tickers=["AAPL", "TSM"])

    lines = result.strip().split("\n")
    # Asia article should have [ASIA] tag
    asia_lines = [line for line in lines if "[ASIA]" in line]
    assert len(asia_lines) == 1
    assert "TSM" in asia_lines[0]

    # US article should NOT have a region tag
    us_lines = [line for line in lines if "AAPL" in line and "[" not in line.split("|")[0]]
    assert len(us_lines) >= 1


def test_search_news_region_filter():
    import asyncio

    from src.agent.tools.news import _search_news

    context = (
        "TICKER|SIGNAL|EVENT|#SRC\n"
        "AAPL|POS|Record high earnings|2\n"
        "[ASIA] TSM|POS|Supply chain restart|1\n"
        "[CHINA] BABA|NEG|Regulatory pressure|1\n"
    )

    result = asyncio.get_event_loop().run_until_complete(_search_news(context, region="asia"))
    assert any("TSM" in a for a in result["articles"])
    assert result["region"] == "asia"

    result_us = asyncio.get_event_loop().run_until_complete(_search_news(context, region="us"))
    assert any("AAPL" in a for a in result_us["articles"])


def test_morning_trigger_includes_fear_greed():
    cm = ContextManager()
    market = {
        "regime": "BULL",
        "vix": 15.2,
        "spy_change_pct": 0.5,
        "fear_greed": format_for_context(25.0, "Extreme Fear"),
    }
    portfolio = {"total_value": 66000, "cash": 10000, "positions_count": 5}

    msg = cm.build_trigger_message("morning", market, portfolio)
    assert "F&G:" in msg
    assert "Extreme Fear" in msg


def test_midday_trigger_includes_fear_greed():
    cm = ContextManager()
    market = {
        "vix": 18.5,
        "fear_greed": format_for_context(75.0, "Greed"),
    }
    portfolio = {"total_value": 66000, "cash": 10000}

    msg = cm.build_trigger_message("midday", market, portfolio)
    assert "F&G:" in msg
    assert "Greed" in msg

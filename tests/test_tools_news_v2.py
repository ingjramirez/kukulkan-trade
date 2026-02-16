"""Tests for Phase 2 news + Portfolio A tools.

Tests cover: search_historical_news, get_portfolio_a_status, and registration.
"""

from datetime import date
from unittest.mock import MagicMock

import pytest

from src.agent.tools import ToolRegistry
from src.agent.tools.news import (
    _get_portfolio_a_status,
    _search_historical_news,
    register_news_tools,
)
from src.storage.database import Database


@pytest.fixture
async def db():
    database = Database(url="sqlite+aiosqlite:///:memory:")
    await database.init_db()
    yield database
    await database.close()


def _mock_news_fetcher(articles: list[dict] | None = None):
    """Create a mock NewsFetcher with search_relevant method."""
    fetcher = MagicMock()
    if articles is None:
        articles = [
            {"title": "NVDA earnings beat expectations", "ticker": "NVDA", "distance": 0.2},
            {"title": "AI chip demand surges", "ticker": "NVDA", "distance": 0.4},
        ]
    fetcher.search_relevant.return_value = articles
    return fetcher


# ── search_historical_news ───────────────────────────────────────────────────


async def test_historical_news_basic():
    """search_historical_news returns articles from ChromaDB."""
    fetcher = _mock_news_fetcher()
    result = await _search_historical_news(fetcher, "NVDA")
    assert result["ticker"] == "NVDA"
    assert result["total"] == 2
    assert len(result["articles"]) == 2
    assert "NVDA earnings" in result["articles"][0]["title"]
    assert "relevance" in result["articles"][0]


async def test_historical_news_custom_query():
    """search_historical_news uses custom query when provided."""
    fetcher = _mock_news_fetcher()
    result = await _search_historical_news(fetcher, "NVDA", query="AI chip supply shortage")
    assert result["query"] == "AI chip supply shortage"
    fetcher.search_relevant.assert_called_once_with("AI chip supply shortage", n_results=5)


async def test_historical_news_default_query():
    """search_historical_news uses default ticker query."""
    fetcher = _mock_news_fetcher()
    await _search_historical_news(fetcher, "XLK")
    fetcher.search_relevant.assert_called_once_with("XLK recent developments", n_results=5)


async def test_historical_news_empty():
    """search_historical_news handles no results."""
    fetcher = _mock_news_fetcher(articles=[])
    result = await _search_historical_news(fetcher, "NONEXIST")
    assert result["articles"] == []
    assert "message" in result


async def test_historical_news_error():
    """search_historical_news handles ChromaDB failure."""
    fetcher = MagicMock()
    fetcher.search_relevant.side_effect = IOError("connection refused")
    result = await _search_historical_news(fetcher, "NVDA")
    assert "ChromaDB search failed" in result["message"]


async def test_historical_news_n_results_clamping():
    """search_historical_news clamps n_results to 1-10."""
    fetcher = _mock_news_fetcher()
    await _search_historical_news(fetcher, "NVDA", n_results=50)
    fetcher.search_relevant.assert_called_once_with("NVDA recent developments", n_results=10)


# ── get_portfolio_a_status ───────────────────────────────────────────────────


async def test_portfolio_a_status_empty(db: Database):
    """get_portfolio_a_status handles empty Portfolio A."""
    await db.upsert_portfolio("A", cash=33000.0, total_value=33000.0)
    result = await _get_portfolio_a_status(db, "default", {})
    assert result["portfolio"] == "A"
    assert result["total_value"] == 33000.0
    assert result["held_etfs"] == []
    assert "Read-only" in result["note"]


async def test_portfolio_a_status_with_positions(db: Database):
    """get_portfolio_a_status shows held ETFs and P&L."""
    await db.upsert_portfolio("A", cash=10000.0, total_value=33000.0)
    await db.upsert_position("A", "XLK", shares=100, avg_price=200.0)

    result = await _get_portfolio_a_status(db, "default", {"XLK": 220.0})
    assert len(result["held_etfs"]) == 1
    assert result["held_etfs"][0]["ticker"] == "XLK"
    assert result["held_etfs"][0]["pnl_pct"] == 10.0  # (220-200)/200 * 100


async def test_portfolio_a_status_with_rankings(db: Database):
    """get_portfolio_a_status includes momentum rankings."""
    from src.storage.models import MomentumRankingRow

    await db.upsert_portfolio("A", cash=33000.0, total_value=33000.0)

    rankings = [
        MomentumRankingRow(date=date.today(), ticker="XLK", return_63d=12.5, rank=1),
        MomentumRankingRow(date=date.today(), ticker="XLE", return_63d=8.3, rank=2),
        MomentumRankingRow(date=date.today(), ticker="XLF", return_63d=5.1, rank=3),
    ]
    await db.save_momentum_rankings(rankings)

    result = await _get_portfolio_a_status(db, "default", {})
    assert len(result["momentum_rankings_top5"]) == 3
    assert result["momentum_rankings_top5"][0]["ticker"] == "XLK"
    assert result["momentum_rankings_top5"][0]["rank"] == 1


# ── Registration ─────────────────────────────────────────────────────────────


async def test_registration_all_tools(db: Database):
    """register_news_tools registers all tools when all deps provided."""
    await db.upsert_portfolio("A", cash=33000.0, total_value=33000.0)
    fetcher = _mock_news_fetcher()
    registry = ToolRegistry()
    register_news_tools(
        registry,
        "news context here",
        news_fetcher=fetcher,
        db=db,
        tenant_id="default",
        current_prices={"XLK": 200.0},
    )

    names = registry.tool_names
    assert "search_news" in names
    assert "search_historical_news" in names
    assert "get_portfolio_a_status" in names


async def test_registration_minimal():
    """register_news_tools works with just news_context (Phase 32 compat)."""
    registry = ToolRegistry()
    register_news_tools(registry, "news context here")

    names = registry.tool_names
    assert "search_news" in names
    assert "search_historical_news" not in names
    assert "get_portfolio_a_status" not in names

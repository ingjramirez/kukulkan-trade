"""News investigation tools for the agentic loop.

Phase 2 upgrade: 2 tools (1 existing + 1 new).
search_news: Pre-fetched today's headlines.
search_historical_news: ChromaDB semantic search for past context.
"""

from __future__ import annotations

from functools import partial
from typing import Any

import structlog

from src.agent.tools import ToolRegistry

log = structlog.get_logger()


async def _search_news(
    news_context: str,
    ticker: str | None = None,
) -> dict:
    """Filter pre-fetched news by ticker.

    Args:
        news_context: Full news context string from the compactor.
        ticker: Optional ticker to filter for.

    Returns:
        Dict with filtered news lines.
    """
    if not news_context or news_context.strip() == "(no recent news available)":
        return {"articles": [], "message": "No news available for this session"}

    lines = news_context.strip().split("\n")

    if ticker:
        ticker_upper = ticker.upper()
        filtered = [line for line in lines if ticker_upper in line.upper()]
        return {
            "ticker": ticker_upper,
            "articles": filtered if filtered else [f"No news found for {ticker_upper}"],
            "total": len(filtered),
        }

    return {
        "articles": lines[:20],
        "total": len(lines),
    }


# ── search_historical_news (new — ChromaDB semantic search) ──────────────────


async def _search_historical_news(
    news_fetcher: Any,
    ticker: str,
    query: str | None = None,
    n_results: int = 5,
) -> dict:
    """Search ChromaDB for historical news context about a ticker.

    Args:
        news_fetcher: NewsFetcher instance with vector store access.
        ticker: Ticker to search for.
        query: Optional custom query (default: "{ticker} recent developments").
        n_results: Number of results to return (1-10).

    Returns:
        Dict with historical articles.
    """
    n_results = min(max(n_results, 1), 10)  # Clamp 1-10
    search_query = query or f"{ticker} recent developments"

    try:
        results = news_fetcher.search_relevant(search_query, n_results=n_results)
    except (ValueError, KeyError, AttributeError, IOError) as e:
        log.warning("chromadb_search_failed", ticker=ticker, query=search_query, error=str(e))
        return {"ticker": ticker, "articles": [], "message": "ChromaDB search failed"}

    if not results:
        return {"ticker": ticker, "articles": [], "message": f"No historical news for {ticker}"}

    # Filter to relevant ticker if possible
    articles = []
    for article in results:
        entry = {
            "title": article.get("title", ""),
            "ticker": article.get("ticker", ""),
            "published_at": article.get("published_at", ""),
        }
        # Include relevance distance if available
        if "distance" in article:
            entry["relevance"] = round(1.0 - article["distance"], 3) if article["distance"] < 1.0 else 0.0
        articles.append(entry)

    return {
        "ticker": ticker,
        "query": search_query,
        "articles": articles,
        "total": len(articles),
    }


# ── get_portfolio_a_status (new — read-only cross-portfolio visibility) ──────


async def _get_portfolio_a_status(
    db: Any,
    tenant_id: str,
    current_prices: dict[str, float],
) -> dict:
    """Read-only view of Portfolio A: current ETF, momentum rankings, return.

    Claude can see but NOT modify Portfolio A.
    """
    portfolio = await db.get_portfolio("A", tenant_id=tenant_id)
    positions = await db.get_positions("A", tenant_id=tenant_id)
    rankings = await db.get_latest_momentum_rankings()

    # Portfolio summary
    cash = portfolio.cash if portfolio else 0
    total_positions_value = 0.0
    held_etfs = []

    for p in positions:
        price = current_prices.get(p.ticker, p.avg_price)
        value = p.shares * price
        total_positions_value += value
        pnl_pct = ((price - p.avg_price) / p.avg_price) * 100 if p.avg_price > 0 else 0
        held_etfs.append(
            {
                "ticker": p.ticker,
                "shares": p.shares,
                "avg_price": round(p.avg_price, 2),
                "current_price": round(price, 2),
                "pnl_pct": round(pnl_pct, 2),
            }
        )

    total_value = cash + total_positions_value

    # Momentum rankings (top 5)
    top_rankings = [
        {
            "ticker": r.ticker,
            "return_63d": round(r.return_63d, 2),
            "rank": r.rank,
        }
        for r in sorted(rankings, key=lambda x: x.rank)[:5]
    ]

    return {
        "portfolio": "A",
        "strategy": "Momentum (mechanical, rule-based)",
        "total_value": round(total_value, 2),
        "cash": round(cash, 2),
        "held_etfs": held_etfs,
        "momentum_rankings_top5": top_rankings,
        "note": "Read-only. Claude cannot modify Portfolio A.",
    }


# ── get_portfolio_a_history (rotation history for correlation) ─────────────────


async def _get_portfolio_a_history(
    db: Any,
    tenant_id: str,
    n_trades: int = 10,
) -> dict:
    """Read Portfolio A's recent momentum rotation trades.

    Returns the last N trades: date, ticker, side, shares, price, reason.
    Useful for cross-portfolio correlation analysis.
    """
    all_trades = await db.get_trades("A", tenant_id=tenant_id)
    trades = all_trades[:n_trades]  # Already sorted desc by executed_at
    history = []
    for t in trades:
        history.append(
            {
                "date": str(t.executed_at.date()) if t.executed_at else "",
                "ticker": t.ticker,
                "side": t.side,
                "shares": t.shares,
                "price": round(t.price, 2),
                "reason": (t.reason or "")[:200],
            }
        )

    # Also get latest momentum rankings
    rankings = await db.get_latest_momentum_rankings()
    top_rankings = [
        {"ticker": r.ticker, "return_63d": round(r.return_63d, 2), "rank": r.rank}
        for r in sorted(rankings, key=lambda x: x.rank)[:5]
    ]

    return {
        "portfolio": "A",
        "strategy": "Momentum rotation (mechanical, rule-based)",
        "recent_trades": history,
        "momentum_rankings_top5": top_rankings,
        "total_entries": len(history),
        "note": "Read-only. Claude cannot modify Portfolio A.",
    }


# ── Registration ──────────────────────────────────────────────────────────────


def register_news_tools(
    registry: ToolRegistry,
    news_context: str,
    news_fetcher: Any | None = None,
    db: Any | None = None,
    tenant_id: str = "default",
    current_prices: dict[str, float] | None = None,
) -> None:
    """Register news and cross-portfolio tools.

    Args:
        registry: ToolRegistry to register tools on.
        news_context: Pre-compacted news context string.
        news_fetcher: NewsFetcher instance (for historical search, optional).
        db: Database instance (for Portfolio A status, optional).
        tenant_id: Tenant UUID.
        current_prices: Dict of ticker -> current price.
    """
    registry.register(
        name="search_news",
        description="Search today's news articles. Optionally filter by ticker symbol.",
        input_schema={
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "Optional ticker to filter news for",
                },
            },
        },
        handler=partial(_search_news, news_context),
    )

    if news_fetcher is not None:
        registry.register(
            name="search_historical_news",
            description=(
                "Search historical news in ChromaDB. Returns past articles relevant to a ticker for multi-day analysis."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "ticker": {"type": "string", "description": "Ticker to search for"},
                    "query": {
                        "type": "string",
                        "description": "Custom search query (optional, defaults to ticker developments)",
                    },
                    "n_results": {
                        "type": "integer",
                        "description": "Number of results (1-10, default: 5)",
                    },
                },
                "required": ["ticker"],
            },
            handler=partial(_search_historical_news, news_fetcher),
        )

    if db is not None and current_prices is not None:
        registry.register(
            name="get_portfolio_a_status",
            description=(
                "Read-only view of Portfolio A (momentum strategy): current ETF held, "
                "momentum rankings (top 5), and return. Use for cross-portfolio coordination."
            ),
            input_schema={"type": "object", "properties": {}},
            handler=partial(_get_portfolio_a_status, db, tenant_id, current_prices),
        )

    if db is not None:
        registry.register(
            name="get_portfolio_a_history",
            description=(
                "Read Portfolio A's recent momentum rotation trades and current momentum rankings. "
                "Use for cross-portfolio correlation analysis."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "n_trades": {
                        "type": "integer",
                        "description": "Number of recent trades to return (1-20, default: 10)",
                    },
                },
            },
            handler=partial(_get_portfolio_a_history, db, tenant_id),
        )

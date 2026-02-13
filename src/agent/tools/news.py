"""News investigation tool for the agentic loop.

Searches pre-fetched news articles — no additional API calls.
"""

from __future__ import annotations

from functools import partial

from src.agent.tools import ToolRegistry


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


def register_news_tools(
    registry: ToolRegistry,
    news_context: str,
) -> None:
    """Register news search tool with pre-fetched articles.

    Args:
        registry: ToolRegistry to register tools on.
        news_context: Pre-compacted news context string.
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

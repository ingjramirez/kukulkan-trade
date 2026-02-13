"""Tests for news search tool."""

import pytest

from src.agent.tools import ToolRegistry
from src.agent.tools.news import _search_news, register_news_tools


@pytest.mark.asyncio
async def test_search_by_ticker():
    news = "XLK|POS|Tech rally|#3\nXLE|NEG|Oil drops|#2\nXLK|INFO|New product|#1"
    result = await _search_news(news, ticker="XLK")
    assert result["total"] == 2
    assert all("XLK" in line for line in result["articles"])


@pytest.mark.asyncio
async def test_search_no_filter():
    news = "XLK|POS|Tech|#3\nXLE|NEG|Oil|#2"
    result = await _search_news(news)
    assert result["total"] == 2


@pytest.mark.asyncio
async def test_empty_news():
    result = await _search_news("")
    assert result["articles"] == []


@pytest.mark.asyncio
async def test_registration():
    registry = ToolRegistry()
    register_news_tools(registry, "XLK|POS|Test|#1")
    assert "search_news" in registry.tool_names
    result = await registry.execute("search_news", {"ticker": "XLK"})
    assert result["total"] == 1

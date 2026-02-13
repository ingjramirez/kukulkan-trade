"""Tests for ToolRegistry — tool registration and execution."""

import pytest

from src.agent.tools import ToolRegistry


@pytest.fixture
def registry():
    return ToolRegistry()


async def _sample_handler(ticker: str = "SPY") -> dict:
    return {"ticker": ticker, "price": 450.0}


async def _failing_handler() -> dict:
    raise ValueError("test error")


def test_register_and_list(registry):
    registry.register(
        name="get_price",
        description="Get current price",
        input_schema={"type": "object", "properties": {"ticker": {"type": "string"}}},
        handler=_sample_handler,
    )
    assert "get_price" in registry.tool_names


def test_definitions_format(registry):
    registry.register(
        name="get_price",
        description="Get current price for a ticker",
        input_schema={
            "type": "object",
            "properties": {"ticker": {"type": "string"}},
            "required": ["ticker"],
        },
        handler=_sample_handler,
    )
    defs = registry.get_tool_definitions()
    assert len(defs) == 1
    assert defs[0]["name"] == "get_price"
    assert defs[0]["description"] == "Get current price for a ticker"
    assert "properties" in defs[0]["input_schema"]


@pytest.mark.asyncio
async def test_execute(registry):
    registry.register(
        name="get_price",
        description="Get price",
        input_schema={"type": "object", "properties": {"ticker": {"type": "string"}}},
        handler=_sample_handler,
    )
    result = await registry.execute("get_price", {"ticker": "AAPL"})
    assert result == {"ticker": "AAPL", "price": 450.0}


@pytest.mark.asyncio
async def test_execute_unknown_tool(registry):
    with pytest.raises(KeyError, match="Unknown tool"):
        await registry.execute("nonexistent", {})


def test_tool_names(registry):
    assert registry.tool_names == []
    registry.register("a", "desc", {}, _sample_handler)
    registry.register("b", "desc", {}, _sample_handler)
    assert sorted(registry.tool_names) == ["a", "b"]


@pytest.mark.asyncio
async def test_execute_error_propagates(registry):
    registry.register("fail", "fails", {}, _failing_handler)
    with pytest.raises(ValueError, match="test error"):
        await registry.execute("fail", {})

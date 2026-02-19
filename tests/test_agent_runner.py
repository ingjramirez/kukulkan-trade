"""Tests for AgentRunner — agentic tool-use loop."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agent.agent_runner import AgentRunner


def _make_text_response(text: str, input_tokens: int = 100, output_tokens: int = 200):
    """Create a mock Anthropic response with text content."""
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text=text)],
        stop_reason="end_turn",
        model="claude-sonnet-4-6",
        usage=SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens),
    )


def _make_tool_use_response(
    tool_name: str,
    tool_input: dict,
    tool_id: str = "tool_123",
    input_tokens: int = 100,
    output_tokens: int = 200,
):
    """Create a mock Anthropic response requesting tool use."""
    return SimpleNamespace(
        content=[
            SimpleNamespace(type="tool_use", id=tool_id, name=tool_name, input=tool_input),
        ],
        stop_reason="tool_use",
        model="claude-sonnet-4-6",
        usage=SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens),
    )


@pytest.mark.asyncio
async def test_single_turn_no_tools():
    """Model responds with JSON on first turn, no tools needed."""
    response_json = json.dumps(
        {
            "regime_assessment": "Bull market",
            "reasoning": "Strong momentum",
            "trades": [{"ticker": "XLK", "side": "BUY", "weight": 0.15}],
            "risk_notes": "Low risk",
        }
    )
    mock_response = _make_text_response(response_json)

    with patch("src.agent.agent_runner.anthropic") as mock_anthropic:
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.Anthropic.return_value = mock_client
        mock_anthropic.NOT_GIVEN = object()

        runner = AgentRunner(api_key="test-key")
        result = await runner.run("system prompt", "user message")

    assert result.response["regime_assessment"] == "Bull market"
    assert len(result.response["trades"]) == 1
    assert result.turns == 1
    assert result.tool_calls == []


@pytest.mark.asyncio
async def test_multi_turn_tool_calls():
    """Model uses a tool, then responds with JSON."""
    tool_response = _make_tool_use_response("get_price", {"ticker": "XLK"})
    final_response = _make_text_response(
        json.dumps(
            {
                "regime_assessment": "Bull",
                "reasoning": "After checking XLK price",
                "trades": [],
                "risk_notes": "None",
            }
        )
    )

    call_count = 0

    def side_effect(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return tool_response
        return final_response

    with patch("src.agent.agent_runner.anthropic") as mock_anthropic:
        mock_client = MagicMock()
        mock_client.messages.create.side_effect = side_effect
        mock_anthropic.Anthropic.return_value = mock_client
        mock_anthropic.NOT_GIVEN = object()

        runner = AgentRunner(api_key="test-key")

        # Register a tool
        async def mock_get_price(ticker: str = "SPY") -> dict:
            return {"ticker": ticker, "price": 200.0}

        runner.registry.register(
            "get_price",
            "Get price",
            {"type": "object", "properties": {"ticker": {"type": "string"}}},
            mock_get_price,
        )
        result = await runner.run("system", "user")

    assert result.turns == 2
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].tool_name == "get_price"
    assert result.tool_calls[0].success is True


@pytest.mark.asyncio
async def test_max_turns_graceful_exit():
    """When max_turns reached, model is forced to finalize."""
    tool_response = _make_tool_use_response("get_price", {"ticker": "SPY"})
    final_response = _make_text_response(
        json.dumps(
            {
                "regime_assessment": "Forced finalize",
                "reasoning": "Budget reached",
                "trades": [],
                "risk_notes": "",
            }
        )
    )

    call_count = 0

    def side_effect(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        # Always request tools (except the finalize call)
        if "tools" in kwargs and kwargs["tools"] is not None and not isinstance(kwargs["tools"], type):
            return tool_response
        return final_response

    with patch("src.agent.agent_runner.anthropic") as mock_anthropic:
        mock_client = MagicMock()
        mock_client.messages.create.side_effect = side_effect
        mock_anthropic.Anthropic.return_value = mock_client
        mock_anthropic.NOT_GIVEN = object()

        runner = AgentRunner(api_key="test-key", max_turns=2)

        async def mock_tool(**kwargs) -> dict:
            return {"ok": True}

        runner.registry.register("get_price", "Get price", {"type": "object", "properties": {}}, mock_tool)
        result = await runner.run("system", "user")

    assert result.response["trades"] == []
    assert result.turns == 2


@pytest.mark.asyncio
async def test_budget_exceeded():
    """When budget is exceeded, model is forced to finalize."""
    final_response = _make_text_response(
        json.dumps(
            {
                "regime_assessment": "Over budget",
                "reasoning": "Forced",
                "trades": [],
                "risk_notes": "",
            }
        )
    )

    with patch("src.agent.agent_runner.anthropic") as mock_anthropic:
        mock_client = MagicMock()
        mock_client.messages.create.return_value = final_response
        mock_anthropic.Anthropic.return_value = mock_client
        mock_anthropic.NOT_GIVEN = object()

        runner = AgentRunner(api_key="test-key", max_cost_usd=0.0001)
        # Pre-exhaust the budget
        runner._token_tracker.record("claude-sonnet-4-6", 100000, 50000, 0)

        result = await runner.run("system", "user")

    assert "trades" in result.response


@pytest.mark.asyncio
async def test_tool_error_handled():
    """Tool execution errors are captured in the log, not raised."""
    tool_response = _make_tool_use_response("bad_tool", {})
    final_response = _make_text_response(
        json.dumps(
            {
                "regime_assessment": "Ok",
                "reasoning": "Recovered",
                "trades": [],
                "risk_notes": "",
            }
        )
    )

    call_count = 0

    def side_effect(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return tool_response
        return final_response

    with patch("src.agent.agent_runner.anthropic") as mock_anthropic:
        mock_client = MagicMock()
        mock_client.messages.create.side_effect = side_effect
        mock_anthropic.Anthropic.return_value = mock_client
        mock_anthropic.NOT_GIVEN = object()

        runner = AgentRunner(api_key="test-key")

        async def failing_tool() -> dict:
            raise RuntimeError("connection failed")

        runner.registry.register("bad_tool", "Fails", {"type": "object", "properties": {}}, failing_tool)
        result = await runner.run("system", "user")

    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].success is False
    assert "connection failed" in result.tool_calls[0].error


@pytest.mark.asyncio
async def test_json_parse_with_markdown_fences():
    """JSON wrapped in ```json fences is parsed correctly."""
    response_text = '```json\n{"regime_assessment": "test", "reasoning": "ok", "trades": [], "risk_notes": ""}\n```'
    mock_response = _make_text_response(response_text)

    with patch("src.agent.agent_runner.anthropic") as mock_anthropic:
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.Anthropic.return_value = mock_client
        mock_anthropic.NOT_GIVEN = object()

        runner = AgentRunner(api_key="test-key")
        result = await runner.run("system", "user")

    assert result.response["regime_assessment"] == "test"


@pytest.mark.asyncio
async def test_invalid_json_fallback():
    """Invalid JSON produces error structure instead of raising."""
    mock_response = _make_text_response("This is not JSON at all")

    with patch("src.agent.agent_runner.anthropic") as mock_anthropic:
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.Anthropic.return_value = mock_client
        mock_anthropic.NOT_GIVEN = object()

        runner = AgentRunner(api_key="test-key")
        result = await runner.run("system", "user")

    # Raw text used as reasoning when JSON parse fails
    assert result.response["regime_assessment"] == ""
    assert result.response["reasoning"] == "This is not JSON at all"
    assert result.response["trades"] == []


@pytest.mark.asyncio
async def test_turn_delay_sleeps_between_turns():
    """AgentRunner sleeps between turns when turn_delay > 0."""
    tool_response = _make_tool_use_response("get_price", {"ticker": "SPY"})
    final_response = _make_text_response(
        json.dumps({"regime_assessment": "Ok", "reasoning": "Done", "trades": [], "risk_notes": ""})
    )

    call_count = 0

    def side_effect(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return tool_response
        return final_response

    mock_sleep = AsyncMock()
    with (
        patch("src.agent.agent_runner.anthropic") as mock_anthropic,
        patch("asyncio.sleep", mock_sleep),
    ):
        mock_client = MagicMock()
        mock_client.messages.create.side_effect = side_effect
        mock_anthropic.Anthropic.return_value = mock_client
        mock_anthropic.NOT_GIVEN = object()

        runner = AgentRunner(api_key="test-key", turn_delay=2.0)

        async def mock_tool(**kwargs) -> dict:
            return {"price": 200.0}

        runner.registry.register("get_price", "Get price", {"type": "object", "properties": {}}, mock_tool)
        result = await runner.run("system", "user")

    assert result.turns == 2
    # Sleep should have been called once (before turn 2, not before turn 1)
    mock_sleep.assert_called_once_with(2.0)


@pytest.mark.asyncio
async def test_turn_delay_zero_no_sleep():
    """AgentRunner does not sleep when turn_delay=0."""
    response_json = json.dumps(
        {"regime_assessment": "Ok", "reasoning": "Done", "trades": [], "risk_notes": ""}
    )
    mock_response = _make_text_response(response_json)

    mock_sleep = AsyncMock()
    with (
        patch("src.agent.agent_runner.anthropic") as mock_anthropic,
        patch("asyncio.sleep", mock_sleep),
    ):
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.Anthropic.return_value = mock_client
        mock_anthropic.NOT_GIVEN = object()

        runner = AgentRunner(api_key="test-key", turn_delay=0)
        await runner.run("system", "user")

    mock_sleep.assert_not_called()


@pytest.mark.asyncio
async def test_turn_delay_default():
    """AgentRunner default turn_delay is 5.0."""
    runner = AgentRunner(api_key="test-key")
    assert runner._turn_delay == 5.0

"""Tests for AgentRunner modifications to support PersistentAgent.

Tests the messages_override parameter that allows PersistentAgent to
inject conversation history into the agent loop.
"""

from unittest.mock import MagicMock, patch

from src.agent.agent_runner import AgentRunner, AgentRunResult


def _mock_anthropic_response(text: str, stop_reason: str = "end_turn"):
    """Build a mock Anthropic API response."""
    block = MagicMock()
    block.type = "text"
    block.text = text

    usage = MagicMock()
    usage.input_tokens = 500
    usage.output_tokens = 200

    response = MagicMock()
    response.content = [block]
    response.stop_reason = stop_reason
    response.model = "claude-sonnet-4-5-20250929"
    response.usage = usage
    return response


async def test_messages_override_uses_provided_messages():
    """When messages_override is set, runner uses those messages instead of building from user_message."""
    runner = AgentRunner(api_key="test-key", max_turns=1)

    history_messages = [
        {"role": "user", "content": "Previous session context."},
        {"role": "assistant", "content": "I remember the previous session."},
        {"role": "user", "content": "New trigger: morning update."},
    ]

    response_text = '{"regime_assessment": "BULL", "reasoning": "Test", "trades": [], "risk_notes": ""}'
    mock_response = _mock_anthropic_response(response_text)

    with patch("src.agent.agent_runner.anthropic") as mock_anthropic:
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.Anthropic.return_value = mock_client

        result = await runner.run(
            system_prompt="System prompt",
            user_message="This should be ignored",
            messages_override=history_messages,
        )

    # Verify messages.create was called with the override messages (not user_message)
    call_kwargs = mock_client.messages.create.call_args
    sent_messages = call_kwargs.kwargs.get("messages") or call_kwargs[1].get("messages")
    # Should start with the 3 history messages
    assert len(sent_messages) >= 3
    assert sent_messages[0]["content"] == "Previous session context."
    assert sent_messages[1]["content"] == "I remember the previous session."
    assert sent_messages[2]["content"] == "New trigger: morning update."

    assert isinstance(result, AgentRunResult)
    assert result.response["regime_assessment"] == "BULL"


async def test_messages_override_none_falls_back_to_user_message():
    """When messages_override is None, runner builds messages from user_message (default behavior)."""
    runner = AgentRunner(api_key="test-key", max_turns=1)

    response_text = '{"regime_assessment": "BULL", "reasoning": "Test", "trades": [], "risk_notes": ""}'
    mock_response = _mock_anthropic_response(response_text)

    with patch("src.agent.agent_runner.anthropic") as mock_anthropic:
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.Anthropic.return_value = mock_client

        await runner.run(
            system_prompt="System prompt",
            user_message="The actual user message",
        )

    call_kwargs = mock_client.messages.create.call_args
    sent_messages = call_kwargs.kwargs.get("messages") or call_kwargs[1].get("messages")
    assert len(sent_messages) >= 1
    assert sent_messages[0]["content"] == "The actual user message"


async def test_messages_override_raw_messages_include_full_history():
    """raw_messages in result includes the full conversation (override + agent turns)."""
    runner = AgentRunner(api_key="test-key", max_turns=1)

    history_messages = [
        {"role": "user", "content": "Session 1 trigger."},
        {"role": "assistant", "content": "Session 1 response."},
        {"role": "user", "content": "Session 2 trigger."},
    ]

    response_text = '{"regime_assessment": "BULL", "reasoning": "Analysis", "trades": [], "risk_notes": ""}'
    mock_response = _mock_anthropic_response(response_text)

    with patch("src.agent.agent_runner.anthropic") as mock_anthropic:
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.Anthropic.return_value = mock_client

        result = await runner.run(
            system_prompt="System prompt",
            user_message="ignored",
            messages_override=history_messages,
        )

    # raw_messages should include history + assistant response appended
    assert len(result.raw_messages) == 4  # 3 history + 1 assistant response
    assert result.raw_messages[0]["content"] == "Session 1 trigger."
    assert result.raw_messages[-1]["role"] == "assistant"


async def test_messages_override_does_not_mutate_original():
    """messages_override list is copied, so the original is not mutated."""
    runner = AgentRunner(api_key="test-key", max_turns=1)

    original = [
        {"role": "user", "content": "Original message."},
    ]
    original_len = len(original)

    response_text = '{"regime_assessment": "BULL", "reasoning": "Test", "trades": [], "risk_notes": ""}'
    mock_response = _mock_anthropic_response(response_text)

    with patch("src.agent.agent_runner.anthropic") as mock_anthropic:
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.Anthropic.return_value = mock_client

        await runner.run(
            system_prompt="System prompt",
            user_message="ignored",
            messages_override=original,
        )

    # Original list should not be modified
    assert len(original) == original_len

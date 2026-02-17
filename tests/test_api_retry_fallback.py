"""Tests for Anthropic API retry and model fallback logic."""

from datetime import date
from unittest.mock import MagicMock, patch

import pytest

from src.agent.claude_agent import ClaudeAgent


class _FakeUsage:
    input_tokens = 100
    output_tokens = 50


class _FakeResponse:
    content = [MagicMock(text='{"regime_assessment":"ok","reasoning":"ok","trades":[],"risk_notes":"ok"}')]
    usage = _FakeUsage()
    model = "claude-sonnet-4-6"


class _FakeOverloadedError(Exception):
    """Simulates anthropic.APIStatusError with status_code 529."""

    status_code = 529

    def __init__(self):
        super().__init__("overloaded")


def _make_agent() -> ClaudeAgent:
    """Create an agent with a dummy key."""
    agent = ClaudeAgent(api_key="test-key", model="claude-opus-4-6")
    return agent


def _base_kwargs() -> dict:
    return {
        "analysis_date": date(2026, 2, 17),
        "cash": 10000.0,
        "total_value": 66000.0,
        "positions": [],
        "prices": {"SPY": [500.0, 501.0, 502.0, 503.0, 504.0]},
        "tickers": ["SPY"],
        "indicators": {"SPY": {"rsi_14": 55.0, "macd": 0.5, "sma_20": 500.0, "sma_50": 495.0}},
        "recent_trades": [],
    }


class TestClaudeAgentFallback:
    """Tests for ClaudeAgent.analyze() model fallback on server errors."""

    @patch("src.agent.claude_agent.settings")
    @patch("anthropic.Anthropic")
    def test_fallback_on_529(self, mock_client_cls, mock_settings) -> None:
        """When primary model returns 529, fallback model is used."""
        import anthropic

        mock_settings.anthropic_api_key = "test-key"
        mock_settings.agent.max_retries = 5
        mock_settings.agent.fallback_model = "claude-sonnet-4-6"

        # Create a real APIStatusError-like exception
        error = anthropic.APIStatusError(
            message="overloaded",
            response=MagicMock(status_code=529, headers={}),
            body={"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}},
        )

        mock_client = MagicMock()
        mock_client.messages.create.side_effect = [error, _FakeResponse()]
        mock_client_cls.return_value = mock_client

        agent = _make_agent()
        agent._client = mock_client

        result = agent.analyze(**_base_kwargs())

        assert result["regime_assessment"] == "ok"
        assert mock_client.messages.create.call_count == 2
        # Second call should use fallback model
        second_call = mock_client.messages.create.call_args_list[1]
        assert second_call.kwargs["model"] == "claude-sonnet-4-6"

    @patch("src.agent.claude_agent.settings")
    @patch("anthropic.Anthropic")
    def test_no_fallback_on_400(self, mock_client_cls, mock_settings) -> None:
        """Client errors (4xx) should NOT trigger fallback."""
        import anthropic

        mock_settings.anthropic_api_key = "test-key"
        mock_settings.agent.max_retries = 5
        mock_settings.agent.fallback_model = "claude-sonnet-4-6"

        error = anthropic.APIStatusError(
            message="bad request",
            response=MagicMock(status_code=400, headers={}),
            body={"type": "error", "error": {"type": "invalid_request_error"}},
        )

        mock_client = MagicMock()
        mock_client.messages.create.side_effect = error
        mock_client_cls.return_value = mock_client

        agent = _make_agent()
        agent._client = mock_client

        with pytest.raises(anthropic.APIStatusError):
            agent.analyze(**_base_kwargs())

        # Only one call — no fallback attempted
        assert mock_client.messages.create.call_count == 1

    @patch("src.agent.claude_agent.settings")
    @patch("anthropic.Anthropic")
    def test_no_fallback_when_same_model(self, mock_client_cls, mock_settings) -> None:
        """If fallback == primary model, don't retry with same model."""
        import anthropic

        mock_settings.anthropic_api_key = "test-key"
        mock_settings.agent.max_retries = 5
        mock_settings.agent.fallback_model = "claude-opus-4-6"  # Same as primary

        error = anthropic.APIStatusError(
            message="overloaded",
            response=MagicMock(status_code=529, headers={}),
            body={"type": "error", "error": {"type": "overloaded_error"}},
        )

        mock_client = MagicMock()
        mock_client.messages.create.side_effect = error
        mock_client_cls.return_value = mock_client

        agent = _make_agent()
        agent._client = mock_client

        with pytest.raises(anthropic.APIStatusError):
            agent.analyze(**_base_kwargs())

        assert mock_client.messages.create.call_count == 1

    @patch("src.agent.claude_agent.settings")
    @patch("anthropic.Anthropic")
    def test_fallback_on_500(self, mock_client_cls, mock_settings) -> None:
        """Fallback should also trigger on generic 500 errors."""
        import anthropic

        mock_settings.anthropic_api_key = "test-key"
        mock_settings.agent.max_retries = 5
        mock_settings.agent.fallback_model = "claude-sonnet-4-6"

        error = anthropic.APIStatusError(
            message="internal server error",
            response=MagicMock(status_code=500, headers={}),
            body={"type": "error", "error": {"type": "api_error"}},
        )

        mock_client = MagicMock()
        mock_client.messages.create.side_effect = [error, _FakeResponse()]
        mock_client_cls.return_value = mock_client

        agent = _make_agent()
        agent._client = mock_client

        result = agent.analyze(**_base_kwargs())
        assert result["regime_assessment"] == "ok"
        assert mock_client.messages.create.call_count == 2


class TestAgentRunnerFallback:
    """Tests for AgentRunner.run() model fallback on server errors."""

    @pytest.mark.asyncio
    @patch("config.settings.settings")
    @patch("anthropic.Anthropic")
    async def test_agent_runner_fallback_on_529(self, mock_client_cls, mock_settings) -> None:
        """AgentRunner should fallback when the agentic loop call hits 529."""
        import anthropic

        from src.agent.agent_runner import AgentRunner

        mock_settings.agent.max_retries = 5
        mock_settings.agent.fallback_model = "claude-sonnet-4-6"

        error = anthropic.APIStatusError(
            message="overloaded",
            response=MagicMock(status_code=529, headers={}),
            body={"type": "error", "error": {"type": "overloaded_error"}},
        )

        fake_response = MagicMock()
        fake_response.stop_reason = "end_turn"
        fake_response.content = [
            MagicMock(
                type="text",
                text='{"regime_assessment":"ok","reasoning":"ok","trades":[],"risk_notes":"ok"}',
            )
        ]
        fake_response.usage = _FakeUsage()
        fake_response.model = "claude-sonnet-4-6"

        mock_client = MagicMock()
        mock_client.messages.create.side_effect = [error, fake_response]
        mock_client_cls.return_value = mock_client

        runner = AgentRunner(api_key="test-key", model="claude-opus-4-6")
        result = await runner.run(system_prompt="test", user_message="test")

        assert result.response["regime_assessment"] == "ok"
        assert mock_client.messages.create.call_count == 2


class TestMaxRetriesConfig:
    """Tests for max_retries configuration propagation."""

    @patch("src.agent.claude_agent.settings")
    def test_claude_agent_passes_max_retries(self, mock_settings) -> None:
        """ClaudeAgent should pass max_retries to the Anthropic client."""
        mock_settings.anthropic_api_key = "test-key"
        mock_settings.agent.max_retries = 7

        with patch("anthropic.Anthropic") as mock_cls:
            agent = ClaudeAgent(api_key="test-key")
            _ = agent.client  # trigger lazy init
            mock_cls.assert_called_once_with(api_key="test-key", max_retries=7)

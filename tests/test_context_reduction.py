"""Tests for context reduction — history settings and tool result truncation."""

from unittest.mock import AsyncMock, patch

import pytest

from src.agent.context_manager import ContextManager, _truncate_tool_results

# ── History settings via persistent_agent ─────────────────────────────────────


@pytest.mark.asyncio
async def test_recent_sessions_default_2():
    """Persistent agent loads 2 recent sessions by default (not 5)."""
    from src.agent.persistent_agent import PersistentAgent

    mock_db = AsyncMock()
    agent = PersistentAgent(db=mock_db, api_key="test-key")

    mock_store = AsyncMock()
    mock_store.load_summaries.return_value = []
    mock_store.load_recent.return_value = []
    mock_store.mark_session_started.return_value = None
    mock_store.save_session.return_value = None
    mock_store.get_uncompressed_sessions.return_value = []
    agent._store = mock_store

    mock_runner = AsyncMock()
    mock_runner._token_tracker = AsyncMock()
    mock_runner._token_tracker.total_input_tokens = 0
    mock_runner._token_tracker.total_output_tokens = 0
    mock_runner._token_tracker.total_cost_usd = 0.0
    mock_result = AsyncMock()
    mock_result.response = {"trades": []}
    mock_result.tool_calls = []
    mock_result.turns = 1
    mock_result.token_tracker = mock_runner._token_tracker
    mock_result.raw_messages = []
    mock_runner.run.return_value = mock_result

    with patch("config.settings.settings") as mock_settings:
        mock_settings.agent.agent_history_recent_n = 2
        mock_settings.agent.agent_history_summaries_n = 10
        mock_settings.agent.agent_skip_history_triggers = "manual,event"
        await agent._execute_session(
            session_id="test-session",
            trigger_type="morning",
            market_data={},
            portfolio_summary={},
            runner_kwargs={"runner": mock_runner, "system_prompt": "test"},
            pinned_context="",
            strategy_directive="",
        )

    mock_store.load_recent.assert_called_once_with("default", n=2)
    mock_store.load_summaries.assert_called_once_with("default", n=10)


@pytest.mark.asyncio
async def test_manual_run_skips_recent_history():
    """Manual triggers skip recent history (n=0) and reduce summaries (n=5)."""
    from src.agent.persistent_agent import PersistentAgent

    mock_db = AsyncMock()
    agent = PersistentAgent(db=mock_db, api_key="test-key")

    mock_store = AsyncMock()
    mock_store.load_summaries.return_value = []
    mock_store.load_recent.return_value = []
    mock_store.mark_session_started.return_value = None
    mock_store.save_session.return_value = None
    mock_store.get_uncompressed_sessions.return_value = []
    agent._store = mock_store

    mock_runner = AsyncMock()
    mock_runner._token_tracker = AsyncMock()
    mock_runner._token_tracker.total_input_tokens = 0
    mock_runner._token_tracker.total_output_tokens = 0
    mock_runner._token_tracker.total_cost_usd = 0.0
    mock_result = AsyncMock()
    mock_result.response = {"trades": []}
    mock_result.tool_calls = []
    mock_result.turns = 1
    mock_result.token_tracker = mock_runner._token_tracker
    mock_result.raw_messages = []
    mock_runner.run.return_value = mock_result

    with patch("config.settings.settings") as mock_settings:
        mock_settings.agent.agent_history_recent_n = 2
        mock_settings.agent.agent_history_summaries_n = 10
        mock_settings.agent.agent_skip_history_triggers = "manual,event"
        await agent._execute_session(
            session_id="test-session",
            trigger_type="manual",
            market_data={},
            portfolio_summary={},
            runner_kwargs={"runner": mock_runner, "system_prompt": "test"},
            pinned_context="",
            strategy_directive="",
        )

    mock_store.load_recent.assert_called_once_with("default", n=0)
    mock_store.load_summaries.assert_called_once_with("default", n=5)


@pytest.mark.asyncio
async def test_event_trigger_skips_recent_history():
    """Event triggers also skip recent history."""
    from src.agent.persistent_agent import PersistentAgent

    mock_db = AsyncMock()
    agent = PersistentAgent(db=mock_db, api_key="test-key")

    mock_store = AsyncMock()
    mock_store.load_summaries.return_value = []
    mock_store.load_recent.return_value = []
    mock_store.mark_session_started.return_value = None
    mock_store.save_session.return_value = None
    mock_store.get_uncompressed_sessions.return_value = []
    agent._store = mock_store

    mock_runner = AsyncMock()
    mock_runner._token_tracker = AsyncMock()
    mock_runner._token_tracker.total_input_tokens = 0
    mock_runner._token_tracker.total_output_tokens = 0
    mock_runner._token_tracker.total_cost_usd = 0.0
    mock_result = AsyncMock()
    mock_result.response = {"trades": []}
    mock_result.tool_calls = []
    mock_result.turns = 1
    mock_result.token_tracker = mock_runner._token_tracker
    mock_result.raw_messages = []
    mock_runner.run.return_value = mock_result

    with patch("config.settings.settings") as mock_settings:
        mock_settings.agent.agent_history_recent_n = 2
        mock_settings.agent.agent_history_summaries_n = 10
        mock_settings.agent.agent_skip_history_triggers = "manual,event"
        await agent._execute_session(
            session_id="test-session",
            trigger_type="event",
            market_data={},
            portfolio_summary={},
            runner_kwargs={"runner": mock_runner, "system_prompt": "test"},
            pinned_context="",
            strategy_directive="",
        )

    mock_store.load_recent.assert_called_once_with("default", n=0)


@pytest.mark.asyncio
async def test_scheduled_run_normal_context():
    """Scheduled (morning) triggers use normal history settings."""
    from src.agent.persistent_agent import PersistentAgent

    mock_db = AsyncMock()
    agent = PersistentAgent(db=mock_db, api_key="test-key")

    mock_store = AsyncMock()
    mock_store.load_summaries.return_value = []
    mock_store.load_recent.return_value = []
    mock_store.mark_session_started.return_value = None
    mock_store.save_session.return_value = None
    mock_store.get_uncompressed_sessions.return_value = []
    agent._store = mock_store

    mock_runner = AsyncMock()
    mock_runner._token_tracker = AsyncMock()
    mock_runner._token_tracker.total_input_tokens = 0
    mock_runner._token_tracker.total_output_tokens = 0
    mock_runner._token_tracker.total_cost_usd = 0.0
    mock_result = AsyncMock()
    mock_result.response = {"trades": []}
    mock_result.tool_calls = []
    mock_result.turns = 1
    mock_result.token_tracker = mock_runner._token_tracker
    mock_result.raw_messages = []
    mock_runner.run.return_value = mock_result

    with patch("config.settings.settings") as mock_settings:
        mock_settings.agent.agent_history_recent_n = 2
        mock_settings.agent.agent_history_summaries_n = 10
        mock_settings.agent.agent_skip_history_triggers = "manual,event"
        await agent._execute_session(
            session_id="test-session",
            trigger_type="morning",
            market_data={},
            portfolio_summary={},
            runner_kwargs={"runner": mock_runner, "system_prompt": "test"},
            pinned_context="",
            strategy_directive="",
        )

    mock_store.load_recent.assert_called_once_with("default", n=2)
    mock_store.load_summaries.assert_called_once_with("default", n=10)


# ── Tool result truncation ─────────────────────────────────────────────────────


def test_tool_result_truncation_in_replayed_history():
    """Tool result content is truncated in replayed session messages."""
    cm = ContextManager()
    long_content = "X" * 1000
    recent = [
        {
            "session_id": "s1",
            "messages": [
                {"role": "user", "content": "trigger"},
                {
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "tool_use_id": "t1", "content": long_content},
                    ],
                },
                {"role": "assistant", "content": "analysis done"},
            ],
        },
    ]
    messages = cm.build_messages([], recent, "new trigger")

    # The tool result in replayed history should be truncated
    tool_msg = messages[1]
    tool_content = tool_msg["content"][0]["content"]
    assert len(tool_content) < 1000
    assert "[truncated]" in tool_content


def test_current_turn_tool_results_not_truncated():
    """The new trigger message is NOT subject to truncation."""
    cm = ContextManager()
    trigger = "This is a very long trigger message " * 50
    messages = cm.build_messages([], [], trigger)
    assert messages[0]["content"] == trigger


def test_truncation_includes_indicator():
    """Truncated content ends with [truncated]."""
    msg = {
        "role": "user",
        "content": [
            {"type": "tool_result", "tool_use_id": "t1", "content": "Z" * 500},
        ],
    }
    result = _truncate_tool_results(msg, max_chars=100)
    assert result["content"][0]["content"].endswith("[truncated]")
    assert len(result["content"][0]["content"]) == 100 + len(" [truncated]")


def test_truncation_preserves_short_content():
    """Short tool results are not modified."""
    msg = {
        "role": "user",
        "content": [
            {"type": "tool_result", "tool_use_id": "t1", "content": "short"},
        ],
    }
    result = _truncate_tool_results(msg, max_chars=300)
    assert result["content"][0]["content"] == "short"


def test_truncation_does_not_mutate_original():
    """Original message dict is not mutated."""
    original_content = "Z" * 500
    msg = {
        "role": "user",
        "content": [
            {"type": "tool_result", "tool_use_id": "t1", "content": original_content},
        ],
    }
    _truncate_tool_results(msg, max_chars=100)
    assert msg["content"][0]["content"] == original_content


def test_truncation_skips_string_content():
    """Messages with string content are passed through unchanged."""
    msg = {"role": "user", "content": "hello world"}
    result = _truncate_tool_results(msg, max_chars=5)
    assert result is msg  # same object

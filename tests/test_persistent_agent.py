"""Tests for PersistentAgent — persistent agentic pipeline for Portfolio B."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agent.persistent_agent import PersistentAgent, PersistentRunResult
from src.agent.token_tracker import TokenTracker
from src.storage.database import Database


@pytest.fixture
async def db():
    database = Database(url="sqlite+aiosqlite:///:memory:")
    await database.init_db()
    await database.ensure_tenant("t1")
    await database.ensure_tenant("t2")
    yield database
    await database.close()


def _make_agent_run_result(response: dict | None = None, tool_calls: list | None = None):
    """Build a mock AgentRunResult."""
    from src.agent.agent_runner import AgentRunResult

    return AgentRunResult(
        response=response or {"regime_assessment": "BULL", "reasoning": "Test", "trades": [], "risk_notes": ""},
        tool_calls=tool_calls or [],
        turns=2,
        token_tracker=TokenTracker(),
        raw_messages=[
            {"role": "user", "content": "Good morning."},
            {"role": "assistant", "content": "Portfolio healthy."},
        ],
    )


def _make_mock_runner(result=None):
    """Build a mock AgentRunner with a run() AsyncMock."""
    runner = MagicMock()
    runner.run = AsyncMock(return_value=result or _make_agent_run_result())
    runner.registry = MagicMock()
    return runner


async def test_run_session_saves_to_store(db: Database):
    """PersistentAgent saves completed session to ConversationStore."""
    agent = PersistentAgent(db=db, api_key="test-key", tenant_id="t1")
    mock_runner = _make_mock_runner()

    with patch.object(agent, "_compressor") as mock_compressor:
        mock_compressor.compress = AsyncMock(return_value="Summary text")
        result = await agent.run_session(
            trigger_type="morning",
            market_data={"regime": "BULL", "vix": 15.0},
            portfolio_summary={"total_value": 66000, "cash": 25000, "positions_count": 3},
            runner_kwargs={"runner": mock_runner, "system_prompt": "You are Kukulkan."},
        )

    assert isinstance(result, PersistentRunResult)
    assert result.session_id.startswith("t1-morning-")
    assert result.response["regime_assessment"] == "BULL"

    # Verify session was saved
    from src.agent.conversation_store import ConversationStore

    store = ConversationStore(db)
    sessions = await store.load_recent("t1")
    assert len(sessions) == 1
    assert sessions[0]["session_id"] == result.session_id


async def test_run_session_marks_started_then_completed(db: Database):
    """Session is marked started before execution and completed after."""
    agent = PersistentAgent(db=db, api_key="test-key", tenant_id="t1")
    mock_runner = _make_mock_runner()

    with patch.object(agent, "_compressor"):
        result = await agent.run_session(
            trigger_type="midday",
            market_data={"vix": 18.0},
            portfolio_summary={"total_value": 65000, "cash": 24000},
            runner_kwargs={"runner": mock_runner, "system_prompt": ""},
        )

    # No crashed sessions — completed successfully
    from src.agent.conversation_store import ConversationStore

    store = ConversationStore(db)
    crashed = await store.check_crashed_sessions("t1")
    assert len(crashed) == 0

    session = await store.get_session(result.session_id)
    assert session["session_status"] == "completed"


async def test_run_session_passes_messages_override_to_runner(db: Database):
    """PersistentAgent passes messages_override to AgentRunner.run()."""
    agent = PersistentAgent(db=db, api_key="test-key", tenant_id="t1")
    mock_runner = _make_mock_runner()

    with patch.object(agent, "_compressor"):
        await agent.run_session(
            trigger_type="morning",
            market_data={"regime": "BULL", "vix": 15.0},
            portfolio_summary={"total_value": 66000, "cash": 25000, "positions_count": 3},
            runner_kwargs={"runner": mock_runner, "system_prompt": "Base prompt."},
        )

    # Verify run() was called with messages_override
    call_kwargs = mock_runner.run.call_args
    assert "messages_override" in call_kwargs.kwargs
    messages = call_kwargs.kwargs["messages_override"]
    assert isinstance(messages, list)
    assert len(messages) >= 1
    # Last message should be the trigger
    assert messages[-1]["role"] == "user"


async def test_run_session_includes_history_in_messages(db: Database):
    """Second session includes first session's messages in context."""
    agent = PersistentAgent(db=db, api_key="test-key", tenant_id="t1")

    # First session
    mock_runner1 = _make_mock_runner()
    with patch.object(agent, "_compressor"):
        await agent.run_session(
            trigger_type="morning",
            market_data={"regime": "BULL", "vix": 15.0},
            portfolio_summary={"total_value": 66000, "cash": 25000, "positions_count": 3},
            runner_kwargs={"runner": mock_runner1, "system_prompt": ""},
        )

    # Second session
    mock_runner2 = _make_mock_runner()
    with patch.object(agent, "_compressor"):
        await agent.run_session(
            trigger_type="midday",
            market_data={"vix": 16.0},
            portfolio_summary={"total_value": 65500, "cash": 24500},
            runner_kwargs={"runner": mock_runner2, "system_prompt": ""},
        )

    # Second run should have more messages (history + trigger)
    call_kwargs = mock_runner2.run.call_args
    messages = call_kwargs.kwargs["messages_override"]
    # Should include recent session messages + new trigger
    assert len(messages) > 1


async def test_run_session_builds_persistent_system_prompt(db: Database):
    """System prompt includes identity + strategy directive + base prompt."""
    agent = PersistentAgent(db=db, api_key="test-key", tenant_id="t1")
    mock_runner = _make_mock_runner()

    with patch.object(agent, "_compressor"):
        await agent.run_session(
            trigger_type="morning",
            market_data={"regime": "BULL", "vix": 15.0},
            portfolio_summary={"total_value": 66000, "cash": 25000, "positions_count": 3},
            runner_kwargs={"runner": mock_runner, "system_prompt": "Performance: +5% this week."},
            pinned_context="## Active Theses\n- NVDA: Half position",
            strategy_directive="Be conservative in BEAR regimes.",
        )

    call_kwargs = mock_runner.run.call_args
    system_prompt = call_kwargs.kwargs["system_prompt"]
    # Should contain persistent identity
    assert "AI portfolio manager" in system_prompt
    # Should contain pinned context
    assert "NVDA" in system_prompt
    # Should contain strategy directive
    assert "conservative" in system_prompt
    # Should contain base prompt
    assert "Performance" in system_prompt


async def test_run_session_compresses_old_sessions(db: Database):
    """Old sessions are compressed after session execution."""
    agent = PersistentAgent(db=db, api_key="test-key", tenant_id="t1")

    # Create 7 sessions (enough that 2 are beyond the keep_recent=5 window)
    from src.agent.conversation_store import ConversationStore

    store = ConversationStore(db)
    for i in range(7):
        await store.save_session(
            tenant_id="t1",
            session_id=f"old-{i:03d}",
            trigger_type="morning",
            messages=[{"role": "user", "content": f"Session {i}"}],
            token_count=1000,
            cost_usd=0.05,
        )

    # Run new session — should trigger compression of sessions beyond recent 5
    mock_runner = _make_mock_runner()
    with patch.object(agent._compressor, "compress", new_callable=AsyncMock) as mock_compress:
        mock_compress.return_value = "Compressed summary."
        result = await agent.run_session(
            trigger_type="close",
            market_data={"vix": 19.0},
            portfolio_summary={"total_value": 65000, "cash": 24000},
            runner_kwargs={"runner": mock_runner, "system_prompt": ""},
        )

    # Should have compressed some old sessions
    assert result.compressed_count >= 1
    assert mock_compress.call_count >= 1


async def test_run_session_handles_compression_error(db: Database):
    """Compression errors are logged but don't fail the session."""
    agent = PersistentAgent(db=db, api_key="test-key", tenant_id="t1")

    from src.agent.conversation_store import ConversationStore
    from src.agent.session_compressor import CompressionError

    store = ConversationStore(db)
    for i in range(7):
        await store.save_session(
            tenant_id="t1",
            session_id=f"old-{i:03d}",
            trigger_type="morning",
            messages=[{"role": "user", "content": f"Session {i}"}],
            token_count=1000,
            cost_usd=0.05,
        )

    mock_runner = _make_mock_runner()
    with patch.object(agent._compressor, "compress", new_callable=AsyncMock) as mock_compress:
        mock_compress.side_effect = CompressionError("Haiku returned empty summary")
        # Should NOT raise
        result = await agent.run_session(
            trigger_type="morning",
            market_data={"vix": 15.0},
            portfolio_summary={"total_value": 66000, "cash": 25000, "positions_count": 3},
            runner_kwargs={"runner": mock_runner, "system_prompt": ""},
        )

    assert result.compressed_count == 0
    assert result.response is not None


async def test_run_session_with_tool_calls(db: Database):
    """PersistentRunResult includes tool call info when tools were used."""
    from src.agent.agent_runner import ToolCallLog

    tool_calls = [
        ToolCallLog(turn=1, tool_name="get_portfolio_state", tool_input={}, tool_output_preview="data", success=True),
        ToolCallLog(
            turn=2, tool_name="get_price", tool_input={"ticker": "NVDA"}, tool_output_preview="118", success=True
        ),
    ]
    mock_result = _make_agent_run_result(tool_calls=tool_calls)
    agent = PersistentAgent(db=db, api_key="test-key", tenant_id="t1")
    mock_runner = _make_mock_runner(result=mock_result)

    with patch.object(agent, "_compressor"):
        result = await agent.run_session(
            trigger_type="morning",
            market_data={"regime": "BULL", "vix": 15.0},
            portfolio_summary={"total_value": 66000, "cash": 25000, "positions_count": 3},
            runner_kwargs={"runner": mock_runner, "system_prompt": ""},
        )

    assert len(result.tool_calls) == 2
    assert result.tool_summary is not None
    assert result.tool_summary["tools_used"] == 2


async def test_run_session_no_tool_summary_when_no_tools(db: Database):
    """tool_summary is None when no tools were used."""
    agent = PersistentAgent(db=db, api_key="test-key", tenant_id="t1")
    mock_runner = _make_mock_runner()

    with patch.object(agent, "_compressor"):
        result = await agent.run_session(
            trigger_type="morning",
            market_data={"regime": "BULL", "vix": 15.0},
            portfolio_summary={"total_value": 66000, "cash": 25000, "positions_count": 3},
            runner_kwargs={"runner": mock_runner, "system_prompt": ""},
        )

    assert result.tool_summary is None


async def test_persistent_run_result_fields():
    """PersistentRunResult has correct default values."""
    result = PersistentRunResult(
        response={"trades": []},
        session_id="t1-morning-abc123",
    )
    assert result.tool_calls == []
    assert result.turns == 0
    assert result.tool_summary is None
    assert result.compressed_count == 0


async def test_run_session_tenant_isolation(db: Database):
    """Two tenants' sessions don't interfere with each other."""
    agent_t1 = PersistentAgent(db=db, api_key="test-key", tenant_id="t1")
    agent_t2 = PersistentAgent(db=db, api_key="test-key", tenant_id="t2")

    mock_runner = _make_mock_runner()
    with patch.object(agent_t1, "_compressor"), patch.object(agent_t2, "_compressor"):
        r1 = await agent_t1.run_session(
            trigger_type="morning",
            market_data={"regime": "BULL", "vix": 15.0},
            portfolio_summary={"total_value": 66000, "cash": 25000, "positions_count": 3},
            runner_kwargs={"runner": mock_runner, "system_prompt": ""},
        )
        r2 = await agent_t2.run_session(
            trigger_type="morning",
            market_data={"regime": "BEAR", "vix": 28.0},
            portfolio_summary={"total_value": 33000, "cash": 10000, "positions_count": 2},
            runner_kwargs={"runner": mock_runner, "system_prompt": ""},
        )

    assert r1.session_id.startswith("t1-")
    assert r2.session_id.startswith("t2-")

    from src.agent.conversation_store import ConversationStore

    store = ConversationStore(db)
    t1_sessions = await store.load_recent("t1")
    t2_sessions = await store.load_recent("t2")
    assert len(t1_sessions) == 1
    assert len(t2_sessions) == 1

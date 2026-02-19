"""Tests for ConversationStore — persistent agent conversation persistence."""

from datetime import datetime, timedelta, timezone

import pytest

from src.agent.conversation_store import ConversationStore
from src.storage.database import Database


@pytest.fixture
async def db():
    database = Database(url="sqlite+aiosqlite:///:memory:")
    await database.init_db()
    await database.ensure_tenant("t1")
    await database.ensure_tenant("t2")
    yield database
    await database.close()


@pytest.fixture
def store(db):
    return ConversationStore(db)


SAMPLE_MESSAGES = [
    {"role": "user", "content": "Good morning. Markets open. VIX 18.2."},
    {"role": "assistant", "content": "Let me check the portfolio."},
    {"role": "user", "content": [{"type": "tool_result", "content": "portfolio data..."}]},
    {"role": "assistant", "content": "Portfolio looks healthy. No trades needed."},
]


async def test_save_and_load_session(store: ConversationStore):
    """Save a session and load it back."""
    await store.save_session(
        tenant_id="t1",
        session_id="sess-001",
        trigger_type="morning",
        messages=SAMPLE_MESSAGES,
        token_count=5000,
        cost_usd=0.15,
    )

    sessions = await store.load_recent("t1", n=5)
    assert len(sessions) == 1
    assert sessions[0]["session_id"] == "sess-001"
    assert sessions[0]["trigger_type"] == "morning"
    assert sessions[0]["messages"] == SAMPLE_MESSAGES


async def test_load_recent_returns_n_sessions_ordered(store: ConversationStore):
    """load_recent returns sessions in chronological order (oldest first)."""
    for i in range(7):
        await store.save_session(
            tenant_id="t1",
            session_id=f"sess-{i:03d}",
            trigger_type="morning",
            messages=[{"role": "user", "content": f"session {i}"}],
            token_count=1000,
            cost_usd=0.05,
        )

    sessions = await store.load_recent("t1", n=3)
    assert len(sessions) == 3
    # Most recent 3 in chronological order: sess-004, sess-005, sess-006
    assert sessions[0]["session_id"] == "sess-004"
    assert sessions[1]["session_id"] == "sess-005"
    assert sessions[2]["session_id"] == "sess-006"


async def test_load_summaries_returns_compressed_only(store: ConversationStore):
    """load_summaries returns only sessions with a non-null summary."""
    await store.save_session(
        tenant_id="t1",
        session_id="sess-001",
        trigger_type="morning",
        messages=SAMPLE_MESSAGES,
        token_count=5000,
        cost_usd=0.15,
    )
    # No summary yet
    summaries = await store.load_summaries("t1")
    assert len(summaries) == 0

    # Add summary
    await store.save_summary("sess-001", "Bought NVDA at $118, half-size position.")
    summaries = await store.load_summaries("t1")
    assert len(summaries) == 1
    assert summaries[0]["summary"] == "Bought NVDA at $118, half-size position."


async def test_mark_started_then_completed(store: ConversationStore):
    """A session can be marked as started and later completed."""
    await store.mark_session_started("t1", "sess-001", "morning")

    # Should appear in crashed sessions
    crashed = await store.check_crashed_sessions("t1")
    assert "sess-001" in crashed

    # Complete it
    await store.save_session(
        tenant_id="t1",
        session_id="sess-001",
        trigger_type="morning",
        messages=SAMPLE_MESSAGES,
        token_count=5000,
        cost_usd=0.15,
    )

    # No longer crashed
    crashed = await store.check_crashed_sessions("t1")
    assert "sess-001" not in crashed

    # Loadable
    sessions = await store.load_recent("t1")
    assert len(sessions) == 1
    assert sessions[0]["messages"] == SAMPLE_MESSAGES


async def test_crashed_session_detected(store: ConversationStore):
    """Sessions marked as started but never completed are detected."""
    await store.mark_session_started("t1", "sess-crash", "morning")
    await store.save_session(
        tenant_id="t1",
        session_id="sess-ok",
        trigger_type="midday",
        messages=SAMPLE_MESSAGES,
        token_count=1000,
        cost_usd=0.05,
    )

    crashed = await store.check_crashed_sessions("t1")
    assert crashed == ["sess-crash"]


async def test_get_uncompressed_sessions_excludes_recent(store: ConversationStore):
    """get_uncompressed_sessions only returns sessions older than the recent N."""
    for i in range(8):
        await store.save_session(
            tenant_id="t1",
            session_id=f"sess-{i:03d}",
            trigger_type="morning",
            messages=[{"role": "user", "content": f"session {i}"}],
            token_count=1000,
            cost_usd=0.05,
        )

    # Keep 5 recent, should find 3 candidates
    candidates = await store.get_uncompressed_sessions("t1", keep_recent=5)
    assert len(candidates) == 3
    session_ids = [c["session_id"] for c in candidates]
    assert "sess-000" in session_ids
    assert "sess-001" in session_ids
    assert "sess-002" in session_ids


async def test_save_summary_updates_session(store: ConversationStore):
    """save_summary sets the summary field on an existing session."""
    await store.save_session(
        tenant_id="t1",
        session_id="sess-001",
        trigger_type="morning",
        messages=SAMPLE_MESSAGES,
        token_count=5000,
        cost_usd=0.15,
    )

    await store.save_summary("sess-001", "Summary of session 001.")

    session = await store.get_session("sess-001")
    assert session is not None
    assert session["summary"] == "Summary of session 001."
    # Messages should still be there
    assert session["messages"] == SAMPLE_MESSAGES


async def test_cleanup_old_messages_preserves_summary(store: ConversationStore, db: Database):
    """cleanup_old_messages clears messages_json but keeps summary."""
    await store.save_session(
        tenant_id="t1",
        session_id="sess-old",
        trigger_type="morning",
        messages=SAMPLE_MESSAGES,
        token_count=5000,
        cost_usd=0.15,
    )
    await store.save_summary("sess-old", "Old session summary.")

    # Backdate the session to make it old enough for cleanup
    async with db.session() as s:
        from sqlalchemy import update

        from src.storage.models import AgentConversationRow

        await s.execute(
            update(AgentConversationRow)
            .where(AgentConversationRow.session_id == "sess-old")
            .values(created_at=datetime.now(timezone.utc) - timedelta(days=60))
        )
        await s.commit()

    count = await store.cleanup_old_messages("t1", days=30)
    assert count == 1

    session = await store.get_session("sess-old")
    assert session["summary"] == "Old session summary."
    assert session["messages"] == []  # Messages cleared


async def test_tenant_isolation(store: ConversationStore):
    """Two tenants see only their own sessions."""
    await store.save_session(
        tenant_id="t1",
        session_id="t1-sess",
        trigger_type="morning",
        messages=[{"role": "user", "content": "tenant 1"}],
        token_count=1000,
        cost_usd=0.05,
    )
    await store.save_session(
        tenant_id="t2",
        session_id="t2-sess",
        trigger_type="morning",
        messages=[{"role": "user", "content": "tenant 2"}],
        token_count=1000,
        cost_usd=0.05,
    )

    t1_sessions = await store.load_recent("t1")
    t2_sessions = await store.load_recent("t2")
    assert len(t1_sessions) == 1
    assert len(t2_sessions) == 1
    assert t1_sessions[0]["messages"][0]["content"] == "tenant 1"
    assert t2_sessions[0]["messages"][0]["content"] == "tenant 2"


async def test_empty_state_returns_empty_lists(store: ConversationStore):
    """All load methods return empty lists when no sessions exist."""
    assert await store.load_recent("nonexistent") == []
    assert await store.load_summaries("nonexistent") == []
    assert await store.check_crashed_sessions("nonexistent") == []
    assert await store.get_uncompressed_sessions("nonexistent") == []


async def test_session_id_uniqueness(store: ConversationStore):
    """Attempting to create two sessions with the same ID raises an error."""
    await store.save_session(
        tenant_id="t1",
        session_id="sess-dup",
        trigger_type="morning",
        messages=SAMPLE_MESSAGES,
        token_count=1000,
        cost_usd=0.05,
    )

    # mark_session_started creates a new row, should fail on unique constraint
    with pytest.raises(Exception):
        await store.mark_session_started("t1", "sess-dup", "midday")


async def test_messages_json_roundtrip(store: ConversationStore):
    """Messages with complex content (tool_use blocks) survive JSON roundtrip."""
    complex_messages = [
        {"role": "user", "content": "Check NVDA."},
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Let me look up NVDA."},
                {"type": "tool_use", "id": "tool-1", "name": "get_price", "input": {"ticker": "NVDA"}},
            ],
        },
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "tool-1", "content": '{"price": 118.50}'}],
        },
        {"role": "assistant", "content": "NVDA is at $118.50."},
    ]

    await store.save_session(
        tenant_id="t1",
        session_id="sess-complex",
        trigger_type="morning",
        messages=complex_messages,
        token_count=3000,
        cost_usd=0.10,
    )

    sessions = await store.load_recent("t1")
    assert sessions[0]["messages"] == complex_messages


async def test_load_recent_skips_crashed_sessions(store: ConversationStore):
    """load_recent does not include sessions with status='started'."""
    await store.mark_session_started("t1", "sess-crash", "morning")
    await store.save_session(
        tenant_id="t1",
        session_id="sess-ok",
        trigger_type="midday",
        messages=SAMPLE_MESSAGES,
        token_count=1000,
        cost_usd=0.05,
    )

    sessions = await store.load_recent("t1")
    assert len(sessions) == 1
    assert sessions[0]["session_id"] == "sess-ok"


async def test_cleanup_respects_days_threshold(store: ConversationStore, db: Database):
    """cleanup_old_messages only cleans sessions older than the threshold."""
    # Session within threshold (recent)
    await store.save_session(
        tenant_id="t1",
        session_id="sess-recent",
        trigger_type="morning",
        messages=SAMPLE_MESSAGES,
        token_count=1000,
        cost_usd=0.05,
    )
    await store.save_summary("sess-recent", "Recent summary.")

    # Session outside threshold (old)
    await store.save_session(
        tenant_id="t1",
        session_id="sess-old",
        trigger_type="morning",
        messages=SAMPLE_MESSAGES,
        token_count=1000,
        cost_usd=0.05,
    )
    await store.save_summary("sess-old", "Old summary.")

    # Backdate the old session
    async with db.session() as s:
        from sqlalchemy import update

        from src.storage.models import AgentConversationRow

        await s.execute(
            update(AgentConversationRow)
            .where(AgentConversationRow.session_id == "sess-old")
            .values(created_at=datetime.now(timezone.utc) - timedelta(days=60))
        )
        await s.commit()

    count = await store.cleanup_old_messages("t1", days=30)
    assert count == 1

    # Recent session messages intact
    recent = await store.get_session("sess-recent")
    assert recent["messages"] == SAMPLE_MESSAGES

    # Old session messages cleared
    old = await store.get_session("sess-old")
    assert old["messages"] == []


async def test_list_sessions(store: ConversationStore):
    """list_sessions returns lightweight session info."""
    await store.save_session(
        tenant_id="t1",
        session_id="sess-001",
        trigger_type="morning",
        messages=SAMPLE_MESSAGES,
        token_count=5000,
        cost_usd=0.15,
    )

    sessions = await store.list_sessions("t1")
    assert len(sessions) == 1
    assert sessions[0]["session_id"] == "sess-001"
    assert sessions[0]["token_count"] == 5000
    assert sessions[0]["cost_usd"] == 0.15
    assert "messages" not in sessions[0]  # Lightweight, no messages


async def test_get_session_returns_none_for_nonexistent(store: ConversationStore):
    """get_session returns None for a session ID that doesn't exist."""
    result = await store.get_session("nonexistent-id")
    assert result is None


async def test_mark_session_failed_updates_status_and_tokens(store: ConversationStore):
    """mark_session_failed transitions started → failed with partial tokens."""
    await store.mark_session_started("t1", "sess-fail", "morning")

    await store.mark_session_failed("sess-fail", token_count=1500, cost_usd=0.04)

    session = await store.get_session("sess-fail")
    assert session is not None
    assert session["session_status"] == "failed"
    assert session["token_count"] == 1500
    assert session["cost_usd"] == 0.04


async def test_mark_session_failed_not_in_crashed(store: ConversationStore):
    """Failed sessions are not returned by check_crashed_sessions (only 'started' are)."""
    await store.mark_session_started("t1", "sess-fail", "morning")
    await store.mark_session_failed("sess-fail", token_count=500, cost_usd=0.01)

    crashed = await store.check_crashed_sessions("t1")
    assert "sess-fail" not in crashed


async def test_mark_session_failed_not_in_load_recent(store: ConversationStore):
    """Failed sessions are not returned by load_recent (only 'completed' are)."""
    await store.mark_session_started("t1", "sess-fail", "morning")
    await store.mark_session_failed("sess-fail", token_count=500, cost_usd=0.01)

    sessions = await store.load_recent("t1")
    assert len(sessions) == 0


async def test_mark_session_failed_nonexistent_is_noop(store: ConversationStore):
    """mark_session_failed on a nonexistent session doesn't raise."""
    # Should not raise — just a no-op update affecting 0 rows
    await store.mark_session_failed("nonexistent", token_count=0, cost_usd=0.0)

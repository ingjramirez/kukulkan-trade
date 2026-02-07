"""Tests for the agent memory system."""

from datetime import datetime, timedelta
from unittest.mock import MagicMock

import pytest

from src.agent.memory import (
    MAX_AGENT_NOTES,
    MAX_SHORT_TERM,
    MAX_WEEKLY_SUMMARIES,
    AgentMemoryManager,
)
from src.storage.database import Database

# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
async def db():
    """In-memory database for testing."""
    test_db = Database(url="sqlite+aiosqlite:///:memory:")
    await test_db.init_db()
    yield test_db
    await test_db.close()


@pytest.fixture
def memory_manager():
    return AgentMemoryManager()


# ── DB CRUD Tests ─────────────────────────────────────────────────────────────


async def test_upsert_and_get_agent_memory(db):
    """Upsert creates, then updates on same category+key."""
    await db.upsert_agent_memory("short_term", "2026-01-15", "first content")
    rows = await db.get_agent_memories("short_term")
    assert len(rows) == 1
    assert rows[0].content == "first content"

    # Upsert same key — updates content
    await db.upsert_agent_memory("short_term", "2026-01-15", "updated content")
    rows = await db.get_agent_memories("short_term")
    assert len(rows) == 1
    assert rows[0].content == "updated content"


async def test_get_memories_by_category(db):
    """get_agent_memories filters by category."""
    await db.upsert_agent_memory("short_term", "k1", "st content")
    await db.upsert_agent_memory("agent_note", "k2", "note content")
    await db.upsert_agent_memory("weekly_summary", "k3", "weekly content")

    st = await db.get_agent_memories("short_term")
    assert len(st) == 1
    assert st[0].key == "k1"

    notes = await db.get_agent_memories("agent_note")
    assert len(notes) == 1
    assert notes[0].key == "k2"


async def test_delete_expired_memories(db):
    """Expired memories are cleaned up."""
    past = datetime.utcnow() - timedelta(hours=1)
    future = datetime.utcnow() + timedelta(hours=1)

    await db.upsert_agent_memory("short_term", "old", "expired", expires_at=past)
    await db.upsert_agent_memory("short_term", "new", "valid", expires_at=future)
    await db.upsert_agent_memory("agent_note", "no_expiry", "permanent")

    deleted = await db.delete_expired_memories()
    assert deleted == 1

    remaining = await db.get_agent_memories("short_term")
    assert len(remaining) == 1
    assert remaining[0].key == "new"

    # Permanent notes unaffected
    notes = await db.get_agent_memories("agent_note")
    assert len(notes) == 1


async def test_get_all_agent_memory_context(db):
    """get_all_agent_memory_context returns all 3 tiers."""
    await db.upsert_agent_memory("short_term", "d1", "st1")
    await db.upsert_agent_memory("weekly_summary", "w1", "ws1")
    await db.upsert_agent_memory("agent_note", "n1", "note1")

    ctx = await db.get_all_agent_memory_context()
    assert len(ctx["short_term"]) == 1
    assert len(ctx["weekly_summary"]) == 1
    assert len(ctx["agent_note"]) == 1


# ── Memory Manager Tests ──────────────────────────────────────────────────────


def test_build_memory_prompt_empty(memory_manager):
    """Empty memories produce empty string."""
    result = memory_manager.build_memory_prompt({
        "short_term": [],
        "weekly_summary": [],
        "agent_note": [],
    })
    assert result == ""


def test_build_memory_prompt_with_data(memory_manager):
    """Memory prompt includes all 3 tiers when data is present."""
    st = MagicMock(key="2026-01-15", content="Bull regime | Tech rotation thesis")
    ws = MagicMock(key="week_2026-02", content="Learned to hold winners longer")
    note = MagicMock(key="thesis-tech", content="XLK showing strength")

    result = memory_manager.build_memory_prompt({
        "short_term": [st],
        "weekly_summary": [ws],
        "agent_note": [note],
    })

    assert "## Memory" in result
    assert "### Recent Decisions" in result
    assert "2026-01-15" in result
    assert "### Weekly Lessons" in result
    assert "week_2026-02" in result
    assert "### Your Notes" in result
    assert "thesis-tech" in result


def test_build_memory_prompt_partial(memory_manager):
    """Only non-empty tiers appear in the prompt."""
    note = MagicMock(key="lesson-1", content="Don't sell too early")

    result = memory_manager.build_memory_prompt({
        "short_term": [],
        "weekly_summary": [],
        "agent_note": [note],
    })

    assert "## Memory" in result
    assert "### Your Notes" in result
    assert "### Recent Decisions" not in result
    assert "### Weekly Lessons" not in result


def test_build_memory_prompt_respects_limits(memory_manager):
    """Only the last N entries per tier are included."""
    # Create more than MAX_SHORT_TERM entries
    entries = [
        MagicMock(key=f"d{i}", content=f"decision {i}")
        for i in range(MAX_SHORT_TERM + 3)
    ]

    result = memory_manager.build_memory_prompt({
        "short_term": entries,
        "weekly_summary": [],
        "agent_note": [],
    })

    # Should only include the last MAX_SHORT_TERM
    assert f"d{MAX_SHORT_TERM + 2}" in result  # last one
    assert "d0" not in result  # first one pruned


async def test_save_short_term(db, memory_manager):
    """save_short_term extracts and stores a compact summary."""
    response = {
        "regime_assessment": "Risk-on bull market",
        "reasoning": (
            "Tech sector showing momentum, rotating into XLK "
            "and QQQ based on strong breadth signals."
        ),
        "trades": [
            {"ticker": "XLK", "side": "BUY", "weight": 0.15, "reason": "momentum"},
            {"ticker": "GLD", "side": "SELL", "weight": 0.0, "reason": "exit"},
        ],
    }

    await memory_manager.save_short_term(db, "2026-01-15", response)

    memories = await db.get_agent_memories("short_term")
    assert len(memories) == 1
    assert "Risk-on bull market" in memories[0].content
    assert "XLK" in memories[0].content
    assert memories[0].key == "2026-01-15"


async def test_save_short_term_prunes_to_max(db, memory_manager):
    """Oldest short-term entries are pruned when exceeding MAX_SHORT_TERM."""
    for i in range(MAX_SHORT_TERM + 2):
        response = {
            "regime_assessment": f"Regime {i}",
            "reasoning": f"Reasoning {i}",
            "trades": [],
        }
        await memory_manager.save_short_term(db, f"2026-01-{10+i:02d}", response)

    memories = await db.get_agent_memories("short_term")
    assert len(memories) == MAX_SHORT_TERM

    # Most recent should be kept
    keys = [m.key for m in memories]
    last_date = f"2026-01-{10 + MAX_SHORT_TERM + 1:02d}"
    assert last_date in keys


async def test_save_agent_notes(db, memory_manager):
    """save_agent_notes upserts by key."""
    notes = [
        {"key": "thesis-tech", "content": "XLK showing relative strength"},
        {"key": "lesson-timing", "content": "Don't sell on first red day"},
    ]

    await memory_manager.save_agent_notes(db, notes)

    all_notes = await db.get_agent_memories("agent_note")
    assert len(all_notes) == 2

    # Update one note
    await memory_manager.save_agent_notes(
        db, [{"key": "thesis-tech", "content": "XLK weakening, watch for breakdown"}]
    )

    all_notes = await db.get_agent_memories("agent_note")
    assert len(all_notes) == 2
    tech_note = [n for n in all_notes if n.key == "thesis-tech"][0]
    assert "weakening" in tech_note.content


async def test_save_agent_notes_enforces_max(db, memory_manager):
    """Notes are capped at MAX_AGENT_NOTES."""
    for i in range(MAX_AGENT_NOTES + 3):
        await memory_manager.save_agent_notes(
            db, [{"key": f"note-{i}", "content": f"Content {i}"}]
        )

    all_notes = await db.get_agent_memories("agent_note")
    assert len(all_notes) == MAX_AGENT_NOTES


async def test_save_agent_notes_skips_invalid(db, memory_manager):
    """Notes with missing key or content are skipped."""
    notes = [
        {"key": "", "content": "no key"},
        {"key": "valid", "content": ""},
        {"key": "good", "content": "good note"},
    ]

    await memory_manager.save_agent_notes(db, notes)

    all_notes = await db.get_agent_memories("agent_note")
    assert len(all_notes) == 1
    assert all_notes[0].key == "good"


async def test_save_agent_notes_empty_list(db, memory_manager):
    """Empty notes list is a no-op."""
    await memory_manager.save_agent_notes(db, [])
    all_notes = await db.get_agent_memories("agent_note")
    assert len(all_notes) == 0


async def test_run_weekly_compaction(db, memory_manager):
    """Weekly compaction calls Claude and saves a summary."""
    # Seed some short-term memories
    await db.upsert_agent_memory("short_term", "2026-01-13", "Bull | Tech up | BUY XLK")
    await db.upsert_agent_memory("short_term", "2026-01-14", "Bull | Continued momentum")
    await db.upsert_agent_memory("short_term", "2026-01-15", "Rotation | Shifting to value")

    # Mock the Claude agent
    mock_agent = MagicMock()
    mock_response = MagicMock()
    mock_response.content = [MagicMock(
        text="Tech rotation was the key theme this week. XLK outperformed."
    )]
    mock_agent.client.messages.create.return_value = mock_response

    await memory_manager.run_weekly_compaction(db, mock_agent)

    # Verify Claude was called
    mock_agent.client.messages.create.assert_called_once()
    call_kwargs = mock_agent.client.messages.create.call_args
    assert "claude-haiku" in call_kwargs.kwargs["model"]

    # Verify summary was saved
    weekly = await db.get_agent_memories("weekly_summary")
    assert len(weekly) == 1
    assert "Tech rotation" in weekly[0].content
    assert weekly[0].key.startswith("week_")


async def test_run_weekly_compaction_no_data(db, memory_manager):
    """Compaction is skipped when there are no short-term memories."""
    mock_agent = MagicMock()

    await memory_manager.run_weekly_compaction(db, mock_agent)

    # Claude should not be called
    mock_agent.client.messages.create.assert_not_called()


async def test_run_weekly_compaction_prunes_old_summaries(db, memory_manager):
    """Only the last MAX_WEEKLY_SUMMARIES are kept."""
    # Seed more than MAX weekly summaries
    for i in range(MAX_WEEKLY_SUMMARIES + 2):
        await db.upsert_agent_memory(
            "weekly_summary", f"week_2026-{i:02d}", f"Summary {i}"
        )

    # Add a short-term memory so compaction runs
    await db.upsert_agent_memory("short_term", "2026-01-15", "Some decision")

    mock_agent = MagicMock()
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="New week summary.")]
    mock_agent.client.messages.create.return_value = mock_response

    await memory_manager.run_weekly_compaction(db, mock_agent)

    weekly = await db.get_agent_memories("weekly_summary")
    assert len(weekly) <= MAX_WEEKLY_SUMMARIES


# ── System Prompt Integration ────────────────────────────────────────────────


def test_build_system_prompt_with_memory():
    """Memory context is included in the system prompt."""
    from src.agent.claude_agent import build_system_prompt

    prompt = build_system_prompt(
        performance_stats="Cumulative: +5.2%",
        memory_context="## Memory\n### Your Notes\n- [thesis] XLK strong",
    )

    assert "## Memory" in prompt
    assert "XLK strong" in prompt
    assert "+5.2%" in prompt


def test_build_system_prompt_without_memory():
    """System prompt works fine with no memory."""
    from src.agent.claude_agent import build_system_prompt

    prompt = build_system_prompt(performance_stats="Cumulative: +3%")
    assert "## Memory" not in prompt
    assert "+3%" in prompt


# ── End-to-End Flow ──────────────────────────────────────────────────────────


async def test_end_to_end_memory_flow(db, memory_manager):
    """Full cycle: save decisions, build prompt, verify memory in prompt."""
    # Simulate 3 days of decisions
    for i, day in enumerate(["2026-01-13", "2026-01-14", "2026-01-15"]):
        response = {
            "regime_assessment": f"Regime day {i}",
            "reasoning": f"Analysis for day {i}",
            "trades": [{"ticker": "XLK", "side": "BUY", "weight": 0.1}],
            "memory_notes": [
                {"key": f"note-{i}", "content": f"Observation from day {i}"},
            ],
        }
        await memory_manager.save_short_term(db, day, response)
        await memory_manager.save_agent_notes(db, response.get("memory_notes", []))

    # Build the memory prompt
    memories = await db.get_all_agent_memory_context()
    prompt = memory_manager.build_memory_prompt(memories)

    # Verify all tiers present
    assert "### Recent Decisions" in prompt
    assert "### Your Notes" in prompt

    # Short-term should have exactly MAX_SHORT_TERM
    st = memories["short_term"]
    assert len(st) == MAX_SHORT_TERM

    # Notes should have 3
    notes = memories["agent_note"]
    assert len(notes) == 3

"""Tests for the weekly compaction evaluation prompt (Fix #5)."""

from unittest.mock import MagicMock

import pytest

from src.agent.memory import AgentMemoryManager
from src.storage.database import Database


@pytest.fixture
async def db():
    database = Database(url="sqlite+aiosqlite:///:memory:")
    await database.init_db()
    yield database
    await database.close()


async def _seed_short_term(db: Database, count: int = 3, tenant_id: str = "default") -> None:
    """Seed short-term memories for compaction."""
    for i in range(count):
        await db.upsert_agent_memory(
            category="short_term",
            key=f"2026-02-{10 + i:02d}",
            content="BULL | Bought XLK for momentum | Trades: BUY XLK @15%",
            tenant_id=tenant_id,
        )


class TestWeeklyCompactionPrompt:
    """Verify the evaluation prompt structure."""

    async def test_prompt_contains_evaluate_questions(self, db: Database) -> None:
        """The compaction prompt should contain 4 evaluation questions."""
        await _seed_short_term(db)

        manager = AgentMemoryManager()
        mock_agent = MagicMock()

        captured_prompt = {}

        def capture_create(**kwargs):
            captured_prompt["messages"] = kwargs.get("messages", [])
            mock_response = MagicMock()
            mock_response.content = [MagicMock(text="Weekly evaluation summary.")]
            return mock_response

        mock_agent.client.messages.create = capture_create

        await manager.run_weekly_compaction(db, mock_agent)

        prompt_text = captured_prompt["messages"][0]["content"]
        assert "Evaluate" in prompt_text
        assert "Which decisions worked" in prompt_text
        assert "What went wrong" in prompt_text
        assert "Patterns" in prompt_text
        assert "What should change" in prompt_text

    async def test_track_record_text_included(self, db: Database) -> None:
        """When track_record_text is provided, it's included in the prompt."""
        await _seed_short_term(db)

        manager = AgentMemoryManager()
        mock_agent = MagicMock()

        captured_prompt = {}

        def capture_create(**kwargs):
            captured_prompt["messages"] = kwargs.get("messages", [])
            mock_response = MagicMock()
            mock_response.content = [MagicMock(text="Summary.")]
            return mock_response

        mock_agent.client.messages.create = capture_create

        await manager.run_weekly_compaction(
            db,
            mock_agent,
            track_record_text="Win rate: 67% (2W/1L/0S from 3 trades)",
        )

        prompt_text = captured_prompt["messages"][0]["content"]
        assert "Track Record:" in prompt_text
        assert "Win rate: 67%" in prompt_text

    async def test_outcome_summary_included(self, db: Database) -> None:
        """When outcome_summary is provided, it's included in the prompt."""
        await _seed_short_term(db)

        manager = AgentMemoryManager()
        mock_agent = MagicMock()

        captured_prompt = {}

        def capture_create(**kwargs):
            captured_prompt["messages"] = kwargs.get("messages", [])
            mock_response = MagicMock()
            mock_response.content = [MagicMock(text="Summary.")]
            return mock_response

        mock_agent.client.messages.create = capture_create

        await manager.run_weekly_compaction(
            db,
            mock_agent,
            outcome_summary="- XLK (BUY): +5.0% in 5d [OPEN, high conviction]",
        )

        prompt_text = captured_prompt["messages"][0]["content"]
        assert "Trade Outcomes" in prompt_text
        assert "XLK" in prompt_text

    async def test_skips_when_no_data(self, db: Database) -> None:
        """Compaction is skipped when there are no short-term memories."""
        manager = AgentMemoryManager()
        mock_agent = MagicMock()

        await manager.run_weekly_compaction(db, mock_agent)

        # Agent should not have been called
        mock_agent.client.messages.create.assert_not_called()

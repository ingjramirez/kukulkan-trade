"""Tests for the weekly compaction evaluation prompt (Fix #5)."""

from unittest.mock import AsyncMock, patch

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

    @patch("src.agent.claude_invoker.claude_cli_call", new_callable=AsyncMock)
    async def test_prompt_contains_evaluate_questions(self, mock_cli, db: Database) -> None:
        """The compaction prompt should contain 4 evaluation questions."""
        await _seed_short_term(db)

        manager = AgentMemoryManager()
        mock_cli.return_value = "Weekly evaluation summary."

        await manager.run_weekly_compaction(db)

        prompt_text = mock_cli.call_args.kwargs["prompt"]
        assert "Evaluate" in prompt_text
        assert "Which decisions worked" in prompt_text
        assert "What went wrong" in prompt_text
        assert "Patterns" in prompt_text
        assert "What should change" in prompt_text

    @patch("src.agent.claude_invoker.claude_cli_call", new_callable=AsyncMock)
    async def test_track_record_text_included(self, mock_cli, db: Database) -> None:
        """When track_record_text is provided, it's included in the prompt."""
        await _seed_short_term(db)

        manager = AgentMemoryManager()
        mock_cli.return_value = "Summary."

        await manager.run_weekly_compaction(
            db,
            track_record_text="Win rate: 67% (2W/1L/0S from 3 trades)",
        )

        prompt_text = mock_cli.call_args.kwargs["prompt"]
        assert "Track Record:" in prompt_text
        assert "Win rate: 67%" in prompt_text

    @patch("src.agent.claude_invoker.claude_cli_call", new_callable=AsyncMock)
    async def test_outcome_summary_included(self, mock_cli, db: Database) -> None:
        """When outcome_summary is provided, it's included in the prompt."""
        await _seed_short_term(db)

        manager = AgentMemoryManager()
        mock_cli.return_value = "Summary."

        await manager.run_weekly_compaction(
            db,
            outcome_summary="- XLK (BUY): +5.0% in 5d [OPEN, high conviction]",
        )

        prompt_text = mock_cli.call_args.kwargs["prompt"]
        assert "Trade Outcomes" in prompt_text
        assert "XLK" in prompt_text

    @patch("src.agent.claude_invoker.claude_cli_call", new_callable=AsyncMock)
    async def test_skips_when_no_data(self, mock_cli, db: Database) -> None:
        """Compaction is skipped when there are no short-term memories."""
        manager = AgentMemoryManager()

        await manager.run_weekly_compaction(db)

        # CLI should not have been called
        mock_cli.assert_not_called()

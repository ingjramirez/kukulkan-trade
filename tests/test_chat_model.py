"""Tests for ChatMessageRow ORM model and CRUD methods."""

from __future__ import annotations

import json
from datetime import datetime, timedelta

import pytest

from src.storage.database import Database
from src.storage.models import ChatMessageRow


@pytest.fixture
async def db():
    """In-memory database for testing."""
    test_db = Database(url="sqlite+aiosqlite:///:memory:")
    await test_db.init_db()
    yield test_db
    await test_db.close()


class TestSaveChatMessage:
    async def test_save_user_message(self, db: Database):
        row_id = await db.save_chat_message(
            tenant_id="default",
            role="user",
            content="What's the portfolio state?",
        )
        assert isinstance(row_id, int)
        assert row_id > 0

    async def test_save_assistant_message_with_tools(self, db: Database):
        tool_calls = [{"name": "get_portfolio_state", "input": {}}]
        row_id = await db.save_chat_message(
            tenant_id="default",
            role="assistant",
            content="The portfolio currently holds 5 positions.",
            session_id="sess_abc123",
            tool_calls_json=json.dumps(tool_calls),
        )
        assert row_id > 0

    async def test_save_message_with_session_id(self, db: Database):
        await db.save_chat_message(
            tenant_id="default",
            role="user",
            content="Hello",
            session_id="sess_xyz",
        )
        rows = await db.get_chat_messages(tenant_id="default", days=1)
        assert len(rows) == 1
        assert rows[0].session_id == "sess_xyz"

    async def test_save_message_null_session(self, db: Database):
        await db.save_chat_message(tenant_id="default", role="user", content="Hi")
        rows = await db.get_chat_messages(tenant_id="default", days=1)
        assert rows[0].session_id is None


class TestGetChatMessages:
    async def test_returns_in_chronological_order(self, db: Database):
        await db.save_chat_message(tenant_id="default", role="user", content="First")
        await db.save_chat_message(tenant_id="default", role="assistant", content="Second")
        await db.save_chat_message(tenant_id="default", role="user", content="Third")

        rows = await db.get_chat_messages(tenant_id="default", days=1)
        assert len(rows) == 3
        assert rows[0].content == "First"
        assert rows[2].content == "Third"

    async def test_returns_correct_roles(self, db: Database):
        await db.save_chat_message(tenant_id="default", role="user", content="Q")
        await db.save_chat_message(tenant_id="default", role="assistant", content="A")

        rows = await db.get_chat_messages(tenant_id="default", days=1)
        assert rows[0].role == "user"
        assert rows[1].role == "assistant"

    async def test_respects_days_filter(self, db: Database):
        """Old messages outside the days window should not be returned."""
        async with db.session() as s:
            old = ChatMessageRow(
                tenant_id="default",
                role="user",
                content="Old message",
                created_at=datetime.utcnow() - timedelta(days=10),
            )
            s.add(old)
            await s.commit()

        # days=1 should exclude the 10-day-old message
        rows = await db.get_chat_messages(tenant_id="default", days=1)
        assert len(rows) == 0

        # days=30 should include it
        rows = await db.get_chat_messages(tenant_id="default", days=30)
        assert len(rows) == 1

    async def test_respects_limit(self, db: Database):
        for i in range(10):
            await db.save_chat_message(tenant_id="default", role="user", content=f"msg {i}")

        rows = await db.get_chat_messages(tenant_id="default", days=1, limit=5)
        assert len(rows) == 5

    async def test_tenant_isolation(self, db: Database):
        await db.ensure_tenant("t2")
        await db.save_chat_message(tenant_id="default", role="user", content="Default msg")
        await db.save_chat_message(tenant_id="t2", role="user", content="T2 msg")

        default_rows = await db.get_chat_messages(tenant_id="default", days=1)
        t2_rows = await db.get_chat_messages(tenant_id="t2", days=1)

        assert len(default_rows) == 1
        assert default_rows[0].content == "Default msg"
        assert len(t2_rows) == 1
        assert t2_rows[0].content == "T2 msg"

    async def test_tool_calls_json_round_trips(self, db: Database):
        tool_calls = [{"name": "get_portfolio_state", "input": {}}]
        await db.save_chat_message(
            tenant_id="default",
            role="assistant",
            content="Here is the portfolio.",
            tool_calls_json=json.dumps(tool_calls),
        )
        rows = await db.get_chat_messages(tenant_id="default", days=1)
        assert rows[0].tool_calls_json is not None
        parsed = json.loads(rows[0].tool_calls_json)
        assert parsed[0]["name"] == "get_portfolio_state"

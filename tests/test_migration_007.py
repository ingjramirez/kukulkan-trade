"""Tests for migration 007: agent loop support.

Verifies that the ToolCallLogRow model and TenantRow.use_agent_loop
column work correctly with the in-memory database.
"""

from datetime import date

import pytest

from src.storage.database import Database
from src.storage.models import TenantRow, ToolCallLogRow


@pytest.fixture
async def db():
    database = Database(url="sqlite+aiosqlite:///:memory:")
    await database.init_db()
    yield database
    await database.close()


class TestToolCallLogTable:
    async def test_table_exists_and_insert_works(self, db: Database) -> None:
        """ToolCallLogRow table is created and accepts inserts."""
        async with db.session() as s:
            s.add(
                ToolCallLogRow(
                    tenant_id="default",
                    session_date=date(2026, 2, 13),
                    session_label="Morning",
                    turn=1,
                    tool_name="get_market_context",
                    tool_input="{}",
                    tool_output_preview='{"regime":"BULL"}',
                    success=True,
                )
            )
            await s.commit()

        logs = await db.get_tool_call_logs(tenant_id="default")
        assert len(logs) == 1
        assert logs[0].tool_name == "get_market_context"
        assert logs[0].session_label == "Morning"
        assert logs[0].success is True

    async def test_error_field_nullable(self, db: Database) -> None:
        """Error field can be None (success) or a string (failure)."""
        async with db.session() as s:
            s.add(
                ToolCallLogRow(
                    tenant_id="default",
                    session_date=date(2026, 2, 13),
                    turn=1,
                    tool_name="get_positions",
                    success=False,
                    error="Database timeout",
                )
            )
            await s.commit()

        logs = await db.get_tool_call_logs()
        assert logs[0].error == "Database timeout"
        assert logs[0].success is False


class TestUseAgentLoopColumn:
    async def test_default_false(self, db: Database) -> None:
        """New tenants have use_agent_loop=False by default."""
        tenant = TenantRow(id="test-1", name="Test")
        await db.create_tenant(tenant)
        row = await db.get_tenant("test-1")
        assert row is not None
        assert row.use_agent_loop is False

    async def test_set_true(self, db: Database) -> None:
        """use_agent_loop can be set to True."""
        tenant = TenantRow(id="test-2", name="Test", use_agent_loop=True)
        await db.create_tenant(tenant)
        row = await db.get_tenant("test-2")
        assert row is not None
        assert row.use_agent_loop is True

    async def test_update_toggle(self, db: Database) -> None:
        """use_agent_loop can be toggled via update."""
        tenant = TenantRow(id="test-3", name="Test", use_agent_loop=False)
        await db.create_tenant(tenant)

        await db.update_tenant("test-3", {"use_agent_loop": True})
        row = await db.get_tenant("test-3")
        assert row.use_agent_loop is True

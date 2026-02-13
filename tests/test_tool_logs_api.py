"""Tests for GET /api/agent/tool-logs endpoint."""

from datetime import date

import pytest
from httpx import ASGITransport, AsyncClient

from src.api.deps import get_current_user, get_db
from src.api.main import app
from src.api.rate_limit import RateLimitMiddleware
from src.storage.database import Database
from src.storage.models import ToolCallLogRow


def _reset_rate_limiter() -> None:
    handler = app.middleware_stack
    while handler:
        if isinstance(handler, RateLimitMiddleware):
            handler.reset()
            return
        handler = getattr(handler, "app", None)


async def _bypass_user() -> dict[str, str | None]:
    return {"username": "test-user", "tenant_id": None}


@pytest.fixture
async def db():
    test_db = Database(url="sqlite+aiosqlite:///:memory:")
    await test_db.init_db()
    yield test_db
    await test_db.close()


@pytest.fixture
async def client(db):
    app.dependency_overrides[get_db] = lambda: db
    app.dependency_overrides[get_current_user] = _bypass_user
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        _reset_rate_limiter()
        yield c
    app.dependency_overrides.clear()


class TestToolLogsEndpoint:
    async def test_empty_returns_empty_list(self, client: AsyncClient) -> None:
        r = await client.get("/api/agent/tool-logs")
        assert r.status_code == 200
        assert r.json() == []

    async def test_returns_tool_logs(self, client: AsyncClient, db: Database) -> None:
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
            s.add(
                ToolCallLogRow(
                    tenant_id="default",
                    session_date=date(2026, 2, 13),
                    session_label="Morning",
                    turn=2,
                    tool_name="propose_trades",
                    tool_input='{"trades":[]}',
                    tool_output_preview='{"status":"ok"}',
                    success=True,
                )
            )
            await s.commit()

        r = await client.get("/api/agent/tool-logs")
        assert r.status_code == 200
        data = r.json()
        assert len(data) == 2
        assert data[0]["tool_name"] in ("get_market_context", "propose_trades")

    async def test_filter_by_date(self, client: AsyncClient, db: Database) -> None:
        async with db.session() as s:
            s.add(
                ToolCallLogRow(
                    tenant_id="default",
                    session_date=date(2026, 2, 12),
                    turn=1,
                    tool_name="tool_a",
                    success=True,
                )
            )
            s.add(
                ToolCallLogRow(
                    tenant_id="default",
                    session_date=date(2026, 2, 13),
                    turn=1,
                    tool_name="tool_b",
                    success=True,
                )
            )
            await s.commit()

        r = await client.get("/api/agent/tool-logs?session_date=2026-02-13")
        assert r.status_code == 200
        data = r.json()
        assert len(data) == 1
        assert data[0]["tool_name"] == "tool_b"

    async def test_limit_param(self, client: AsyncClient, db: Database) -> None:
        async with db.session() as s:
            for i in range(5):
                s.add(
                    ToolCallLogRow(
                        tenant_id="default",
                        session_date=date(2026, 2, 13),
                        turn=i + 1,
                        tool_name=f"tool_{i}",
                        success=True,
                    )
                )
            await s.commit()

        r = await client.get("/api/agent/tool-logs?limit=2")
        assert r.status_code == 200
        data = r.json()
        assert len(data) == 2

"""Tests for GET /api/agent/conversations endpoints."""

import pytest
from httpx import ASGITransport, AsyncClient

from src.agent.conversation_store import ConversationStore
from src.api.deps import get_current_user, get_db
from src.api.main import app
from src.api.rate_limit import RateLimitMiddleware
from src.storage.database import Database


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
    app.state.db = db
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        _reset_rate_limiter()
        yield c
    app.dependency_overrides.clear()
    app.state.db = None


class TestConversationsListEndpoint:
    async def test_empty_returns_empty_list(self, client: AsyncClient) -> None:
        r = await client.get("/api/agent/conversations")
        assert r.status_code == 200
        assert r.json() == []

    async def test_returns_sessions(self, client: AsyncClient, db: Database) -> None:
        store = ConversationStore(db)
        await store.save_session(
            tenant_id="default",
            session_id="sess-001",
            trigger_type="morning",
            messages=[{"role": "user", "content": "Hello"}],
            token_count=1000,
            cost_usd=0.05,
        )

        r = await client.get("/api/agent/conversations")
        assert r.status_code == 200
        data = r.json()
        assert len(data) == 1
        assert data[0]["session_id"] == "sess-001"
        assert data[0]["trigger_type"] == "morning"
        assert data[0]["token_count"] == 1000
        assert data[0]["session_status"] == "completed"
        # No messages in list response
        assert "messages" not in data[0]

    async def test_respects_limit(self, client: AsyncClient, db: Database) -> None:
        store = ConversationStore(db)
        for i in range(5):
            await store.save_session(
                tenant_id="default",
                session_id=f"sess-{i:03d}",
                trigger_type="morning",
                messages=[{"role": "user", "content": f"Session {i}"}],
                token_count=1000,
                cost_usd=0.05,
            )

        r = await client.get("/api/agent/conversations?limit=3")
        assert r.status_code == 200
        assert len(r.json()) == 3


class TestConversationDetailEndpoint:
    async def test_returns_full_session(self, client: AsyncClient, db: Database) -> None:
        store = ConversationStore(db)
        await store.save_session(
            tenant_id="default",
            session_id="sess-detail",
            trigger_type="midday",
            messages=[
                {"role": "user", "content": "Midday check."},
                {"role": "assistant", "content": "All clear."},
            ],
            token_count=2000,
            cost_usd=0.10,
        )

        r = await client.get("/api/agent/conversations/sess-detail")
        assert r.status_code == 200
        data = r.json()
        assert data["session_id"] == "sess-detail"
        assert len(data["messages"]) == 2
        assert data["messages"][0]["content"] == "Midday check."

    async def test_nonexistent_returns_not_found(self, client: AsyncClient) -> None:
        r = await client.get("/api/agent/conversations/nonexistent")
        assert r.status_code == 200  # Returns {"detail": "Session not found"}
        assert r.json()["detail"] == "Session not found"

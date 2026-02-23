"""Tests for the Agent Chat API endpoints."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest
from httpx import ASGITransport, AsyncClient

from src.agent.claude_invoker import ChatResult, ClaudeInvoker
from src.api.deps import get_authorized_tenant_id, get_current_user, get_db, get_invoker
from src.api.main import app
from src.storage.database import Database

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
async def db():
    test_db = Database(url="sqlite+aiosqlite:///:memory:")
    await test_db.init_db()
    yield test_db
    await test_db.close()


async def _bypass_user() -> dict[str, str | None]:
    return {"username": "test-user", "tenant_id": None}


async def _bypass_tenant_id() -> str:
    return "default"


@pytest.fixture
def mock_invoker(tmp_path):
    """ClaudeInvoker with mocked chat methods."""
    inv = ClaudeInvoker(workspace=tmp_path, timeout=10, tenant_id="default")
    return inv


@pytest.fixture
async def client(db, mock_invoker):
    app.dependency_overrides[get_db] = lambda: db
    app.dependency_overrides[get_current_user] = _bypass_user
    app.dependency_overrides[get_authorized_tenant_id] = _bypass_tenant_id
    app.dependency_overrides[get_invoker] = lambda: mock_invoker
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c
    app.dependency_overrides.clear()


# ── POST /api/chat ────────────────────────────────────────────────────────────


class TestPostChat:
    async def test_returns_200_with_content(self, client, mock_invoker):
        mock_invoker.chat = AsyncMock(
            return_value=ChatResult(
                content="The portfolio has 5 positions.",
                session_id="sess_1",
                tool_calls=[{"name": "get_portfolio_state", "input": {}}],
                num_turns=3,
                duration_ms=900,
            )
        )
        resp = await client.post("/api/chat", json={"message": "What positions do I hold?"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["content"] == "The portfolio has 5 positions."
        assert data["session_id"] == "sess_1"
        assert data["num_turns"] == 3
        assert len(data["tool_calls"]) == 1

    async def test_saves_user_and_assistant_messages(self, client, db, mock_invoker):
        mock_invoker.chat = AsyncMock(
            return_value=ChatResult(content="All good.", session_id="s1", num_turns=2, duration_ms=500)
        )
        await client.post("/api/chat", json={"message": "How are things?"})

        rows = await db.get_chat_messages(tenant_id="default", days=1)
        roles = [r.role for r in rows]
        assert "user" in roles
        assert "assistant" in roles
        contents = {r.role: r.content for r in rows}
        assert contents["user"] == "How are things?"
        assert contents["assistant"] == "All good."

    async def test_returns_503_on_error(self, client, mock_invoker):
        mock_invoker.chat = AsyncMock(return_value=ChatResult(error="Agent failed"))
        resp = await client.post("/api/chat", json={"message": "Hello"})
        assert resp.status_code == 503
        assert "Agent failed" in resp.json()["detail"]

    async def test_rejects_empty_message(self, client):
        resp = await client.post("/api/chat", json={"message": ""})
        assert resp.status_code == 422

    async def test_rejects_missing_message(self, client):
        resp = await client.post("/api/chat", json={})
        assert resp.status_code == 422

    async def test_requires_auth(self, db):
        """Without auth override, unauthenticated request should fail."""
        app.dependency_overrides[get_db] = lambda: db
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.post("/api/chat", json={"message": "Hello"})
        app.dependency_overrides.clear()
        assert resp.status_code in (401, 403)


# ── GET /api/chat/history ─────────────────────────────────────────────────────


class TestGetChatHistory:
    async def test_returns_empty_history(self, client):
        resp = await client.get("/api/chat/history")
        assert resp.status_code == 200
        assert resp.json()["messages"] == []

    async def test_returns_messages_in_order(self, client, db):
        await db.save_chat_message(tenant_id="default", role="user", content="Hello")
        await db.save_chat_message(tenant_id="default", role="assistant", content="Hi there!")

        resp = await client.get("/api/chat/history")
        assert resp.status_code == 200
        msgs = resp.json()["messages"]
        assert len(msgs) == 2
        assert msgs[0]["role"] == "user"
        assert msgs[0]["content"] == "Hello"
        assert msgs[1]["role"] == "assistant"

    async def test_message_includes_tool_calls(self, client, db):
        tool_calls = [{"name": "get_portfolio_state", "input": {}}]
        await db.save_chat_message(
            tenant_id="default",
            role="assistant",
            content="Here's the portfolio.",
            tool_calls_json=json.dumps(tool_calls),
        )
        resp = await client.get("/api/chat/history")
        msgs = resp.json()["messages"]
        assert len(msgs[0]["tool_calls"]) == 1
        assert msgs[0]["tool_calls"][0]["name"] == "get_portfolio_state"

    async def test_respects_days_param(self, client, db):
        # Only one recent message
        await db.save_chat_message(tenant_id="default", role="user", content="Recent")

        resp = await client.get("/api/chat/history?days=1")
        assert resp.status_code == 200
        assert len(resp.json()["messages"]) == 1

    async def test_days_param_clamped_to_30(self, client):
        resp = await client.get("/api/chat/history?days=100")
        assert resp.status_code == 422

    async def test_requires_auth(self, db):
        app.dependency_overrides[get_db] = lambda: db
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.get("/api/chat/history")
        app.dependency_overrides.clear()
        assert resp.status_code in (401, 403)


# ── POST /api/chat/stream ─────────────────────────────────────────────────────


class TestChatStream:
    async def test_returns_sse_content_type(self, client, mock_invoker):
        async def _fake_stream(message, today=None):
            yield {"type": "text", "text": "Hello"}
            yield {"type": "done", "session_id": "s1", "num_turns": 1, "duration_ms": 100}

        mock_invoker.chat_stream = _fake_stream

        resp = await client.post("/api/chat/stream", json={"message": "Hi"})
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers.get("content-type", "")

    async def test_stream_saves_user_message(self, client, db, mock_invoker):
        async def _fake_stream(message, today=None):
            yield {"type": "text", "text": "Response"}
            yield {"type": "done", "session_id": None, "num_turns": 1, "duration_ms": 0}

        mock_invoker.chat_stream = _fake_stream

        await client.post("/api/chat/stream", json={"message": "Stream test"})
        rows = await db.get_chat_messages(tenant_id="default", days=1)
        user_msgs = [r for r in rows if r.role == "user"]
        assert len(user_msgs) == 1
        assert user_msgs[0].content == "Stream test"

    async def test_stream_saves_assistant_message(self, client, db, mock_invoker):
        async def _fake_stream(message, today=None):
            yield {"type": "text", "text": "This is the answer."}
            yield {"type": "done", "session_id": "s2", "num_turns": 2, "duration_ms": 500}

        mock_invoker.chat_stream = _fake_stream

        await client.post("/api/chat/stream", json={"message": "Question"})
        rows = await db.get_chat_messages(tenant_id="default", days=1)
        asst_msgs = [r for r in rows if r.role == "assistant"]
        assert len(asst_msgs) == 1
        assert asst_msgs[0].content == "This is the answer."

    async def test_stream_events_are_valid_ndjson(self, client, mock_invoker):
        async def _fake_stream(message, today=None):
            yield {"type": "text", "text": "Part 1"}
            yield {"type": "tool_use", "id": "t1", "name": "get_portfolio_state", "input": {}}
            yield {"type": "done", "session_id": "s1", "num_turns": 1, "duration_ms": 200}

        mock_invoker.chat_stream = _fake_stream

        resp = await client.post("/api/chat/stream", json={"message": "Hi"})
        # Parse SSE lines
        events = []
        for line in resp.text.splitlines():
            line = line.strip()
            if line.startswith("data:"):
                events.append(json.loads(line[len("data:"):].strip()))

        types = [e["type"] for e in events]
        assert "text" in types
        assert "tool_use" in types
        assert "done" in types

"""Tests for SSE /connections admin endpoint."""

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from src.api.auth import create_access_token
from src.api.main import app
from src.api.rate_limit import RateLimitMiddleware
from src.events.event_bus import event_bus
from src.storage.database import Database


def _reset_rate_limiter() -> None:
    handler = app.middleware_stack
    while handler:
        if isinstance(handler, RateLimitMiddleware):
            handler.reset()
            return
        handler = getattr(handler, "app", None)


@pytest.fixture(autouse=True)
def _clean_bus():
    event_bus.clear()
    yield
    event_bus.clear()


@pytest.fixture
async def db():
    database = Database(url="sqlite+aiosqlite:///:memory:")
    await database.init_db()
    yield database
    await database.close()


@pytest.fixture
async def client(db):
    _reset_rate_limiter()
    app.state.db = db
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c
    app.state.db = None


@pytest.fixture
def admin_headers():
    token = create_access_token("admin")
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def tenant_headers():
    token = create_access_token("user1", tenant_id="t1")
    return {"Authorization": f"Bearer {token}"}


async def test_connections_empty(client: AsyncClient, admin_headers: dict):
    resp = await client.get("/api/events/connections", headers=admin_headers)
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 0
    assert data["connections"] == []


async def test_connections_shows_subscribers(client: AsyncClient, admin_headers: dict):
    sub_id, _ = event_bus.subscribe(tenant_id="t1")
    resp = await client.get("/api/events/connections", headers=admin_headers)
    data = resp.json()
    assert data["total"] == 1
    assert data["connections"][0]["tenant_id"] == "t1"
    event_bus.unsubscribe(sub_id)


async def test_connections_requires_admin(client: AsyncClient, tenant_headers: dict):
    resp = await client.get("/api/events/connections", headers=tenant_headers)
    assert resp.status_code == 403

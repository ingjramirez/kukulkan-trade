"""Tests for event history buffer and /recent endpoint."""

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from src.api.auth import create_access_token
from src.api.main import app
from src.api.rate_limit import RateLimitMiddleware
from src.events.event_bus import Event, EventType, event_bus
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


# --- /recent endpoint ---


async def test_recent_empty(client: AsyncClient, admin_headers: dict):
    resp = await client.get("/api/events/recent", headers=admin_headers)
    assert resp.status_code == 200
    assert resp.json() == []


async def test_recent_returns_events(client: AsyncClient, admin_headers: dict):
    for i in range(3):
        event_bus.publish(Event(type=EventType.TRADE_EXECUTED, tenant_id="default", data={"i": i}))

    resp = await client.get("/api/events/recent", headers=admin_headers)
    data = resp.json()
    assert len(data) == 3
    assert data[0]["data"]["i"] == 0
    assert data[2]["data"]["i"] == 2


async def test_recent_limit_param(client: AsyncClient, admin_headers: dict):
    for i in range(10):
        event_bus.publish(Event(type=EventType.TRADE_EXECUTED, tenant_id="default", data={"i": i}))

    resp = await client.get("/api/events/recent?limit=3", headers=admin_headers)
    data = resp.json()
    assert len(data) == 3
    # Should be the 3 most recent
    assert data[0]["data"]["i"] == 7


async def test_recent_tenant_scoping(client: AsyncClient, tenant_headers: dict):
    """Tenant user only sees their own events in /recent."""
    event_bus.publish(Event(type=EventType.TRADE_EXECUTED, tenant_id="t1", data={"ticker": "AAPL"}))
    event_bus.publish(Event(type=EventType.TRADE_EXECUTED, tenant_id="t2", data={"ticker": "MSFT"}))

    resp = await client.get("/api/events/recent", headers=tenant_headers)
    data = resp.json()
    assert len(data) == 1
    assert data[0]["data"]["ticker"] == "AAPL"


async def test_recent_requires_auth(client: AsyncClient):
    resp = await client.get("/api/events/recent")
    assert resp.status_code in (401, 403)

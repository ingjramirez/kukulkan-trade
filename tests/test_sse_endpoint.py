"""Tests for SSE endpoint routes (/api/events/*)."""

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from src.api.auth import create_access_token
from src.api.main import app
from src.api.rate_limit import RateLimitMiddleware
from src.api.routes.events import _sse_generator
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
def _clean_event_bus():
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


# --- SSE generator (unit tests) ---


async def test_generator_delivers_event():
    """Generator yields SSE-formatted events from the queue."""
    sub_id, queue = event_bus.subscribe(tenant_id="t1")
    event_bus.publish(Event(type=EventType.TRADE_EXECUTED, tenant_id="t1", data={"ticker": "AAPL"}))

    gen = _sse_generator(sub_id, queue, heartbeat_s=5.0)
    chunk = await gen.__anext__()
    assert "event: trade_executed" in chunk
    assert "AAPL" in chunk
    await gen.aclose()


async def test_generator_heartbeat_on_timeout():
    """Generator emits heartbeat when no events arrive within timeout."""
    sub_id, queue = event_bus.subscribe(tenant_id="t1")
    gen = _sse_generator(sub_id, queue, heartbeat_s=0.1)
    chunk = await gen.__anext__()
    assert "event: heartbeat" in chunk
    await gen.aclose()


async def test_generator_unsubscribes_on_close():
    """Generator unsubscribes when closed after yielding."""
    sub_id, queue = event_bus.subscribe(tenant_id="t1")
    assert event_bus.subscriber_count == 1
    gen = _sse_generator(sub_id, queue, heartbeat_s=0.1)
    # Must enter the generator (get a heartbeat) before closing
    await gen.__anext__()
    await gen.aclose()
    assert event_bus.subscriber_count == 0


async def test_generator_tenant_scoping():
    """Generator only yields events for subscribed tenant."""
    sub_id, queue = event_bus.subscribe(tenant_id="t1")
    # Publish for t2 (filtered by bus) then t1
    event_bus.publish(Event(type=EventType.TRADE_EXECUTED, tenant_id="t2", data={"ticker": "MSFT"}))
    event_bus.publish(Event(type=EventType.TRADE_EXECUTED, tenant_id="t1", data={"ticker": "AAPL"}))

    gen = _sse_generator(sub_id, queue, heartbeat_s=5.0)
    chunk = await gen.__anext__()
    assert "AAPL" in chunk
    assert "MSFT" not in chunk
    await gen.aclose()


# --- /stream auth ---


async def test_stream_requires_auth(client: AsyncClient):
    resp = await client.get("/api/events/stream")
    assert resp.status_code in (401, 403)


# --- /recent ---


async def test_recent_returns_events(client: AsyncClient, admin_headers: dict):
    event_bus.publish(Event(type=EventType.TRADE_EXECUTED, tenant_id="default", data={"ticker": "GOOG"}))
    resp = await client.get("/api/events/recent", headers=admin_headers)
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 1
    assert data[0]["type"] == "trade_executed"
    assert data[0]["data"]["ticker"] == "GOOG"


async def test_recent_requires_auth(client: AsyncClient):
    resp = await client.get("/api/events/recent")
    assert resp.status_code in (401, 403)

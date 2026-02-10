"""Tests for the universe API endpoint."""

import pytest
from httpx import ASGITransport, AsyncClient

from config.universe import PORTFOLIO_B_UNIVERSE
from src.api.deps import get_current_user, get_db
from src.api.main import app
from src.api.rate_limit import RateLimitMiddleware
from src.storage.database import Database


def _reset_rate_limiter() -> None:
    """Find and reset the rate limiter middleware on the app."""
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
    """Authenticated client for universe endpoint."""
    _reset_rate_limiter()
    app.dependency_overrides[get_db] = lambda: db
    app.dependency_overrides[get_current_user] = _bypass_user
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c
    app.dependency_overrides.clear()


@pytest.fixture
async def unauth_client(db):
    """Unauthenticated client for testing auth requirement."""
    _reset_rate_limiter()
    app.dependency_overrides[get_db] = lambda: db
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c
    app.dependency_overrides.clear()


async def test_get_base_universe(client):
    resp = await client.get("/api/universe/base")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == len(PORTFOLIO_B_UNIVERSE)
    assert "sectors" in data
    assert isinstance(data["sectors"], dict)
    # Every ticker in PORTFOLIO_B_UNIVERSE appears exactly once
    all_tickers = [t for tickers in data["sectors"].values() for t in tickers]
    assert len(all_tickers) == data["total"]
    assert len(set(all_tickers)) == data["total"]  # no duplicates


async def test_base_universe_sectors_sorted(client):
    resp = await client.get("/api/universe/base")
    data = resp.json()
    sector_names = list(data["sectors"].keys())
    assert sector_names == sorted(sector_names)
    # Tickers within each sector are sorted
    for tickers in data["sectors"].values():
        assert tickers == sorted(tickers)


async def test_base_universe_requires_auth(unauth_client):
    """Unauthenticated request should get 401/403."""
    resp = await unauth_client.get("/api/universe/base")
    assert resp.status_code in (401, 403)

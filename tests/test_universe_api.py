"""Tests for the universe API endpoint."""

import pytest
from httpx import ASGITransport, AsyncClient

from config.universe import PORTFOLIO_B_UNIVERSE
from src.api.deps import get_db
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


@pytest.fixture
async def db():
    test_db = Database(url="sqlite+aiosqlite:///:memory:")
    await test_db.init_db()
    yield test_db
    await test_db.close()


@pytest.fixture
async def client(db):
    """Unauthenticated client — universe endpoint is public."""
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

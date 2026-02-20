"""Tests for GET /api/signals/rankings endpoint."""

import json
from datetime import datetime, timezone

import pytest
from httpx import ASGITransport, AsyncClient

from src.api.deps import get_current_user, get_db
from src.api.main import app
from src.storage.database import Database
from src.storage.models import TickerSignalRow

# ── Fixtures ─────────────────────────────────────────────────────────────────


def _bypass_user():
    return {"sub": "test-user", "tenant_id": "default", "role": "user"}


@pytest.fixture
async def db():
    test_db = Database(url="sqlite+aiosqlite:///:memory:")
    await test_db.init_db()
    yield test_db
    await test_db.close()


@pytest.fixture
async def seeded_db(db):
    """DB with signal data."""
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    rows = [
        TickerSignalRow(
            tenant_id="default",
            ticker="XLK",
            composite_score=85.3,
            rank=1,
            prev_rank=3,
            rank_velocity=2.0,
            momentum_20d=0.08,
            momentum_63d=0.15,
            rsi=55,
            macd_histogram=0.5,
            sma_trend_score=3,
            bollinger_pct_b=0.7,
            volume_ratio=1.2,
            alerts=json.dumps(["golden_cross"]),
            scored_at=now,
        ),
        TickerSignalRow(
            tenant_id="default",
            ticker="XLF",
            composite_score=60.0,
            rank=2,
            prev_rank=2,
            rank_velocity=0,
            momentum_20d=0.03,
            momentum_63d=0.05,
            rsi=45,
            macd_histogram=0.1,
            sma_trend_score=2,
            bollinger_pct_b=0.5,
            volume_ratio=1.0,
            alerts=json.dumps([]),
            scored_at=now,
        ),
    ]
    async with db.session() as s:
        s.add_all(rows)
        await s.commit()
    return db


@pytest.fixture
async def client(seeded_db):
    app.dependency_overrides[get_db] = lambda: seeded_db
    app.dependency_overrides[get_current_user] = _bypass_user
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c
    app.dependency_overrides.clear()
    app.state.db = None


@pytest.fixture
async def empty_client(db):
    app.dependency_overrides[get_db] = lambda: db
    app.dependency_overrides[get_current_user] = _bypass_user
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c
    app.dependency_overrides.clear()
    app.state.db = None


# ── Tests ────────────────────────────────────────────────────────────────────


async def test_get_signal_rankings(client):
    """Returns signal rankings with correct structure."""
    resp = await client.get("/api/signals/rankings")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 2
    assert data["scored_at"] is not None
    assert len(data["signals"]) == 2

    first = data["signals"][0]
    assert first["ticker"] == "XLK"
    assert first["composite_score"] == 85.3
    assert first["rank"] == 1
    assert first["prev_rank"] == 3
    assert first["rank_velocity"] == 2.0
    assert first["alerts"] == ["golden_cross"]


async def test_get_signal_rankings_empty(empty_client):
    """Returns empty response when no signals exist."""
    resp = await empty_client.get("/api/signals/rankings")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 0
    assert data["signals"] == []
    assert data["scored_at"] is None


async def test_signal_rankings_requires_auth(seeded_db):
    """Endpoint requires authentication."""
    app.dependency_overrides[get_db] = lambda: seeded_db
    # No auth override
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        resp = await c.get("/api/signals/rankings")
    app.dependency_overrides.clear()
    app.state.db = None
    assert resp.status_code == 401


async def test_signal_response_fields(client):
    """All indicator fields are present in the response."""
    resp = await client.get("/api/signals/rankings")
    first = resp.json()["signals"][0]
    for field in [
        "momentum_20d",
        "momentum_63d",
        "rsi",
        "macd_histogram",
        "sma_trend_score",
        "bollinger_pct_b",
        "volume_ratio",
    ]:
        assert field in first

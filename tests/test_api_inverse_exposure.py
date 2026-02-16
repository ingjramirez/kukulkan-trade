"""Tests for GET /api/agent/inverse-exposure endpoint."""

import pytest
from httpx import ASGITransport, AsyncClient

from src.api.deps import get_current_user, get_db
from src.api.main import app
from src.storage.database import Database
from src.storage.models import Base

try:
    from src.api.main import _reset_rate_limiter
except ImportError:
    _reset_rate_limiter = None


@pytest.fixture
async def db():
    from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
    from sqlalchemy.orm import sessionmaker

    engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    session_factory = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    db_instance = Database.__new__(Database)
    db_instance._engine = engine
    db_instance._session_factory = session_factory
    db_instance._url = "sqlite+aiosqlite:///:memory:"
    yield db_instance
    await engine.dispose()


@pytest.fixture
def _bypass_user():
    return {"sub": "admin", "role": "admin"}


@pytest.fixture
async def client(db, _bypass_user):
    app.dependency_overrides[get_db] = lambda: db
    app.dependency_overrides[get_current_user] = lambda: _bypass_user
    if _reset_rate_limiter:
        _reset_rate_limiter()
    app.state.db = None
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c
    app.dependency_overrides.clear()
    app.state.db = None


async def test_inverse_exposure_empty(client: AsyncClient, db: Database) -> None:
    """No inverse positions returns zeros."""
    await db.upsert_portfolio("B", cash=100_000.0, total_value=100_000.0)
    resp = await client.get("/api/agent/inverse-exposure")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_value"] == 0.0
    assert data["total_pct"] == 0.0
    assert data["positions"] == []
    assert "rules" in data


async def test_inverse_exposure_with_positions(client: AsyncClient, db: Database) -> None:
    """Inverse positions appear in response."""
    await db.upsert_portfolio("B", cash=80_000.0, total_value=100_000.0)
    await db.upsert_position("B", "SH", shares=200, avg_price=15.0)  # $3,000
    await db.upsert_position("B", "XLK", shares=50, avg_price=200.0)  # $10,000

    resp = await client.get("/api/agent/inverse-exposure")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_value"] == 3000.0
    assert data["total_pct"] == 3.0
    assert len(data["positions"]) == 1
    assert data["positions"][0]["ticker"] == "SH"
    assert data["positions"][0]["equity_hedge"] is True


async def test_inverse_exposure_rules_present(client: AsyncClient, db: Database) -> None:
    """Rules dict includes limits."""
    await db.upsert_portfolio("B", cash=100_000.0, total_value=100_000.0)
    resp = await client.get("/api/agent/inverse-exposure")
    data = resp.json()
    rules = data["rules"]
    assert rules["max_single_pct"] == 10.0
    assert rules["max_total_pct"] == 15.0
    assert rules["max_positions"] == 2


async def test_inverse_exposure_requires_auth() -> None:
    """Endpoint requires authentication."""
    app.dependency_overrides.clear()
    if _reset_rate_limiter:
        _reset_rate_limiter()
    app.state.db = None
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        resp = await c.get("/api/agent/inverse-exposure")
    assert resp.status_code == 401
    app.state.db = None


async def test_inverse_exposure_hold_alert(client: AsyncClient, db: Database) -> None:
    """Positions with BUY trades get days_held computed."""
    await db.upsert_portfolio("B", cash=90_000.0, total_value=100_000.0)
    await db.upsert_position("B", "SH", shares=200, avg_price=15.0)
    await db.log_trade("B", "SH", "BUY", 200, 15.0, reason="hedge")

    resp = await client.get("/api/agent/inverse-exposure")
    data = resp.json()
    pos = data["positions"][0]
    assert pos["ticker"] == "SH"
    assert pos["days_held"] is not None
    assert pos["days_held"] >= 0

"""Tests for GET /api/agent/budget endpoint."""

from datetime import date

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
    app.state.db = None  # Prevent stale DB usage in login()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c
    app.dependency_overrides.clear()
    app.state.db = None


@pytest.mark.asyncio
async def test_budget_endpoint_returns_defaults(client, db):
    """Budget endpoint returns zero spend with default limits."""
    resp = await client.get("/api/agent/budget")
    assert resp.status_code == 200
    data = resp.json()
    assert data["daily_spent"] == 0.0
    assert data["monthly_spent"] == 0.0
    assert data["daily_exhausted"] is False
    assert data["monthly_exhausted"] is False
    assert "daily_limit" in data
    assert "monthly_limit" in data
    assert "daily_remaining" in data
    assert "monthly_remaining" in data
    assert "haiku_only" in data


@pytest.mark.asyncio
async def test_budget_endpoint_with_spend(client, db):
    """Budget endpoint reflects recorded spend."""
    today = date.today()
    await db.save_budget_log(
        tenant_id="default",
        session_date=today,
        session_label="morning",
        session_id="test-session",
        input_tokens=1000,
        output_tokens=500,
        cache_read_tokens=0,
        cache_creation_tokens=0,
        cost_usd=0.50,
    )
    resp = await client.get("/api/agent/budget")
    assert resp.status_code == 200
    data = resp.json()
    assert data["daily_spent"] == 0.50
    assert data["monthly_spent"] == 0.50

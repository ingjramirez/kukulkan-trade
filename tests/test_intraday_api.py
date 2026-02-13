"""Tests for intraday API endpoints."""

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

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


async def _bypass_admin() -> dict[str, str | None]:
    return {"username": "admin", "tenant_id": None}


async def _bypass_tenant_user() -> dict[str, str | None]:
    return {"username": "tenant-user", "tenant_id": "t-1"}


@pytest.fixture
async def db():
    test_db = Database(url="sqlite+aiosqlite:///:memory:")
    await test_db.init_db()
    yield test_db
    await test_db.close()


@pytest.fixture
async def admin_client(db):
    _reset_rate_limiter()
    app.dependency_overrides[get_db] = lambda: db
    app.dependency_overrides[get_current_user] = _bypass_admin
    app.state.db = db
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c
    app.dependency_overrides.clear()
    app.state.db = None


@pytest.fixture
async def tenant_client(db):
    _reset_rate_limiter()
    app.dependency_overrides[get_db] = lambda: db
    app.dependency_overrides[get_current_user] = _bypass_tenant_user
    app.state.db = db
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c
    app.dependency_overrides.clear()
    app.state.db = None


# ── GET /api/snapshots/intraday ────────────────────────────────────


async def test_intraday_snapshots_empty(admin_client, db):
    resp = await admin_client.get("/api/snapshots/intraday")
    assert resp.status_code == 200
    assert resp.json() == []


async def test_intraday_snapshots_returns_data(admin_client, db):
    now = datetime.now(timezone.utc)
    await db.save_intraday_snapshot(
        tenant_id="default", portfolio="A", timestamp=now,
        total_value=35000.0, cash=30000.0, positions_value=5000.0,
    )
    resp = await admin_client.get("/api/snapshots/intraday")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 1
    assert data[0]["portfolio"] == "A"
    assert data[0]["total_value"] == 35000.0


async def test_intraday_snapshots_filter_portfolio(admin_client, db):
    now = datetime.now(timezone.utc)
    await db.save_intraday_snapshot(
        tenant_id="default", portfolio="A", timestamp=now,
        total_value=35000.0, cash=30000.0, positions_value=5000.0,
    )
    await db.save_intraday_snapshot(
        tenant_id="default", portfolio="B", timestamp=now,
        total_value=68000.0, cash=60000.0, positions_value=8000.0,
    )
    resp = await admin_client.get("/api/snapshots/intraday?portfolio=B")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 1
    assert data[0]["portfolio"] == "B"


async def test_intraday_snapshots_period_filter(admin_client, db):
    now = datetime.now(timezone.utc)
    old_ts = now - timedelta(days=5)
    recent_ts = now - timedelta(hours=2)

    await db.save_intraday_snapshot(
        tenant_id="default", portfolio="A", timestamp=old_ts,
        total_value=34000.0, cash=30000.0, positions_value=4000.0,
    )
    await db.save_intraday_snapshot(
        tenant_id="default", portfolio="A", timestamp=recent_ts,
        total_value=35000.0, cash=30000.0, positions_value=5000.0,
    )

    # period=1d should only return recent
    resp = await admin_client.get("/api/snapshots/intraday?period=1d")
    assert resp.status_code == 200
    assert len(resp.json()) == 1

    # period=1w should return both
    resp = await admin_client.get("/api/snapshots/intraday?period=1w")
    assert resp.status_code == 200
    assert len(resp.json()) == 2


async def test_intraday_snapshots_tenant_isolation(tenant_client, db):
    now = datetime.now(timezone.utc)
    await db.save_intraday_snapshot(
        tenant_id="t-1", portfolio="B", timestamp=now,
        total_value=68000.0, cash=60000.0, positions_value=8000.0,
    )
    await db.save_intraday_snapshot(
        tenant_id="t-2", portfolio="B", timestamp=now,
        total_value=50000.0, cash=45000.0, positions_value=5000.0,
    )
    resp = await tenant_client.get("/api/snapshots/intraday")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 1
    assert data[0]["total_value"] == 68000.0


async def test_intraday_snapshots_invalid_period(admin_client):
    resp = await admin_client.get("/api/snapshots/intraday?period=2y")
    assert resp.status_code == 422


# ── GET /api/account/history ───────────────────────────────────────


async def test_account_history_success(admin_client):
    mock_data = {
        "timestamps": [1739440200, 1739440500],
        "equity": [99726.36, 99730.12],
        "profit_loss": [-273.64, -269.88],
        "profit_loss_pct": [-0.27, -0.27],
        "base_value": 100000.0,
        "timeframe": "5Min",
    }
    with patch(
        "src.api.routes.account.get_portfolio_history",
        new_callable=AsyncMock,
        return_value=mock_data,
    ):
        resp = await admin_client.get("/api/account/history?period=1D&timeframe=5Min")

    assert resp.status_code == 200
    data = resp.json()
    assert data["timestamps"] == [1739440200, 1739440500]
    assert data["equity"] == [99726.36, 99730.12]
    assert data["base_value"] == 100000.0
    assert data["timeframe"] == "5Min"


async def test_account_history_unavailable(admin_client):
    with patch(
        "src.api.routes.account.get_portfolio_history",
        new_callable=AsyncMock,
        return_value=None,
    ):
        resp = await admin_client.get("/api/account/history")

    assert resp.status_code == 503


async def test_account_history_invalid_period(admin_client):
    resp = await admin_client.get("/api/account/history?period=2Y")
    assert resp.status_code == 422


async def test_account_history_invalid_timeframe(admin_client):
    resp = await admin_client.get("/api/account/history?timeframe=30Min")
    assert resp.status_code == 422


async def test_account_history_default_params(admin_client):
    mock_data = {
        "timestamps": [1739440200],
        "equity": [99726.36],
        "profit_loss": [-273.64],
        "profit_loss_pct": [-0.27],
        "base_value": 100000.0,
        "timeframe": "5Min",
    }
    with patch(
        "src.api.routes.account.get_portfolio_history",
        new_callable=AsyncMock,
        return_value=mock_data,
    ) as mock_fn:
        resp = await admin_client.get("/api/account/history")

    assert resp.status_code == 200
    mock_fn.assert_awaited_once_with("1D", "5Min", False)


async def test_account_history_extended_hours(admin_client):
    mock_data = {
        "timestamps": [1739440200],
        "equity": [99726.36],
        "profit_loss": [-273.64],
        "profit_loss_pct": [-0.27],
        "base_value": 100000.0,
        "timeframe": "5Min",
    }
    with patch(
        "src.api.routes.account.get_portfolio_history",
        new_callable=AsyncMock,
        return_value=mock_data,
    ) as mock_fn:
        resp = await admin_client.get(
            "/api/account/history?extended_hours=true"
        )

    assert resp.status_code == 200
    mock_fn.assert_awaited_once_with("1D", "5Min", True)

"""Tests for improvement API endpoints: snapshots, changelog, trend."""

import json
from datetime import date, timedelta

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


async def _bypass_user() -> dict[str, str | None]:
    return {"username": "test-user", "tenant_id": None}


async def _bypass_user_tenant() -> dict[str, str | None]:
    return {"username": "test-user", "tenant_id": "t1"}


@pytest.fixture
async def db():
    test_db = Database(url="sqlite+aiosqlite:///:memory:")
    await test_db.init_db()
    yield test_db
    await test_db.close()


@pytest.fixture
async def client(db):
    app.dependency_overrides[get_db] = lambda: db
    app.dependency_overrides[get_current_user] = _bypass_user
    app.state.db = db
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        _reset_rate_limiter()
        yield c
    app.dependency_overrides.clear()
    app.state.db = None


# ── List Snapshots ────────────────────────────────────────────────


async def test_list_snapshots_empty(client: AsyncClient):
    resp = await client.get("/api/agent/improvements")
    assert resp.status_code == 200
    assert resp.json() == []


async def test_list_snapshots_with_data(client: AsyncClient, db: Database):
    await db.save_improvement_snapshot(
        tenant_id="default",
        week_start=date(2026, 2, 10),
        week_end=date(2026, 2, 17),
        total_trades=15,
        win_rate_pct=60.0,
        avg_pnl_pct=1.5,
        avg_alpha_vs_spy=0.8,
        total_cost_usd=2.50,
        strategy_mode="conservative",
        trailing_stop_multiplier=1.0,
        proposal_json=None,
        applied_changes=None,
        report_text=None,
    )

    resp = await client.get("/api/agent/improvements")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 1
    assert data[0]["total_trades"] == 15
    assert data[0]["win_rate_pct"] == 60.0


# ── Detail Snapshot ───────────────────────────────────────────────


async def test_get_snapshot_detail(client: AsyncClient, db: Database):
    snap_id = await db.save_improvement_snapshot(
        tenant_id="default",
        week_start=date(2026, 2, 10),
        week_end=date(2026, 2, 17),
        total_trades=10,
        win_rate_pct=55.0,
        avg_pnl_pct=1.2,
        avg_alpha_vs_spy=0.5,
        total_cost_usd=1.50,
        strategy_mode="conservative",
        trailing_stop_multiplier=1.0,
        proposal_json=json.dumps({"changes": [], "summary": "ok"}),
        applied_changes=json.dumps([{"parameter": "strategy_mode", "status": "applied"}]),
        report_text="Test report",
    )

    resp = await client.get(f"/api/agent/improvements/{snap_id}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["id"] == snap_id
    assert data["total_trades"] == 10
    assert data["proposal_json"]["summary"] == "ok"
    assert len(data["applied_changes"]) == 1
    assert data["report_text"] == "Test report"


async def test_get_snapshot_not_found(client: AsyncClient):
    resp = await client.get("/api/agent/improvements/999")
    assert resp.status_code == 404


# ── Changelog ─────────────────────────────────────────────────────


async def test_changelog_empty(client: AsyncClient):
    resp = await client.get("/api/agent/improvements/changelog")
    assert resp.status_code == 200
    assert resp.json() == []


async def test_changelog_with_data(client: AsyncClient, db: Database):
    await db.insert_parameter_changelog(
        tenant_id="default",
        parameter="strategy_mode",
        old_value="conservative",
        new_value="standard",
        reason="Higher win rate",
    )

    resp = await client.get("/api/agent/improvements/changelog")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 1
    assert data[0]["parameter"] == "strategy_mode"
    assert data[0]["old_value"] == "conservative"
    assert data[0]["new_value"] == "standard"


# ── Trend ─────────────────────────────────────────────────────────


async def test_trend_insufficient_data(client: AsyncClient):
    resp = await client.get("/api/agent/improvements/trend")
    assert resp.status_code == 200
    data = resp.json()
    assert data["classification"] == "insufficient_data"


async def test_trend_with_data(client: AsyncClient, db: Database):
    base = date(2026, 1, 6)
    for i, wr in enumerate([40.0, 50.0, 60.0, 70.0]):
        ws = base + timedelta(weeks=i)
        we = ws + timedelta(days=7)
        await db.save_improvement_snapshot(
            tenant_id="default",
            week_start=ws,
            week_end=we,
            total_trades=10,
            win_rate_pct=wr,
            avg_pnl_pct=wr / 10,
            avg_alpha_vs_spy=None,
            total_cost_usd=1.0,
            strategy_mode="conservative",
            trailing_stop_multiplier=1.0,
            proposal_json=None,
            applied_changes=None,
            report_text=None,
        )

    resp = await client.get("/api/agent/improvements/trend")
    assert resp.status_code == 200
    data = resp.json()
    assert data["classification"] == "improving"
    assert data["weeks_analyzed"] == 4
    assert len(data["data_points"]) == 4


# ── Tenant Isolation ─────────────────────────────────────────────


async def test_snapshot_tenant_isolation(db: Database):
    """Tenant user can only see their own snapshots."""
    await db.ensure_tenant("t1")
    await db.save_improvement_snapshot(
        tenant_id="default",
        week_start=date(2026, 2, 10),
        week_end=date(2026, 2, 17),
        total_trades=5,
        win_rate_pct=50.0,
        avg_pnl_pct=0.0,
        avg_alpha_vs_spy=None,
        total_cost_usd=0.0,
        strategy_mode="conservative",
        trailing_stop_multiplier=1.0,
        proposal_json=None,
        applied_changes=None,
        report_text=None,
    )

    app.dependency_overrides[get_db] = lambda: db
    app.dependency_overrides[get_current_user] = _bypass_user_tenant
    app.state.db = db
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        _reset_rate_limiter()
        resp = await c.get("/api/agent/improvements")
        assert resp.status_code == 200
        assert resp.json() == []  # t1 sees nothing from default
    app.dependency_overrides.clear()
    app.state.db = None

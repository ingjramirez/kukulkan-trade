"""Tests for agent insights API endpoints: posture, playbook, calibration."""

from datetime import date

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


# ── Posture Endpoints ────────────────────────────────────────────────────────


class TestGetPosture:
    async def test_get_posture_empty(self, client: AsyncClient) -> None:
        resp = await client.get("/api/agent/posture")
        assert resp.status_code == 200
        assert resp.json() == []

    async def test_get_posture_with_data(self, client: AsyncClient, db: Database) -> None:
        await db.save_posture(
            tenant_id="default",
            session_date=date.today(),
            session_label="Morning",
            posture="defensive",
            effective_posture="defensive",
            reason="VIX elevated above 25",
        )

        resp = await client.get("/api/agent/posture")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        row = data[0]
        assert row["session_date"] == date.today().isoformat()
        assert row["session_label"] == "Morning"
        assert row["posture"] == "defensive"
        assert row["effective_posture"] == "defensive"
        assert row["reason"] == "VIX elevated above 25"
        assert "created_at" in row


# ── Playbook Endpoints ───────────────────────────────────────────────────────


class TestGetPlaybook:
    async def test_get_playbook_empty(self, client: AsyncClient) -> None:
        resp = await client.get("/api/agent/playbook")
        assert resp.status_code == 200
        assert resp.json() == []

    async def test_get_playbook_with_data(self, client: AsyncClient, db: Database) -> None:
        cells = [
            {
                "regime": "BULL",
                "sector": "Technology",
                "total_trades": 15,
                "wins": 10,
                "losses": 4,
                "win_rate_pct": 71.4,
                "avg_pnl_pct": 2.5,
                "recommendation": "sweet_spot",
            },
            {
                "regime": "BEAR",
                "sector": "Healthcare",
                "total_trades": 8,
                "wins": 3,
                "losses": 5,
                "win_rate_pct": 37.5,
                "avg_pnl_pct": -1.2,
                "recommendation": "avoid",
            },
        ]
        await db.save_playbook_snapshot(cells, tenant_id="default")

        resp = await client.get("/api/agent/playbook")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 2

        regimes = {row["regime"] for row in data}
        assert regimes == {"BULL", "BEAR"}

        bull_row = next(r for r in data if r["regime"] == "BULL")
        assert bull_row["sector"] == "Technology"
        assert bull_row["total_trades"] == 15
        assert bull_row["wins"] == 10
        assert bull_row["losses"] == 4
        assert bull_row["win_rate_pct"] == pytest.approx(71.4)
        assert bull_row["avg_pnl_pct"] == pytest.approx(2.5)
        assert bull_row["recommendation"] == "sweet_spot"


# ── Calibration Endpoints ────────────────────────────────────────────────────


class TestGetCalibration:
    async def test_get_calibration_empty(self, client: AsyncClient) -> None:
        resp = await client.get("/api/agent/calibration")
        assert resp.status_code == 200
        assert resp.json() == []

    async def test_get_calibration_with_data(self, client: AsyncClient, db: Database) -> None:
        buckets = [
            {
                "conviction_level": "high",
                "total_trades": 20,
                "wins": 14,
                "losses": 4,
                "win_rate_pct": 77.8,
                "avg_pnl_pct": 4.21,
                "assessment": "validated",
                "suggested_multiplier": 1.2,
            },
            {
                "conviction_level": "low",
                "total_trades": 10,
                "wins": 4,
                "losses": 5,
                "win_rate_pct": 44.4,
                "avg_pnl_pct": -0.5,
                "assessment": "over_confident",
                "suggested_multiplier": 0.7,
            },
        ]
        await db.save_conviction_calibration(buckets, tenant_id="default")

        resp = await client.get("/api/agent/calibration")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 2

        levels = {row["conviction_level"] for row in data}
        assert levels == {"high", "low"}

        high = next(r for r in data if r["conviction_level"] == "high")
        assert high["total_trades"] == 20
        assert high["wins"] == 14
        assert high["losses"] == 4
        assert high["win_rate_pct"] == pytest.approx(77.8)
        assert high["avg_pnl_pct"] == pytest.approx(4.21)
        assert high["assessment"] == "validated"
        assert high["suggested_multiplier"] == pytest.approx(1.2)


# ── Auth Required ─────────────────────────────────────────────────────────────


class TestEndpointsRequireAuth:
    async def test_posture_requires_auth(self, db: Database) -> None:
        _reset_rate_limiter()
        app.dependency_overrides[get_db] = lambda: db
        # No get_current_user override → should fail auth
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.get("/api/agent/posture")
            assert resp.status_code in (401, 403)
        app.dependency_overrides.clear()

    async def test_playbook_requires_auth(self, db: Database) -> None:
        _reset_rate_limiter()
        app.dependency_overrides[get_db] = lambda: db
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.get("/api/agent/playbook")
            assert resp.status_code in (401, 403)
        app.dependency_overrides.clear()

    async def test_calibration_requires_auth(self, db: Database) -> None:
        _reset_rate_limiter()
        app.dependency_overrides[get_db] = lambda: db
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.get("/api/agent/calibration")
            assert resp.status_code in (401, 403)
        app.dependency_overrides.clear()

"""Tests for the discovered tickers API endpoints."""

from datetime import date

import pytest
from httpx import ASGITransport, AsyncClient

from src.api.deps import get_current_user, get_db
from src.api.main import app
from src.api.rate_limit import RateLimitMiddleware
from src.storage.database import Database
from src.storage.models import DiscoveredTickerRow


def _reset_rate_limiter() -> None:
    """Find and reset the rate limiter middleware on the app."""
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
    await test_db.ensure_tenant("t-1")
    await test_db.ensure_tenant("t-2")
    yield test_db
    await test_db.close()


@pytest.fixture
async def admin_client(db):
    """Admin client (can query any tenant via ?tenant_id=)."""
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
    """Tenant user client (locked to t-1)."""
    _reset_rate_limiter()
    app.dependency_overrides[get_db] = lambda: db
    app.dependency_overrides[get_current_user] = _bypass_tenant_user
    app.state.db = db
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c
    app.dependency_overrides.clear()
    app.state.db = None


async def _seed_tickers(db: Database) -> None:
    """Insert test discovered tickers for t-1 and t-2."""
    await db.save_discovered_ticker(
        DiscoveredTickerRow(
            tenant_id="t-1",
            ticker="PLTR",
            source="agent",
            rationale="AI defense play",
            status="proposed",
            proposed_at=date(2026, 2, 5),
            expires_at=date(2026, 3, 7),
            sector="Technology",
            market_cap=50e9,
        )
    )
    await db.save_discovered_ticker(
        DiscoveredTickerRow(
            tenant_id="t-1",
            ticker="ORCL",
            source="agent",
            rationale="Cloud growth",
            status="approved",
            proposed_at=date(2026, 2, 3),
            expires_at=date(2026, 3, 5),
            sector="Technology",
            market_cap=200e9,
        )
    )
    await db.save_discovered_ticker(
        DiscoveredTickerRow(
            tenant_id="t-2",
            ticker="UBER",
            source="agent",
            rationale="Mobility",
            status="proposed",
            proposed_at=date(2026, 2, 4),
            expires_at=date(2026, 3, 6),
        )
    )


class TestListDiscoveredTickers:
    async def test_list_all_for_tenant(self, tenant_client, db) -> None:
        await _seed_tickers(db)
        resp = await tenant_client.get("/api/discovered")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 2  # only t-1's tickers
        tickers = {d["ticker"] for d in data}
        assert tickers == {"PLTR", "ORCL"}

    async def test_filter_by_status(self, tenant_client, db) -> None:
        await _seed_tickers(db)
        resp = await tenant_client.get("/api/discovered?status_filter=proposed")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["ticker"] == "PLTR"

    async def test_admin_queries_specific_tenant(self, admin_client, db) -> None:
        await _seed_tickers(db)
        resp = await admin_client.get("/api/discovered?tenant_id=t-2")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["ticker"] == "UBER"

    async def test_empty_list(self, tenant_client, db) -> None:
        resp = await tenant_client.get("/api/discovered")
        assert resp.status_code == 200
        assert resp.json() == []

    async def test_tenant_isolation(self, tenant_client, db) -> None:
        """Tenant user cannot see other tenants' tickers."""
        await _seed_tickers(db)
        resp = await tenant_client.get("/api/discovered")
        data = resp.json()
        # Should NOT contain UBER (t-2)
        assert all(d["ticker"] != "UBER" for d in data)


class TestUpdateDiscoveredTicker:
    async def test_approve_proposed(self, tenant_client, db) -> None:
        await _seed_tickers(db)
        resp = await tenant_client.patch(
            "/api/discovered/PLTR",
            json={"status": "approved"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["ticker"] == "PLTR"
        assert data["status"] == "approved"

        # Verify in DB
        row = await db.get_discovered_ticker("PLTR", tenant_id="t-1")
        assert row.status == "approved"

    async def test_reject_proposed(self, tenant_client, db) -> None:
        await _seed_tickers(db)
        resp = await tenant_client.patch(
            "/api/discovered/PLTR",
            json={"status": "rejected"},
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "rejected"

    async def test_404_unknown_ticker(self, tenant_client, db) -> None:
        resp = await tenant_client.patch(
            "/api/discovered/NOPE",
            json={"status": "approved"},
        )
        assert resp.status_code == 404

    async def test_409_already_processed(self, tenant_client, db) -> None:
        await _seed_tickers(db)
        # ORCL is already approved
        resp = await tenant_client.patch(
            "/api/discovered/ORCL",
            json={"status": "approved"},
        )
        assert resp.status_code == 409

    async def test_422_invalid_status(self, tenant_client, db) -> None:
        await _seed_tickers(db)
        resp = await tenant_client.patch(
            "/api/discovered/PLTR",
            json={"status": "expired"},  # not allowed
        )
        assert resp.status_code == 422

    async def test_case_insensitive_ticker(self, tenant_client, db) -> None:
        await _seed_tickers(db)
        resp = await tenant_client.patch(
            "/api/discovered/pltr",  # lowercase
            json={"status": "approved"},
        )
        assert resp.status_code == 200
        assert resp.json()["ticker"] == "PLTR"

    async def test_tenant_cannot_update_other_tenants_ticker(
        self,
        tenant_client,
        db,
    ) -> None:
        """Tenant user t-1 cannot modify t-2's ticker."""
        await _seed_tickers(db)
        resp = await tenant_client.patch(
            "/api/discovered/UBER",
            json={"status": "approved"},
        )
        assert resp.status_code == 404  # not found for this tenant


class TestRequiresAuth:
    async def test_unauthenticated_list(self, db) -> None:
        _reset_rate_limiter()
        app.dependency_overrides[get_db] = lambda: db
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.get("/api/discovered")
            assert resp.status_code in (401, 403)
        app.dependency_overrides.clear()

    async def test_unauthenticated_patch(self, db) -> None:
        _reset_rate_limiter()
        app.dependency_overrides[get_db] = lambda: db
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.patch(
                "/api/discovered/PLTR",
                json={"status": "approved"},
            )
            assert resp.status_code in (401, 403)
        app.dependency_overrides.clear()

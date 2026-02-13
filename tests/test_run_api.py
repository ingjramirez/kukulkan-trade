"""Tests for POST /api/run — trigger bot run per tenant."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from cryptography.fernet import Fernet
from httpx import ASGITransport, AsyncClient

from config.settings import settings
from src.api.auth import create_access_token
from src.api.main import app
from src.api.rate_limit import RateLimitMiddleware
from src.api.routes.run import _reset_run_state, _running
from src.storage.database import Database
from src.storage.models import TenantRow
from src.utils.crypto import encrypt_value

_TEST_KEY = Fernet.generate_key().decode()


def _reset_rate_limiter() -> None:
    """Find and reset the rate limiter middleware on the app."""
    handler = app.middleware_stack
    while handler:
        if isinstance(handler, RateLimitMiddleware):
            handler.reset()
            return
        handler = getattr(handler, "app", None)


@pytest.fixture(autouse=True)
def _set_encryption_key(monkeypatch):
    monkeypatch.setattr(settings, "tenant_encryption_key", _TEST_KEY)


@pytest.fixture
async def db():
    database = Database(url="sqlite+aiosqlite:///:memory:")
    await database.init_db()
    yield database
    await database.close()


@pytest.fixture
async def client(db):
    _reset_rate_limiter()
    _reset_run_state()
    app.state.db = db
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c
    # Clean up to avoid polluting other test modules with a stale db reference
    app.state.db = None


@pytest.fixture
def admin_headers():
    token = create_access_token("admin")
    return {"Authorization": f"Bearer {token}"}


async def _create_fully_configured_tenant(db: Database) -> TenantRow:
    """Insert a tenant with all required credentials."""
    tenant = TenantRow(
        id="test-tenant-001",
        name="TestTenant",
        alpaca_api_key_enc=encrypt_value("PKTEST123"),
        alpaca_api_secret_enc=encrypt_value("SECRET456"),
        telegram_bot_token_enc=encrypt_value("123456:ABC"),
        telegram_chat_id_enc=encrypt_value("999888"),
        dashboard_user="testuser",
        dashboard_password_enc="hashed",
    )
    await db.create_tenant(tenant)
    return tenant


async def _create_unconfigured_tenant(db: Database) -> TenantRow:
    """Insert a tenant without Alpaca/Telegram credentials."""
    tenant = TenantRow(
        id="test-tenant-002",
        name="UnconfiguredTenant",
        dashboard_user="noconfig",
        dashboard_password_enc="hashed",
    )
    await db.create_tenant(tenant)
    return tenant


def _tenant_headers(tenant_id: str) -> dict:
    """Generate auth headers for a tenant user."""
    token = create_access_token("testuser", tenant_id=tenant_id)
    return {"Authorization": f"Bearer {token}"}


class TestTriggerRun:
    @patch("src.api.routes.run._run_pipeline", new_callable=AsyncMock)
    async def test_happy_path_returns_202(self, mock_pipeline, client, db):
        tenant = await _create_fully_configured_tenant(db)
        headers = _tenant_headers(tenant.id)

        r = await client.post("/api/run", headers=headers)
        assert r.status_code == 202
        data = r.json()
        assert data["status"] == "triggered"
        assert data["tenant_id"] == tenant.id

        mock_pipeline.assert_called_once_with(db, tenant.id)

    async def test_unauthenticated_returns_401(self, client):
        r = await client.post("/api/run")
        assert r.status_code == 401

    @patch("src.api.routes.run._run_pipeline", new_callable=AsyncMock)
    async def test_tenant_not_found_returns_404(self, mock_pipeline, client):
        headers = _tenant_headers("nonexistent-tenant-id")
        r = await client.post("/api/run", headers=headers)
        assert r.status_code == 404

    @patch("src.api.routes.run._run_pipeline", new_callable=AsyncMock)
    async def test_incomplete_credentials_returns_422(
        self,
        mock_pipeline,
        client,
        db,
    ):
        tenant = await _create_unconfigured_tenant(db)
        headers = _tenant_headers(tenant.id)

        r = await client.post("/api/run", headers=headers)
        assert r.status_code == 422
        assert "credentials" in r.json()["detail"].lower()
        mock_pipeline.assert_not_called()

    @patch("src.api.routes.run._run_pipeline", new_callable=AsyncMock)
    async def test_concurrent_run_returns_409(self, mock_pipeline, client, db):
        tenant = await _create_fully_configured_tenant(db)
        headers = _tenant_headers(tenant.id)

        # Simulate an already-running pipeline
        _running[tenant.id] = True

        r = await client.post("/api/run", headers=headers)
        assert r.status_code == 409
        assert "already in progress" in r.json()["detail"].lower()
        mock_pipeline.assert_not_called()

    @patch("src.api.routes.run._run_pipeline", new_callable=AsyncMock)
    async def test_rate_limit_returns_429(self, mock_pipeline, client, db):
        tenant = await _create_fully_configured_tenant(db)
        headers = _tenant_headers(tenant.id)

        # First request succeeds
        r1 = await client.post("/api/run", headers=headers)
        assert r1.status_code == 202

        # Second request within rate window should be rate-limited
        # (need to clear concurrency lock since mock doesn't run _run_pipeline's finally)
        _running.pop(tenant.id, None)

        r2 = await client.post("/api/run", headers=headers)
        assert r2.status_code == 429
        assert "recently" in r2.json()["detail"].lower()

    @patch("src.api.routes.run._run_pipeline", new_callable=AsyncMock)
    async def test_admin_with_tenant_id_param(self, mock_pipeline, client, db):
        tenant = await _create_fully_configured_tenant(db)

        r = await client.post(
            f"/api/run?tenant_id={tenant.id}",
            headers=self._admin_headers(),
        )
        assert r.status_code == 202
        assert r.json()["tenant_id"] == tenant.id
        mock_pipeline.assert_called_once_with(db, tenant.id)

    @patch("src.api.routes.run._run_pipeline", new_callable=AsyncMock)
    async def test_admin_default_tenant_not_found(self, mock_pipeline, client, db):
        """Admin without tenant_id param uses 'default', which may not exist."""
        r = await client.post("/api/run", headers=self._admin_headers())
        assert r.status_code == 404

    @staticmethod
    def _admin_headers() -> dict:
        token = create_access_token("admin")
        return {"Authorization": f"Bearer {token}"}

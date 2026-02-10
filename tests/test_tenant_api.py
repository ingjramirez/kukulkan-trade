"""Tests for tenant admin API endpoints."""

import pytest
from cryptography.fernet import Fernet
from httpx import ASGITransport, AsyncClient

from config.settings import settings
from src.api.auth import create_access_token
from src.api.main import app
from src.api.rate_limit import RateLimitMiddleware
from src.storage.database import Database

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
    app.state.db = db
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest.fixture
def auth_headers():
    token = create_access_token("admin")
    return {"Authorization": f"Bearer {token}"}


class TestCreateTenant:
    async def test_create_tenant(self, client, auth_headers):
        r = await client.post(
            "/api/tenants",
            json={
                "name": "Papa",
                "alpaca_api_key": "PKOS12345678abcd",
                "alpaca_api_secret": "APCA-SECRET-LONG",
                "telegram_bot_token": "123456:ABCdefGHIjklMNO",
                "telegram_chat_id": "987654321",
                "strategy_mode": "aggressive",
            },
            headers=auth_headers,
        )
        assert r.status_code == 201
        data = r.json()
        assert data["name"] == "Papa"
        assert data["strategy_mode"] == "aggressive"
        assert "..." in data["alpaca_api_key_masked"]
        # Full credentials should NOT appear
        assert "PKOS12345678abcd" != data["alpaca_api_key_masked"]

    async def test_create_requires_auth(self, client):
        r = await client.post(
            "/api/tenants",
            json={
                "name": "X",
                "alpaca_api_key": "k",
                "alpaca_api_secret": "s",
                "telegram_bot_token": "t",
                "telegram_chat_id": "c",
            },
        )
        assert r.status_code == 401

    async def test_invalid_strategy(self, client, auth_headers):
        r = await client.post(
            "/api/tenants",
            json={
                "name": "Bad",
                "alpaca_api_key": "k",
                "alpaca_api_secret": "s",
                "telegram_bot_token": "t",
                "telegram_chat_id": "c",
                "strategy_mode": "yolo",
            },
            headers=auth_headers,
        )
        assert r.status_code == 422


class TestListTenants:
    async def test_list_empty(self, client, auth_headers):
        r = await client.get("/api/tenants", headers=auth_headers)
        assert r.status_code == 200
        assert r.json() == []

    async def test_list_after_create(self, client, auth_headers):
        await client.post(
            "/api/tenants",
            json={
                "name": "T1",
                "alpaca_api_key": "k",
                "alpaca_api_secret": "s",
                "telegram_bot_token": "t",
                "telegram_chat_id": "c",
            },
            headers=auth_headers,
        )
        r = await client.get("/api/tenants", headers=auth_headers)
        assert len(r.json()) == 1


class TestGetTenant:
    async def test_get_nonexistent(self, client, auth_headers):
        r = await client.get("/api/tenants/nope", headers=auth_headers)
        assert r.status_code == 404

    async def test_get_existing(self, client, auth_headers):
        create = await client.post(
            "/api/tenants",
            json={
                "name": "T1",
                "alpaca_api_key": "k",
                "alpaca_api_secret": "s",
                "telegram_bot_token": "t",
                "telegram_chat_id": "c",
            },
            headers=auth_headers,
        )
        tid = create.json()["id"]
        r = await client.get(f"/api/tenants/{tid}", headers=auth_headers)
        assert r.status_code == 200
        assert r.json()["name"] == "T1"


class TestUpdateTenant:
    async def test_update_strategy(self, client, auth_headers):
        create = await client.post(
            "/api/tenants",
            json={
                "name": "T1",
                "alpaca_api_key": "k",
                "alpaca_api_secret": "s",
                "telegram_bot_token": "t",
                "telegram_chat_id": "c",
                "strategy_mode": "conservative",
            },
            headers=auth_headers,
        )
        tid = create.json()["id"]
        r = await client.patch(
            f"/api/tenants/{tid}",
            json={"strategy_mode": "aggressive"},
            headers=auth_headers,
        )
        assert r.status_code == 200
        assert r.json()["strategy_mode"] == "aggressive"

    async def test_update_nonexistent(self, client, auth_headers):
        r = await client.patch(
            "/api/tenants/nope",
            json={"name": "x"},
            headers=auth_headers,
        )
        assert r.status_code == 404


    async def test_update_tickers_without_credentials(self, client, auth_headers):
        """Ticker customization should work without Alpaca/Telegram credentials."""
        create = await client.post(
            "/api/tenants",
            json={"name": "NoCreds", "username": "nocreds", "password": "pass123"},
            headers=auth_headers,
        )
        tid = create.json()["id"]
        r = await client.patch(
            f"/api/tenants/{tid}",
            json={"ticker_additions": ["INTL", "COIN"]},
            headers=auth_headers,
        )
        assert r.status_code == 200
        assert r.json()["ticker_additions"] == ["INTL", "COIN"]

    async def test_update_portfolio_config_requires_credentials(self, client, auth_headers):
        """Portfolio trading config (strategy_mode, etc.) still requires credentials."""
        create = await client.post(
            "/api/tenants",
            json={"name": "NoCreds2", "username": "nocreds2", "password": "pass456"},
            headers=auth_headers,
        )
        tid = create.json()["id"]
        r = await client.patch(
            f"/api/tenants/{tid}",
            json={"strategy_mode": "aggressive"},
            headers=auth_headers,
        )
        assert r.status_code == 422
        assert "credentials" in r.json()["detail"].lower()


class TestDeactivateTenant:
    async def test_deactivate(self, client, auth_headers):
        create = await client.post(
            "/api/tenants",
            json={
                "name": "T1",
                "alpaca_api_key": "k",
                "alpaca_api_secret": "s",
                "telegram_bot_token": "t",
                "telegram_chat_id": "c",
            },
            headers=auth_headers,
        )
        tid = create.json()["id"]
        r = await client.delete(f"/api/tenants/{tid}", headers=auth_headers)
        assert r.status_code == 204

        # Verify it's inactive
        get = await client.get(f"/api/tenants/{tid}", headers=auth_headers)
        assert get.json()["is_active"] is False

    async def test_deactivate_nonexistent(self, client, auth_headers):
        r = await client.delete("/api/tenants/nope", headers=auth_headers)
        assert r.status_code == 404


class TestTenantSelfService:
    """Tests for /api/tenants/me — tenant user self-service."""

    async def _create_tenant_and_get_headers(
        self, client: AsyncClient, auth_headers: dict,
    ) -> tuple[str, dict]:
        """Create a tenant with login creds and return (tenant_id, tenant_headers)."""
        r = await client.post(
            "/api/tenants",
            json={"name": "SelfUser", "username": "selfuser", "password": "pass123"},
            headers=auth_headers,
        )
        tid = r.json()["id"]
        tenant_token = create_access_token("selfuser", tenant_id=tid)
        return tid, {"Authorization": f"Bearer {tenant_token}"}

    async def test_get_me(self, client, auth_headers):
        tid, tenant_headers = await self._create_tenant_and_get_headers(
            client, auth_headers,
        )
        r = await client.get("/api/tenants/me", headers=tenant_headers)
        assert r.status_code == 200
        assert r.json()["id"] == tid
        assert r.json()["name"] == "SelfUser"

    async def test_get_me_admin_rejected(self, client, auth_headers):
        """Admin users (no tenant_id) should get 403 on /me."""
        r = await client.get("/api/tenants/me", headers=auth_headers)
        assert r.status_code == 403

    async def test_patch_me_ticker_additions(self, client, auth_headers):
        _, tenant_headers = await self._create_tenant_and_get_headers(
            client, auth_headers,
        )
        r = await client.patch(
            "/api/tenants/me",
            json={"ticker_additions": ["INTL", "COIN"]},
            headers=tenant_headers,
        )
        assert r.status_code == 200
        assert r.json()["ticker_additions"] == ["INTL", "COIN"]

    async def test_patch_me_ticker_exclusions(self, client, auth_headers):
        _, tenant_headers = await self._create_tenant_and_get_headers(
            client, auth_headers,
        )
        r = await client.patch(
            "/api/tenants/me",
            json={"ticker_exclusions": ["AAPL", "MSFT"]},
            headers=tenant_headers,
        )
        assert r.status_code == 200
        assert r.json()["ticker_exclusions"] == ["AAPL", "MSFT"]

    async def test_patch_me_no_admin_fields(self, client, auth_headers):
        """Tenant users cannot set admin-level fields via /me."""
        _, tenant_headers = await self._create_tenant_and_get_headers(
            client, auth_headers,
        )
        # strategy_mode is not in TenantSelfUpdateRequest — Pydantic ignores it
        r = await client.patch(
            "/api/tenants/me",
            json={"strategy_mode": "aggressive", "ticker_additions": ["INTL"]},
            headers=tenant_headers,
        )
        assert r.status_code == 200
        # strategy_mode should remain default (conservative), not changed
        assert r.json()["strategy_mode"] == "conservative"
        assert r.json()["ticker_additions"] == ["INTL"]

    async def test_patch_me_admin_rejected(self, client, auth_headers):
        """Admin users (no tenant_id) should get 403 on PATCH /me."""
        r = await client.patch(
            "/api/tenants/me",
            json={"ticker_additions": ["INTL"]},
            headers=auth_headers,
        )
        assert r.status_code == 403

    async def test_patch_me_unauthenticated(self, client):
        """No token should get 401."""
        r = await client.patch(
            "/api/tenants/me",
            json={"ticker_additions": ["INTL"]},
        )
        assert r.status_code == 401

    async def test_patch_me_alpaca_credentials(self, client, auth_headers):
        """Tenant users can update their own Alpaca credentials via /me."""
        _, tenant_headers = await self._create_tenant_and_get_headers(
            client, auth_headers,
        )
        # Initially no Alpaca key
        r = await client.get("/api/tenants/me", headers=tenant_headers)
        assert r.json()["alpaca_api_key_masked"] is None

        # Update credentials
        r = await client.patch(
            "/api/tenants/me",
            json={
                "alpaca_api_key": "PKTEST123456",
                "alpaca_api_secret": "secret789",
            },
            headers=tenant_headers,
        )
        assert r.status_code == 200
        assert r.json()["alpaca_api_key_masked"] is not None
        assert "..." in r.json()["alpaca_api_key_masked"]
        # Full credential must not appear
        assert "PKTEST123456" not in str(r.json())

    async def test_patch_me_telegram_credentials(self, client, auth_headers):
        """Tenant users can update their own Telegram credentials via /me."""
        _, tenant_headers = await self._create_tenant_and_get_headers(
            client, auth_headers,
        )
        r = await client.patch(
            "/api/tenants/me",
            json={
                "telegram_bot_token": "123456:ABCdef",
                "telegram_chat_id": "999888777",
            },
            headers=tenant_headers,
        )
        assert r.status_code == 200
        assert r.json()["telegram_chat_id_masked"] is not None
        assert "..." in r.json()["telegram_chat_id_masked"]

    async def test_me_test_alpaca_no_creds(self, client, auth_headers):
        """Test Alpaca connection fails gracefully when no credentials."""
        _, tenant_headers = await self._create_tenant_and_get_headers(
            client, auth_headers,
        )
        r = await client.post("/api/tenants/me/test-alpaca", headers=tenant_headers)
        assert r.status_code == 200
        assert r.json()["success"] is False
        assert "not configured" in r.json()["error"]

    async def test_me_test_telegram_no_creds(self, client, auth_headers):
        """Test Telegram connection fails gracefully when no credentials."""
        _, tenant_headers = await self._create_tenant_and_get_headers(
            client, auth_headers,
        )
        r = await client.post("/api/tenants/me/test-telegram", headers=tenant_headers)
        assert r.status_code == 200
        assert r.json()["success"] is False
        assert "not configured" in r.json()["error"]

    async def test_me_test_alpaca_admin_rejected(self, client, auth_headers):
        """Admin users (no tenant_id) should get 403 on /me/test-alpaca."""
        r = await client.post("/api/tenants/me/test-alpaca", headers=auth_headers)
        assert r.status_code == 403

    async def test_me_test_telegram_admin_rejected(self, client, auth_headers):
        """Admin users (no tenant_id) should get 403 on /me/test-telegram."""
        r = await client.post("/api/tenants/me/test-telegram", headers=auth_headers)
        assert r.status_code == 403


class TestCredentialMasking:
    async def test_credentials_never_exposed(self, client, auth_headers):
        await client.post(
            "/api/tenants",
            json={
                "name": "T1",
                "alpaca_api_key": "PK_LIVE_supersecretkey1234",
                "alpaca_api_secret": "SK_LIVE_anothersecretkey56",
                "telegram_bot_token": "123456:ABCdefGHIjklMNOpqrsTUVwxyz",
                "telegram_chat_id": "987654321",
            },
            headers=auth_headers,
        )
        r = await client.get("/api/tenants", headers=auth_headers)
        data = r.json()[0]

        # Full credentials must never appear in responses
        assert "PK_LIVE_supersecretkey1234" not in str(data)
        assert "SK_LIVE_anothersecretkey56" not in str(data)
        assert "ABCdefGHIjklMNOpqrsTUVwxyz" not in str(data)

        # Masked versions should be present
        assert "..." in data["alpaca_api_key_masked"]
        assert "..." in data["telegram_chat_id_masked"]

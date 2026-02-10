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

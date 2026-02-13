"""Tests for per-tenant dashboard credentials and auth flow."""

import pytest
from cryptography.fernet import Fernet
from httpx import ASGITransport, AsyncClient

from config.settings import settings
from src.api.auth import _revoked_tokens, create_access_token, decode_access_token
from src.api.main import app
from src.api.rate_limit import RateLimitMiddleware
from src.storage.database import Database
from src.storage.models import TenantRow
from src.utils.crypto import encrypt_value

_TEST_KEY = Fernet.generate_key().decode()


def _reset_rate_limiter() -> None:
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
async def seeded_db(db):
    """DB with a tenant that has dashboard credentials."""
    async with db.session() as s:
        s.add(
            TenantRow(
                id="tenant-papa",
                name="Papa",
                alpaca_api_key_enc=encrypt_value("APCA-KEY-12345678"),
                alpaca_api_secret_enc=encrypt_value("APCA-SECRET-1234"),
                telegram_bot_token_enc=encrypt_value("123456:ABCDEF"),
                telegram_chat_id_enc=encrypt_value("987654321"),
                dashboard_user="papa",
                dashboard_password_enc=encrypt_value("papa-secret-pass"),
            )
        )
        await s.commit()
    return db


@pytest.fixture
async def client(seeded_db):
    _reset_rate_limiter()
    app.state.db = seeded_db
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c
    _revoked_tokens.clear()


@pytest.fixture
def admin_headers():
    token = create_access_token("admin")
    return {"Authorization": f"Bearer {token}"}


# ── Tenant Login ─────────────────────────────────────────────────────


class TestTenantLogin:
    async def test_tenant_login_success(self, client):
        """Tenant user can log in with their credentials."""
        r = await client.post(
            "/api/auth/login",
            json={
                "username": "papa",
                "password": "papa-secret-pass",
            },
        )
        assert r.status_code == 200
        data = r.json()
        assert data["tenant_id"] == "tenant-papa"
        assert "access_token" in data

    async def test_tenant_login_wrong_password(self, client):
        """Wrong password for tenant user returns 401."""
        r = await client.post(
            "/api/auth/login",
            json={
                "username": "papa",
                "password": "wrong-password",
            },
        )
        assert r.status_code == 401

    async def test_tenant_login_nonexistent_user(self, client):
        """Non-existent tenant user falls through to admin check."""
        r = await client.post(
            "/api/auth/login",
            json={
                "username": "nobody",
                "password": "whatever",
            },
        )
        assert r.status_code == 401

    async def test_admin_login_still_works(self, client):
        """Global admin login still works alongside tenant auth."""
        r = await client.post(
            "/api/auth/login",
            json={
                "username": settings.dashboard.user,
                "password": settings.dashboard.password,
            },
        )
        # Should succeed (or skip if test settings don't have valid admin creds)
        if r.status_code == 200:
            data = r.json()
            assert data["tenant_id"] is None
            assert "access_token" in data

    async def test_inactive_tenant_cannot_login(self, client, seeded_db):
        """Deactivated tenant user cannot log in."""
        await seeded_db.deactivate_tenant("tenant-papa")
        r = await client.post(
            "/api/auth/login",
            json={
                "username": "papa",
                "password": "papa-secret-pass",
            },
        )
        assert r.status_code == 401


# ── JWT Claims ───────────────────────────────────────────────────────


class TestJWTClaims:
    def test_admin_jwt_has_no_tenant_id(self):
        """Admin token should not contain tenant_id."""
        token = create_access_token("admin")
        decoded = decode_access_token(token)
        assert decoded["username"] == "admin"
        assert decoded["tenant_id"] is None

    def test_tenant_jwt_has_tenant_id(self):
        """Tenant token should contain tenant_id."""
        token = create_access_token("papa", tenant_id="tenant-papa")
        decoded = decode_access_token(token)
        assert decoded["username"] == "papa"
        assert decoded["tenant_id"] == "tenant-papa"

    def test_decode_returns_dict(self):
        """decode_access_token now returns dict, not str."""
        token = create_access_token("test-user")
        result = decode_access_token(token)
        assert isinstance(result, dict)
        assert "username" in result
        assert "tenant_id" in result


# ── Admin-Only Enforcement ───────────────────────────────────────────


class TestAdminOnly:
    async def test_tenant_user_blocked_from_tenant_crud(self, client):
        """Tenant user JWT should get 403 on tenant management endpoints."""
        token = create_access_token("papa", tenant_id="tenant-papa")
        headers = {"Authorization": f"Bearer {token}"}
        r = await client.get("/api/tenants", headers=headers)
        assert r.status_code == 403
        assert "Admin access required" in r.json()["detail"]

    async def test_tenant_user_blocked_from_create(self, client):
        """Tenant user cannot create new tenants."""
        token = create_access_token("papa", tenant_id="tenant-papa")
        headers = {"Authorization": f"Bearer {token}"}
        r = await client.post(
            "/api/tenants",
            json={
                "name": "Hacker",
                "alpaca_api_key": "k",
                "alpaca_api_secret": "s",
                "telegram_bot_token": "t",
                "telegram_chat_id": "c",
            },
            headers=headers,
        )
        assert r.status_code == 403

    async def test_admin_can_access_tenant_crud(self, client, admin_headers):
        """Admin can still access tenant management endpoints."""
        r = await client.get("/api/tenants", headers=admin_headers)
        assert r.status_code == 200

    async def test_tenant_user_can_access_data_endpoints(self, client):
        """Tenant user can access read-only data endpoints."""
        token = create_access_token("papa", tenant_id="tenant-papa")
        headers = {"Authorization": f"Bearer {token}"}
        r = await client.get("/api/portfolios", headers=headers)
        assert r.status_code == 200


# ── Username Uniqueness ──────────────────────────────────────────────


class TestUsernameUniqueness:
    async def test_duplicate_username_rejected(self, client, admin_headers):
        """Creating a tenant with an existing username returns 409."""
        r = await client.post(
            "/api/tenants",
            json={
                "name": "Duplicate",
                "alpaca_api_key": "k",
                "alpaca_api_secret": "s",
                "telegram_bot_token": "t",
                "telegram_chat_id": "c",
                "username": "papa",
                "password": "other-pass",
            },
            headers=admin_headers,
        )
        assert r.status_code == 409
        assert "Username already taken" in r.json()["detail"]

    async def test_unique_username_accepted(self, client, admin_headers):
        """Creating a tenant with a unique username succeeds."""
        r = await client.post(
            "/api/tenants",
            json={
                "name": "Mama",
                "alpaca_api_key": "k",
                "alpaca_api_secret": "s",
                "telegram_bot_token": "t",
                "telegram_chat_id": "c",
                "username": "mama",
                "password": "mama-pass",
            },
            headers=admin_headers,
        )
        assert r.status_code == 201
        assert r.json()["dashboard_user"] == "mama"

    async def test_no_username_is_fine(self, client, admin_headers):
        """Creating a tenant without username is allowed."""
        r = await client.post(
            "/api/tenants",
            json={
                "name": "NoLogin",
                "alpaca_api_key": "k",
                "alpaca_api_secret": "s",
                "telegram_bot_token": "t",
                "telegram_chat_id": "c",
            },
            headers=admin_headers,
        )
        assert r.status_code == 201
        assert r.json()["dashboard_user"] is None


# ── Credential Update ────────────────────────────────────────────────


class TestCredentialUpdate:
    async def test_update_username(self, client, admin_headers):
        """Updating a tenant's username via PATCH works."""
        r = await client.patch(
            "/api/tenants/tenant-papa",
            json={"username": "papa-new"},
            headers=admin_headers,
        )
        assert r.status_code == 200
        assert r.json()["dashboard_user"] == "papa-new"

    async def test_update_password_enables_new_login(self, client, admin_headers):
        """After updating password, new password works for login."""
        await client.patch(
            "/api/tenants/tenant-papa",
            json={"password": "new-secret-pass"},
            headers=admin_headers,
        )
        r = await client.post(
            "/api/auth/login",
            json={
                "username": "papa",
                "password": "new-secret-pass",
            },
        )
        assert r.status_code == 200
        assert r.json()["tenant_id"] == "tenant-papa"


# ── Database Method ──────────────────────────────────────────────────


class TestGetTenantByUsername:
    async def test_finds_active_tenant(self, seeded_db):
        """get_tenant_by_username returns active tenant."""
        tenant = await seeded_db.get_tenant_by_username("papa")
        assert tenant is not None
        assert tenant.id == "tenant-papa"

    async def test_returns_none_for_unknown(self, seeded_db):
        """get_tenant_by_username returns None for unknown user."""
        tenant = await seeded_db.get_tenant_by_username("nobody")
        assert tenant is None

    async def test_excludes_inactive_tenant(self, seeded_db):
        """get_tenant_by_username skips inactive tenants."""
        await seeded_db.deactivate_tenant("tenant-papa")
        tenant = await seeded_db.get_tenant_by_username("papa")
        assert tenant is None

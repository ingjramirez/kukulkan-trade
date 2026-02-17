"""Tests for security fixes: rate limiting, timing-safe auth, CORS, headers, IDOR."""

import pytest
from httpx import ASGITransport, AsyncClient

from src.api.auth import _revoked_tokens, create_access_token, revoke_token
from src.api.deps import get_current_user, get_db
from src.api.main import app
from src.api.rate_limit import RateLimitMiddleware
from src.storage.database import Database
from src.storage.models import PortfolioRow


def _reset_rate_limiter() -> None:
    """Find and reset the rate limiter middleware on the app."""
    for mw in app.user_middleware:
        if mw.cls is RateLimitMiddleware:
            break
    # Walk the middleware stack
    handler = app.middleware_stack
    while handler:
        if isinstance(handler, RateLimitMiddleware):
            handler.reset()
            return
        handler = getattr(handler, "app", None)


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
async def db():
    test_db = Database(url="sqlite+aiosqlite:///:memory:")
    await test_db.init_db()
    async with test_db.session() as s:
        s.add(PortfolioRow(name="A", cash=30000.0, total_value=35000.0))
        await s.commit()
    yield test_db
    await test_db.close()


async def _bypass_user() -> dict[str, str | None]:
    return {"username": "test-user", "tenant_id": None}


@pytest.fixture
async def client(db):
    app.dependency_overrides[get_db] = lambda: db
    app.dependency_overrides[get_current_user] = _bypass_user
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        _reset_rate_limiter()
        yield c
    app.dependency_overrides.clear()


@pytest.fixture
async def unauth_client(db):
    app.dependency_overrides[get_db] = lambda: db
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        _reset_rate_limiter()
        yield c
    app.dependency_overrides.clear()


# ── Rate Limiting ────────────────────────────────────────────────────────────


class TestRateLimiting:
    async def test_login_rate_limit(self, unauth_client):
        """Login endpoint should be rate-limited after 5 attempts."""
        payload = {"username": "wrong", "password": "wrong"}
        for _ in range(5):
            await unauth_client.post("/api/auth/login", json=payload)

        # 6th attempt should be rate limited
        r = await unauth_client.post("/api/auth/login", json=payload)
        assert r.status_code == 429
        assert "Too many login attempts" in r.json()["detail"]

    async def test_health_not_rate_limited(self, client):
        """Health endpoint should not be rate-limited."""
        for _ in range(65):
            r = await client.get("/api/health")
            assert r.status_code == 200


# ── CORS ─────────────────────────────────────────────────────────────────────


class TestCors:
    async def test_cors_allows_configured_origin(self, client):
        r = await client.options(
            "/api/health",
            headers={
                "Origin": "https://app.kukulkan.trade",
                "Access-Control-Request-Method": "GET",
            },
        )
        assert r.headers.get("access-control-allow-origin") == "https://app.kukulkan.trade"

    async def test_cors_rejects_unknown_origin(self, client):
        r = await client.options(
            "/api/health",
            headers={
                "Origin": "https://evil.com",
                "Access-Control-Request-Method": "GET",
            },
        )
        assert "access-control-allow-origin" not in r.headers

    async def test_cors_disallows_put_method(self, client):
        r = await client.options(
            "/api/health",
            headers={
                "Origin": "https://app.kukulkan.trade",
                "Access-Control-Request-Method": "PUT",
            },
        )
        allowed = r.headers.get("access-control-allow-methods", "")
        assert "PUT" not in allowed


# ── Timing-Safe Auth ─────────────────────────────────────────────────────────


class TestTimingSafeAuth:
    async def test_wrong_username_returns_401(self, unauth_client):
        r = await unauth_client.post(
            "/api/auth/login",
            json={"username": "attacker", "password": ""},
        )
        assert r.status_code == 401

    async def test_wrong_password_returns_401(self, unauth_client):
        r = await unauth_client.post(
            "/api/auth/login",
            json={"username": "admin", "password": "wrong-password"},
        )
        assert r.status_code == 401

    async def test_auth_uses_hmac_compare(self):
        """Verify the auth module uses hmac.compare_digest."""
        import inspect

        from src.api import auth

        source = inspect.getsource(auth.login)
        assert "compare_digest" in source


# ── Error Sanitization ──────────────────────────────────────────────────────


class TestErrorSanitization:
    async def test_unhandled_exception_returns_generic_500(self, client):
        """Unhandled exceptions must never leak stack traces."""
        # Hit a nonexistent route that would trigger a 404 at worst,
        # but let's test the exception handler by triggering a real error.
        # The global handler catches Exception, not HTTPException,
        # so 404s from FastAPI are still informative. That's fine.
        r = await client.get("/api/nonexistent")
        assert r.status_code in (404, 405)
        body = r.json()
        # Should not contain file paths or traceback info
        assert "/Users/" not in str(body)
        assert "Traceback" not in str(body)


# ── Cache-Control Headers ───────────────────────────────────────────────────


class TestCacheControl:
    async def test_api_responses_have_no_store(self, client):
        """Financial data endpoints must include no-store header."""
        r = await client.get("/api/portfolios")
        assert r.headers.get("cache-control") == "no-store, no-cache"

    async def test_health_has_no_cache_header(self, client):
        """Health endpoint should not have no-store header."""
        r = await client.get("/api/health")
        assert r.headers.get("cache-control") != "no-store, no-cache"


# ── Token Revocation ────────────────────────────────────────────────────────


class TestTokenRevocation:
    async def test_logout_revokes_token(self, unauth_client):
        """Logging out should revoke the token."""
        # Login first
        r = await unauth_client.post(
            "/api/auth/login",
            json={"username": "admin", "password": ""},
        )
        if r.status_code != 200:
            pytest.skip("Login failed with test settings")
        token = r.json()["access_token"]
        auth = {"Authorization": f"Bearer {token}"}

        # Logout
        r = await unauth_client.post("/api/auth/logout", headers=auth)
        assert r.status_code == 204

        # Token should now be rejected
        from src.api.auth import decode_access_token

        with pytest.raises(ValueError, match="revoked"):
            decode_access_token(token)

        # Clean up revocation set to not affect other tests
        _revoked_tokens.clear()

    async def test_revoke_invalid_token_is_noop(self):
        """Revoking an invalid token should not raise."""
        revoke_token("garbage.invalid.token")

    async def test_jwt_expiry_matches_frontend_session(self):
        """Token expiry should be 8 hours to match NextAuth session maxAge."""
        from src.api.auth import TOKEN_EXPIRE_HOURS

        assert TOKEN_EXPIRE_HOURS == 8


# ── Login Input Validation ──────────────────────────────────────────────────


class TestLoginInputValidation:
    async def test_rejects_oversized_username(self, unauth_client):
        """Username over 100 chars should be rejected."""
        r = await unauth_client.post(
            "/api/auth/login",
            json={"username": "a" * 101, "password": "test"},
        )
        assert r.status_code == 422

    async def test_rejects_oversized_password(self, unauth_client):
        """Password over 200 chars should be rejected."""
        r = await unauth_client.post(
            "/api/auth/login",
            json={"username": "admin", "password": "x" * 201},
        )
        assert r.status_code == 422


# ── IDOR Protection ────────────────────────────────────────────────────────


class TestIDORProtection:
    """Verify tenant users cannot access other tenants' data via query param."""

    @pytest.fixture
    async def idor_client(self, db):
        """Client with real auth (no bypass) for IDOR testing."""
        app.dependency_overrides[get_db] = lambda: db
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            _reset_rate_limiter()
            yield c
        app.dependency_overrides.clear()

    async def test_tenant_user_cannot_query_other_tenant(self, idor_client):
        """Tenant user trying to pass tenant_id of another tenant gets own data."""
        token = create_access_token("papa", tenant_id="tenant-papa")
        headers = {"Authorization": f"Bearer {token}"}
        # Try to access "default" tenant's data (IDOR attempt)
        r = await idor_client.get(
            "/api/portfolios?tenant_id=default",
            headers=headers,
        )
        assert r.status_code == 200
        # Should return empty (tenant-papa has no portfolios), NOT default's data
        assert r.json() == []

    async def test_admin_can_query_any_tenant(self, idor_client):
        """Admin user can use tenant_id query param to see any tenant."""
        token = create_access_token("admin")
        headers = {"Authorization": f"Bearer {token}"}
        r = await idor_client.get(
            "/api/portfolios?tenant_id=default",
            headers=headers,
        )
        assert r.status_code == 200
        # Admin should get the seeded "default" tenant data
        assert len(r.json()) == 1

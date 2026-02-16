"""JWT authentication: token creation, logout, and login endpoint."""

import hmac
from datetime import datetime, timedelta, timezone

import structlog
from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt

from config.settings import settings
from src.api.schemas import LoginRequest, TokenResponse
from src.storage.database import Database
from src.utils.crypto import decrypt_value, hash_password, verify_password

log = structlog.get_logger()

_bearer = HTTPBearer()

router = APIRouter(prefix="/api/auth", tags=["auth"])

ALGORITHM = "HS256"
TOKEN_EXPIRE_HOURS = 2

# In-memory set of revoked JTI (JWT ID) values.
# Entries auto-expire with the token, so this stays small.
_revoked_tokens: set[str] = set()


def create_access_token(
    subject: str,
    *,
    tenant_id: str | None = None,
) -> str:
    now = datetime.now(timezone.utc)
    expire = now + timedelta(hours=TOKEN_EXPIRE_HOURS)
    jti = f"{subject}:{int(now.timestamp())}"
    payload: dict = {"sub": subject, "exp": expire, "jti": jti}
    if tenant_id is not None:
        payload["tenant_id"] = tenant_id
    return jwt.encode(payload, settings.jwt_secret, algorithm=ALGORITHM)


def decode_access_token(token: str) -> dict[str, str | None]:
    """Decode and validate a JWT. Returns {"username": str, "tenant_id": str | None}."""
    payload = jwt.decode(token, settings.jwt_secret, algorithms=[ALGORITHM])
    sub: str | None = payload.get("sub")
    if sub is None:
        raise ValueError("Missing subject claim")
    jti = payload.get("jti")
    if jti and jti in _revoked_tokens:
        raise ValueError("Token has been revoked")
    return {
        "username": sub,
        "tenant_id": payload.get("tenant_id"),
    }


def revoke_token(token: str) -> None:
    """Add a token's JTI to the revocation set."""
    try:
        payload = jwt.decode(token, settings.jwt_secret, algorithms=[ALGORITHM])
        jti = payload.get("jti")
        if jti:
            _revoked_tokens.add(jti)
    except (JWTError, ValueError):
        pass  # Invalid tokens are already effectively revoked


@router.post("/login", response_model=TokenResponse)
async def login(body: LoginRequest, request: Request) -> TokenResponse:
    """Authenticate via tenant credentials or global admin."""
    db: Database | None = getattr(
        getattr(request.app, "state", None),
        "db",
        None,
    )

    # 1. Try tenant login (if db is available)
    if db is not None:
        tenant = await db.get_tenant_by_username(body.username)
        if tenant and tenant.dashboard_password_enc:
            password_ok = False
            stored = tenant.dashboard_password_enc
            # Try bcrypt first (new format)
            if stored.startswith("$2"):
                password_ok = verify_password(body.password, stored)
            else:
                # Fallback: legacy Fernet-encrypted password
                try:
                    decrypted = decrypt_value(stored)
                    if hmac.compare_digest(body.password, decrypted):
                        password_ok = True
                        # Re-hash with bcrypt for future logins
                        new_hash = hash_password(body.password)
                        await db.update_tenant(
                            tenant.id,
                            {"dashboard_password_enc": new_hash},
                        )
                        log.info("password_migrated_to_bcrypt", tenant_id=tenant.id)
                except Exception:
                    pass  # Corrupted stored value — treat as wrong password

            if password_ok:
                token = create_access_token(
                    subject=body.username,
                    tenant_id=tenant.id,
                )
                return TokenResponse(
                    access_token=token,
                    tenant_id=tenant.id,
                )

    # 2. Fall back to global admin
    user_ok = hmac.compare_digest(body.username, settings.dashboard.user)
    pass_ok = hmac.compare_digest(body.password, settings.dashboard.password)
    if user_ok and pass_ok:
        token = create_access_token(subject=body.username)
        return TokenResponse(access_token=token)

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid credentials",
    )


@router.post("/logout", status_code=204)
async def logout(
    credentials: HTTPAuthorizationCredentials = Depends(_bearer),
):
    """Revoke the current token so it can no longer be used."""
    revoke_token(credentials.credentials)
    return None

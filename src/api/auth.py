"""JWT authentication: token creation, logout, and login endpoint."""

import hmac
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import jwt

from config.settings import settings
from src.api.schemas import LoginRequest, TokenResponse

_bearer = HTTPBearer()

router = APIRouter(prefix="/api/auth", tags=["auth"])

ALGORITHM = "HS256"
TOKEN_EXPIRE_HOURS = 2

# In-memory set of revoked JTI (JWT ID) values.
# Entries auto-expire with the token, so this stays small.
_revoked_tokens: set[str] = set()


def create_access_token(subject: str) -> str:
    now = datetime.now(timezone.utc)
    expire = now + timedelta(hours=TOKEN_EXPIRE_HOURS)
    jti = f"{subject}:{int(now.timestamp())}"
    payload = {"sub": subject, "exp": expire, "jti": jti}
    return jwt.encode(payload, settings.jwt_secret, algorithm=ALGORITHM)


def decode_access_token(token: str) -> str:
    """Decode and validate a JWT. Returns the subject claim."""
    payload = jwt.decode(token, settings.jwt_secret, algorithms=[ALGORITHM])
    sub: str | None = payload.get("sub")
    if sub is None:
        raise ValueError("Missing subject claim")
    jti = payload.get("jti")
    if jti and jti in _revoked_tokens:
        raise ValueError("Token has been revoked")
    return sub


def revoke_token(token: str) -> None:
    """Add a token's JTI to the revocation set."""
    try:
        payload = jwt.decode(token, settings.jwt_secret, algorithms=[ALGORITHM])
        jti = payload.get("jti")
        if jti:
            _revoked_tokens.add(jti)
    except Exception:
        pass  # Invalid tokens are already effectively revoked


@router.post("/login", response_model=TokenResponse)
async def login(body: LoginRequest) -> TokenResponse:
    user_ok = hmac.compare_digest(body.username, settings.dashboard.user)
    pass_ok = hmac.compare_digest(body.password, settings.dashboard.password)
    if not (user_ok and pass_ok):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
        )
    token = create_access_token(subject=body.username)
    return TokenResponse(access_token=token)


@router.post("/logout", status_code=204)
async def logout(
    credentials: HTTPAuthorizationCredentials = Depends(_bearer),
):
    """Revoke the current token so it can no longer be used."""
    revoke_token(credentials.credentials)
    return None

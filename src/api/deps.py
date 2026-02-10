"""FastAPI dependencies: database session and auth."""

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from src.api.auth import decode_access_token
from src.storage.database import Database

_bearer_scheme = HTTPBearer()


def get_db(request: Request) -> Database:
    return request.app.state.db


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(_bearer_scheme),
) -> dict[str, str | None]:
    """Decode JWT and return {"username": str, "tenant_id": str | None}."""
    try:
        return decode_access_token(credentials.credentials)
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
        )


async def require_admin(
    user: dict[str, str | None] = Depends(get_current_user),
) -> dict[str, str | None]:
    """Require the caller to be a global admin (no tenant_id in JWT)."""
    if user.get("tenant_id") is not None:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required",
        )
    return user

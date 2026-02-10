"""In-memory sliding-window rate limiter middleware."""

import time
from collections import defaultdict

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Per-IP sliding-window rate limiter.

    Args:
        app: ASGI app.
        general_rpm: Max requests per minute for all endpoints.
        login_rpm: Max requests per minute for login endpoint.
    """

    def __init__(self, app, general_rpm: int = 60, login_rpm: int = 5):
        super().__init__(app)
        self.general_rpm = general_rpm
        self.login_rpm = login_rpm
        # {ip: [timestamps]}
        self._general: dict[str, list[float]] = defaultdict(list)
        self._login: dict[str, list[float]] = defaultdict(list)

    def reset(self) -> None:
        """Clear all rate-limit state. Used in tests."""
        self._general.clear()
        self._login.clear()

    def _is_limited(
        self, bucket: dict[str, list[float]], ip: str, limit: int
    ) -> bool:
        now = time.monotonic()
        window = now - 60.0
        # Evict expired entries
        timestamps = bucket[ip] = [t for t in bucket[ip] if t > window]
        if len(timestamps) >= limit:
            return True
        timestamps.append(now)
        return False

    async def dispatch(self, request: Request, call_next) -> Response:
        ip = request.client.host if request.client else "unknown"
        path = request.url.path

        # Stricter limit on login
        if path == "/api/auth/login" and request.method == "POST":
            if self._is_limited(self._login, ip, self.login_rpm):
                return JSONResponse(
                    status_code=429,
                    content={"detail": "Too many login attempts. Try again later."},
                )

        # General limit on all endpoints (except health)
        if path != "/api/health":
            if self._is_limited(self._general, ip, self.general_rpm):
                return JSONResponse(
                    status_code=429,
                    content={"detail": "Rate limit exceeded. Try again later."},
                )

        return await call_next(request)

"""FastAPI application — read-only REST API for Kukulkan Trade."""

from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from config.settings import settings
from src.api.auth import router as auth_router
from src.api.rate_limit import RateLimitMiddleware
from src.api.routes.account import router as account_router
from src.api.routes.agent_insights import router as agent_insights_router
from src.api.routes.chat import router as chat_router
from src.api.routes.decisions import router as decisions_router
from src.api.routes.discovered import router as discovered_router
from src.api.routes.earnings import router as earnings_router
from src.api.routes.events import router as events_router
from src.api.routes.improvements import router as improvements_router
from src.api.routes.momentum import router as momentum_router
from src.api.routes.outcomes import router as outcomes_router
from src.api.routes.portfolios import router as portfolios_router
from src.api.routes.run import router as run_router
from src.api.routes.signals import router as signals_router
from src.api.routes.snapshots import router as snapshots_router
from src.api.routes.tenants import router as tenants_router
from src.api.routes.trades import router as trades_router
from src.api.routes.universe import router as universe_router
from src.storage.database import Database

log = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    db = Database(url=settings.database_url)
    await db.init_db()
    app.state.db = db
    yield
    await db.close()


app = FastAPI(
    title="Kukulkan Trade API",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(RateLimitMiddleware, general_rpm=60, login_rpm=5)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://app.kukulkan.trade",
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
)

app.include_router(auth_router)
app.include_router(account_router)
app.include_router(portfolios_router)
app.include_router(snapshots_router)
app.include_router(trades_router)
app.include_router(momentum_router)
app.include_router(decisions_router)
app.include_router(tenants_router)
app.include_router(universe_router)
app.include_router(earnings_router)
app.include_router(discovered_router)
app.include_router(outcomes_router)
app.include_router(agent_insights_router)
app.include_router(improvements_router)
app.include_router(run_router)
app.include_router(signals_router)
app.include_router(events_router)
app.include_router(chat_router)


@app.exception_handler(Exception)
async def _unhandled_exception_handler(request: Request, exc: Exception):
    """Catch-all: never leak stack traces or internal paths to clients."""
    log.error(
        "unhandled_exception",
        path=request.url.path,
        method=request.method,
        ip=request.client.host if request.client else "unknown",
        error=str(exc),
    )
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


@app.middleware("http")
async def _security_headers(request: Request, call_next):
    """Add Cache-Control and audit logging to every response."""
    response = await call_next(request)
    ip = request.client.host if request.client else "unknown"

    # Prevent caching of sensitive financial data
    if request.url.path.startswith("/api/") and request.url.path != "/api/health":
        response.headers["Cache-Control"] = "no-store, no-cache"

    # Audit log for auth and error responses
    if request.url.path == "/api/auth/login" and request.method == "POST":
        log.info(
            "auth_attempt",
            ip=ip,
            status=response.status_code,
            success=response.status_code == 200,
        )
    if response.status_code >= 400:
        log.warning(
            "api_error_response",
            ip=ip,
            method=request.method,
            path=request.url.path,
            status=response.status_code,
        )
    return response


@app.get("/api/health")
async def health():
    return {"status": "ok"}

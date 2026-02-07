"""FastAPI application — read-only REST API for Kukulkan Trade."""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config.settings import settings
from src.api.auth import router as auth_router
from src.api.routes.account import router as account_router
from src.api.routes.decisions import router as decisions_router
from src.api.routes.momentum import router as momentum_router
from src.api.routes.portfolios import router as portfolios_router
from src.api.routes.snapshots import router as snapshots_router
from src.api.routes.trades import router as trades_router
from src.storage.database import Database


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

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://app.kukulkan.trade",
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router)
app.include_router(account_router)
app.include_router(portfolios_router)
app.include_router(snapshots_router)
app.include_router(trades_router)
app.include_router(momentum_router)
app.include_router(decisions_router)


@app.get("/api/health")
async def health():
    return {"status": "ok"}

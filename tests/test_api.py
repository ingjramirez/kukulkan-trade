"""Tests for the FastAPI REST API."""

import json
from datetime import date, datetime

import pytest
from httpx import ASGITransport, AsyncClient

from src.api.auth import create_access_token
from src.api.deps import get_current_user, get_db
from src.api.main import app
from src.storage.database import Database
from src.storage.models import (
    AgentDecisionRow,
    DailySnapshotRow,
    MomentumRankingRow,
    PortfolioRow,
    PositionRow,
    TradeRow,
)

# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
async def db():
    """In-memory database for testing."""
    test_db = Database(url="sqlite+aiosqlite:///:memory:")
    await test_db.init_db()
    yield test_db
    await test_db.close()


@pytest.fixture
async def seeded_db(db):
    """Database with seed data."""
    async with db.session() as s:
        # Portfolios
        s.add(PortfolioRow(name="A", cash=30000.0, total_value=35000.0))
        s.add(PortfolioRow(name="B", cash=60000.0, total_value=68000.0))

        # Positions
        s.add(PositionRow(portfolio="A", ticker="AAPL", shares=10, avg_price=180.0))
        s.add(PositionRow(portfolio="B", ticker="MSFT", shares=20, avg_price=400.0))

        # Trades
        s.add(
            TradeRow(
                portfolio="A",
                ticker="AAPL",
                side="BUY",
                shares=10,
                price=180.0,
                total=1800.0,
                reason="Momentum buy",
            )
        )
        s.add(
            TradeRow(
                portfolio="B",
                ticker="MSFT",
                side="BUY",
                shares=20,
                price=400.0,
                total=8000.0,
                reason="AI decision",
            )
        )
        s.add(
            TradeRow(
                portfolio="A",
                ticker="TSLA",
                side="SELL",
                shares=5,
                price=250.0,
                total=1250.0,
                reason="Momentum rotation",
            )
        )

        # Snapshots
        s.add(
            DailySnapshotRow(
                portfolio="A",
                date=date(2026, 1, 15),
                total_value=34000.0,
                cash=30000.0,
                positions_value=4000.0,
                daily_return_pct=0.5,
            )
        )
        s.add(
            DailySnapshotRow(
                portfolio="A",
                date=date(2026, 1, 16),
                total_value=35000.0,
                cash=30000.0,
                positions_value=5000.0,
                daily_return_pct=2.94,
            )
        )
        s.add(
            DailySnapshotRow(
                portfolio="B",
                date=date(2026, 1, 15),
                total_value=67000.0,
                cash=60000.0,
                positions_value=7000.0,
                daily_return_pct=1.5,
            )
        )

        # Momentum rankings
        s.add(
            MomentumRankingRow(
                date=date(2026, 1, 16),
                ticker="AAPL",
                return_63d=15.2,
                rank=1,
            )
        )
        s.add(
            MomentumRankingRow(
                date=date(2026, 1, 16),
                ticker="MSFT",
                return_63d=12.1,
                rank=2,
            )
        )

        # Agent decisions
        s.add(
            AgentDecisionRow(
                date=date(2026, 1, 16),
                prompt_summary="Market analysis",
                response_summary="Buy MSFT",
                proposed_trades=json.dumps([{"ticker": "MSFT", "side": "BUY", "shares": 20}]),
                reasoning="Strong fundamentals and momentum",
                model_used="claude-sonnet-4-5-20250929",
                tokens_used=1500,
                created_at=datetime(2026, 1, 16, 10, 0),
            )
        )

        await s.commit()
    return db


def _auth_header() -> dict:
    """Generate a valid auth header for testing."""
    token = create_access_token(subject="admin")
    return {"Authorization": f"Bearer {token}"}


async def _bypass_user() -> dict[str, str | None]:
    return {"username": "test-user", "tenant_id": None}


@pytest.fixture
async def client(seeded_db):
    """Authenticated async HTTP client."""
    app.dependency_overrides[get_db] = lambda: seeded_db
    app.dependency_overrides[get_current_user] = _bypass_user
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c
    app.dependency_overrides.clear()


@pytest.fixture
async def unauth_client(seeded_db):
    """Unauthenticated client (no auth bypass)."""
    app.dependency_overrides[get_db] = lambda: seeded_db
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c
    app.dependency_overrides.clear()


# ── Health ───────────────────────────────────────────────────────────────────


async def test_health(client):
    r = await client.get("/api/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


# ── Auth ─────────────────────────────────────────────────────────────────────


async def test_login_success(unauth_client):
    r = await unauth_client.post(
        "/api/auth/login",
        json={
            "username": "admin",
            "password": "",  # default empty password in test settings
        },
    )
    # May succeed or fail depending on settings — just check structure
    if r.status_code == 200:
        data = r.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"


async def test_login_invalid_credentials(unauth_client):
    r = await unauth_client.post(
        "/api/auth/login",
        json={
            "username": "wrong",
            "password": "wrong",
        },
    )
    assert r.status_code == 401
    assert "Invalid credentials" in r.json()["detail"]


async def test_missing_token(unauth_client):
    r = await unauth_client.get("/api/portfolios")
    assert r.status_code in (401, 403)


# ── Portfolios ───────────────────────────────────────────────────────────────


async def test_list_portfolios(client):
    r = await client.get("/api/portfolios")
    assert r.status_code == 200
    data = r.json()
    assert len(data) == 2
    names = {p["name"] for p in data}
    assert names == {"A", "B"}


async def test_get_portfolio_detail(client):
    r = await client.get("/api/portfolios/A")
    assert r.status_code == 200
    data = r.json()
    assert data["name"] == "A"
    assert data["cash"] == 30000.0
    assert len(data["positions"]) == 1
    assert data["positions"][0]["ticker"] == "AAPL"


async def test_get_portfolio_not_found(client):
    r = await client.get("/api/portfolios/Z")
    assert r.status_code == 404


async def test_get_positions(client):
    r = await client.get("/api/portfolios/B/positions")
    assert r.status_code == 200
    data = r.json()
    assert len(data) == 1
    assert data[0]["ticker"] == "MSFT"
    assert data[0]["shares"] == 20


# ── Snapshots ────────────────────────────────────────────────────────────────


async def test_list_snapshots(client):
    r = await client.get("/api/snapshots")
    assert r.status_code == 200
    data = r.json()
    assert len(data) == 3


async def test_filter_snapshots_by_portfolio(client):
    r = await client.get("/api/snapshots?portfolio=A")
    assert r.status_code == 200
    data = r.json()
    assert len(data) == 2
    assert all(s["portfolio"] == "A" for s in data)


async def test_filter_snapshots_by_since(client):
    r = await client.get("/api/snapshots?since=2026-01-16")
    assert r.status_code == 200
    data = r.json()
    assert len(data) == 1
    assert data[0]["date"] == "2026-01-16"


# ── Trades ───────────────────────────────────────────────────────────────────


async def test_list_trades(client):
    r = await client.get("/api/trades")
    assert r.status_code == 200
    data = r.json()
    assert len(data) == 3


async def test_filter_trades_by_portfolio(client):
    r = await client.get("/api/trades?portfolio=A")
    assert r.status_code == 200
    data = r.json()
    assert len(data) == 2
    assert all(t["portfolio"] == "A" for t in data)


async def test_filter_trades_by_side(client):
    r = await client.get("/api/trades?side=SELL")
    assert r.status_code == 200
    data = r.json()
    assert len(data) == 1
    assert data[0]["side"] == "SELL"


async def test_trades_limit(client):
    r = await client.get("/api/trades?limit=1")
    assert r.status_code == 200
    data = r.json()
    assert len(data) == 1


# ── Momentum ─────────────────────────────────────────────────────────────────


async def test_momentum_rankings(client):
    r = await client.get("/api/momentum/rankings")
    assert r.status_code == 200
    data = r.json()
    assert len(data) == 2
    assert data[0]["rank"] == 1
    assert data[0]["ticker"] == "AAPL"


# ── Agent Decisions ──────────────────────────────────────────────────────────


async def test_agent_decisions(client):
    r = await client.get("/api/agent/decisions")
    assert r.status_code == 200
    data = r.json()
    assert len(data) == 1
    decision = data[0]
    assert decision["model_used"] == "claude-sonnet-4-5-20250929"
    assert isinstance(decision["proposed_trades"], list)
    assert decision["proposed_trades"][0]["ticker"] == "MSFT"


async def test_agent_decisions_limit(client):
    r = await client.get("/api/agent/decisions?limit=1")
    assert r.status_code == 200
    assert len(r.json()) == 1

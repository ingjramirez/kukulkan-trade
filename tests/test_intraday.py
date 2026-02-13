"""Tests for the intraday snapshot collector."""

from unittest.mock import MagicMock, patch

import pytest

from src.intraday import _enabled_portfolios, collect_intraday_snapshot
from src.storage.database import Database
from src.storage.models import TenantRow


@pytest.fixture
async def db():
    test_db = Database(url="sqlite+aiosqlite:///:memory:")
    await test_db.init_db()
    yield test_db
    await test_db.close()


def _make_tenant(
    tenant_id: str = "t-1",
    run_a: bool = True,
    run_b: bool = True,
) -> TenantRow:
    """Create a minimal TenantRow for testing."""
    return TenantRow(
        id=tenant_id,
        name="Test Tenant",
        alpaca_api_key_enc="enc-key",
        alpaca_api_secret_enc="enc-secret",
        telegram_bot_token_enc="enc-token",
        telegram_chat_id_enc="enc-chat",
        run_portfolio_a=run_a,
        run_portfolio_b=run_b,
    )


def _mock_alpaca_position(symbol: str, current_price: float):
    """Create a mock Alpaca position."""
    pos = MagicMock()
    pos.symbol = symbol
    pos.current_price = str(current_price)
    return pos


# ── _enabled_portfolios ─────────────────────────────────────────


def test_enabled_portfolios_both():
    tenant = _make_tenant(run_a=True, run_b=True)
    assert _enabled_portfolios(tenant) == ["A", "B"]


def test_enabled_portfolios_a_only():
    tenant = _make_tenant(run_a=True, run_b=False)
    assert _enabled_portfolios(tenant) == ["A"]


def test_enabled_portfolios_b_only():
    tenant = _make_tenant(run_a=False, run_b=True)
    assert _enabled_portfolios(tenant) == ["B"]


def test_enabled_portfolios_none():
    tenant = _make_tenant(run_a=False, run_b=False)
    assert _enabled_portfolios(tenant) == []


# ── collect_intraday_snapshot ─────────────────────────────────────


async def test_collect_intraday_snapshot_basic(db: Database):
    """Test basic snapshot collection with one portfolio and one position."""
    tenant = _make_tenant(run_a=False, run_b=True)

    # Seed portfolio and position
    await db.upsert_portfolio("B", cash=60000.0, total_value=68000.0, tenant_id="t-1")
    await db.upsert_position("B", "AAPL", shares=50, avg_price=160.0, tenant_id="t-1")

    mock_positions = [_mock_alpaca_position("AAPL", 170.0)]

    with patch("src.execution.client_factory.AlpacaClientFactory") as mock_factory:
        mock_client = MagicMock()
        mock_client.get_all_positions.return_value = mock_positions
        mock_factory.get_trading_client.return_value = mock_client

        saved = await collect_intraday_snapshot(db, tenant)

    assert saved == 1
    rows = await db.get_intraday_snapshots("t-1", portfolio="B")
    assert len(rows) == 1
    assert rows[0].positions_value == 50 * 170.0  # 8500
    assert rows[0].cash == 60000.0
    assert rows[0].total_value == 60000.0 + 50 * 170.0


async def test_collect_intraday_snapshot_both_portfolios(db: Database):
    """Test snapshot collection for both portfolios."""
    tenant = _make_tenant(run_a=True, run_b=True)

    await db.upsert_portfolio("A", cash=30000.0, total_value=35000.0, tenant_id="t-1")
    await db.upsert_portfolio("B", cash=60000.0, total_value=68000.0, tenant_id="t-1")
    await db.upsert_position("A", "MSFT", shares=10, avg_price=400.0, tenant_id="t-1")
    await db.upsert_position("B", "AAPL", shares=50, avg_price=160.0, tenant_id="t-1")

    mock_positions = [
        _mock_alpaca_position("MSFT", 410.0),
        _mock_alpaca_position("AAPL", 170.0),
    ]

    with patch("src.execution.client_factory.AlpacaClientFactory") as mock_factory:
        mock_client = MagicMock()
        mock_client.get_all_positions.return_value = mock_positions
        mock_factory.get_trading_client.return_value = mock_client

        saved = await collect_intraday_snapshot(db, tenant)

    assert saved == 2
    all_rows = await db.get_intraday_snapshots("t-1")
    assert len(all_rows) == 2


async def test_collect_intraday_snapshot_no_portfolios(db: Database):
    """Test with both portfolios disabled — should save nothing."""
    tenant = _make_tenant(run_a=False, run_b=False)

    with patch("src.execution.client_factory.AlpacaClientFactory") as mock_factory:
        mock_client = MagicMock()
        mock_client.get_all_positions.return_value = []
        mock_factory.get_trading_client.return_value = mock_client

        saved = await collect_intraday_snapshot(db, tenant)

    assert saved == 0


async def test_collect_intraday_snapshot_alpaca_failure(db: Database):
    """Test graceful handling of Alpaca API failure."""
    tenant = _make_tenant(run_a=True, run_b=True)

    with patch("src.execution.client_factory.AlpacaClientFactory") as mock_factory:
        mock_client = MagicMock()
        mock_client.get_all_positions.side_effect = Exception("Connection refused")
        mock_factory.get_trading_client.return_value = mock_client

        saved = await collect_intraday_snapshot(db, tenant)

    assert saved == 0


async def test_collect_intraday_snapshot_missing_portfolio_row(db: Database):
    """Test when portfolio row doesn't exist in DB (no crash)."""
    tenant = _make_tenant(run_a=False, run_b=True)
    # Don't seed portfolio B — it doesn't exist in DB

    with patch("src.execution.client_factory.AlpacaClientFactory") as mock_factory:
        mock_client = MagicMock()
        mock_client.get_all_positions.return_value = []
        mock_factory.get_trading_client.return_value = mock_client

        saved = await collect_intraday_snapshot(db, tenant)

    assert saved == 0


async def test_collect_intraday_snapshot_uses_avg_price_fallback(db: Database):
    """Test that positions without live prices fall back to avg_price."""
    tenant = _make_tenant(run_a=False, run_b=True)

    await db.upsert_portfolio("B", cash=60000.0, total_value=68000.0, tenant_id="t-1")
    await db.upsert_position("B", "AAPL", shares=50, avg_price=160.0, tenant_id="t-1")
    await db.upsert_position("B", "XYZ", shares=100, avg_price=10.0, tenant_id="t-1")

    # Only AAPL has a live price, XYZ will use avg_price
    mock_positions = [_mock_alpaca_position("AAPL", 170.0)]

    with patch("src.execution.client_factory.AlpacaClientFactory") as mock_factory:
        mock_client = MagicMock()
        mock_client.get_all_positions.return_value = mock_positions
        mock_factory.get_trading_client.return_value = mock_client

        saved = await collect_intraday_snapshot(db, tenant)

    assert saved == 1
    rows = await db.get_intraday_snapshots("t-1", portfolio="B")
    # AAPL: 50 * 170 = 8500, XYZ: 100 * 10 = 1000 (avg_price fallback)
    assert rows[0].positions_value == pytest.approx(9500.0)
    assert rows[0].total_value == pytest.approx(69500.0)

"""Tests for deposit detection in the orchestrator."""

from unittest.mock import MagicMock

import pytest

from src.orchestrator import Orchestrator
from src.storage.database import Database
from src.utils.allocations import DEPOSIT_THRESHOLD, TenantAllocations, resolve_allocations


@pytest.fixture
async def db():
    database = Database(url="sqlite+aiosqlite:///:memory:")
    await database.init_db()
    yield database
    await database.close()


def _make_allocations(
    equity: float = 100_000.0,
    a_pct: float = 33.33,
    b_pct: float = 66.67,
) -> TenantAllocations:
    return resolve_allocations(
        initial_equity=equity,
        portfolio_a_pct=a_pct,
        portfolio_b_pct=b_pct,
    )


class TestDetectDeposits:
    async def test_deposit_detected(self, db: Database) -> None:
        """Deposit detected when broker equity exceeds tracked totals."""
        orchestrator = Orchestrator(db)
        # Initialize portfolios with known values
        await db.upsert_portfolio("A", cash=33_330.0, total_value=33_330.0)
        await db.upsert_portfolio("B", cash=66_670.0, total_value=66_670.0)

        alloc = _make_allocations(equity=100_000.0)

        # Mock executor with Alpaca client that reports higher equity
        mock_client = MagicMock()
        mock_account = MagicMock()
        mock_account.equity = "101_000.0"  # $1000 deposit
        mock_client.get_account.return_value = mock_account
        orchestrator._executor = MagicMock()
        orchestrator._executor._client = mock_client

        new_alloc = await orchestrator._detect_deposits(alloc, "default")

        # Verify equity increased
        assert new_alloc.initial_equity == 101_000.0
        assert new_alloc.portfolio_a_cash == pytest.approx(
            101_000.0 * 33.33 / 100, rel=1e-2,
        )
        assert new_alloc.portfolio_b_cash == pytest.approx(
            101_000.0 * 66.67 / 100, rel=1e-2,
        )

        # Verify portfolio cash was increased in DB
        port_a = await db.get_portfolio("A")
        assert port_a.cash > 33_330.0

    async def test_no_deposit_small_delta(self, db: Database) -> None:
        """No deposit if delta is below threshold."""
        orchestrator = Orchestrator(db)
        await db.upsert_portfolio("A", cash=33_330.0, total_value=33_330.0)
        await db.upsert_portfolio("B", cash=66_670.0, total_value=66_670.0)

        alloc = _make_allocations(equity=100_000.0)

        mock_client = MagicMock()
        mock_account = MagicMock()
        # Delta of $10 — below threshold
        mock_account.equity = "100_010.0"
        mock_client.get_account.return_value = mock_account
        orchestrator._executor = MagicMock()
        orchestrator._executor._client = mock_client

        new_alloc = await orchestrator._detect_deposits(alloc, "default")

        # Allocations unchanged
        assert new_alloc.initial_equity == alloc.initial_equity

    async def test_negative_delta_ignored(self, db: Database) -> None:
        """Negative delta (portfolio loss) is not treated as deposit."""
        orchestrator = Orchestrator(db)
        await db.upsert_portfolio("A", cash=33_330.0, total_value=33_330.0)
        await db.upsert_portfolio("B", cash=66_670.0, total_value=66_670.0)

        alloc = _make_allocations(equity=100_000.0)

        mock_client = MagicMock()
        mock_account = MagicMock()
        mock_account.equity = "95_000.0"  # Loss, not deposit
        mock_client.get_account.return_value = mock_account
        orchestrator._executor = MagicMock()
        orchestrator._executor._client = mock_client

        new_alloc = await orchestrator._detect_deposits(alloc, "default")
        assert new_alloc.initial_equity == alloc.initial_equity

    async def test_no_executor_client(self, db: Database) -> None:
        """Graceful fallback when executor has no Alpaca client."""
        orchestrator = Orchestrator(db)
        alloc = _make_allocations()

        # PaperTrader has no _client attribute
        new_alloc = await orchestrator._detect_deposits(alloc, "default")
        assert new_alloc is alloc  # Unchanged

    async def test_alpaca_fetch_failure(self, db: Database) -> None:
        """Graceful fallback when Alpaca equity fetch fails."""
        orchestrator = Orchestrator(db)
        alloc = _make_allocations()

        mock_client = MagicMock()
        mock_client.get_account.side_effect = Exception("Connection error")
        orchestrator._executor = MagicMock()
        orchestrator._executor._client = mock_client

        new_alloc = await orchestrator._detect_deposits(alloc, "default")
        assert new_alloc is alloc  # Unchanged

    async def test_deposit_updates_tenant_record(self, db: Database) -> None:
        """Deposit detection updates the tenant DB record."""
        from cryptography.fernet import Fernet

        from config.settings import settings
        settings.tenant_encryption_key = Fernet.generate_key().decode()
        from src.storage.models import TenantRow
        from src.utils.crypto import encrypt_value

        tenant = TenantRow(
            id="t1",
            name="Test",
            alpaca_api_key_enc=encrypt_value("KEY"),
            alpaca_api_secret_enc=encrypt_value("SECRET"),
            telegram_bot_token_enc=encrypt_value("TOKEN"),
            telegram_chat_id_enc=encrypt_value("123"),
            initial_equity=100_000.0,
            portfolio_a_pct=33.33,
            portfolio_b_pct=66.67,
            portfolio_a_cash=33_330.0,
            portfolio_b_cash=66_670.0,
        )
        await db.create_tenant(tenant)
        await db.upsert_portfolio("A", cash=33_330.0, total_value=33_330.0, tenant_id="t1")
        await db.upsert_portfolio("B", cash=66_670.0, total_value=66_670.0, tenant_id="t1")

        orchestrator = Orchestrator(db)
        alloc = _make_allocations(equity=100_000.0)

        mock_client = MagicMock()
        mock_account = MagicMock()
        mock_account.equity = "105_000.0"  # $5000 deposit
        mock_client.get_account.return_value = mock_account
        orchestrator._executor = MagicMock()
        orchestrator._executor._client = mock_client

        await orchestrator._detect_deposits(alloc, "t1")

        # Verify tenant record was updated
        updated_tenant = await db.get_tenant("t1")
        assert updated_tenant.initial_equity == 105_000.0


class TestCaptureAlpacaEquity:
    async def test_capture_success(self) -> None:
        mock_executor = MagicMock()
        mock_account = MagicMock()
        mock_account.equity = "100500.50"
        mock_executor._client.get_account.return_value = mock_account

        equity = await Orchestrator._capture_alpaca_equity(mock_executor)
        assert equity == 100_500.50

    async def test_capture_no_client(self) -> None:
        """Non-Alpaca executor returns None."""
        mock_executor = MagicMock(spec=[])  # No _client attribute
        equity = await Orchestrator._capture_alpaca_equity(mock_executor)
        assert equity is None

    async def test_capture_failure(self) -> None:
        mock_executor = MagicMock()
        mock_executor._client.get_account.side_effect = Exception("API error")

        equity = await Orchestrator._capture_alpaca_equity(mock_executor)
        assert equity is None


class TestDepositThreshold:
    def test_threshold_value(self) -> None:
        assert DEPOSIT_THRESHOLD == 50.0

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


def _mock_executor_with_activities(
    activities: list[dict],
    equity: str = "101_000.0",
) -> MagicMock:
    """Create a mock executor that returns CSD activities and account equity."""
    mock_client = MagicMock()
    mock_client.get.return_value = activities
    mock_account = MagicMock()
    mock_account.equity = equity
    mock_client.get_account.return_value = mock_account
    mock_executor = MagicMock()
    mock_executor._client = mock_client
    return mock_executor


class TestDetectDeposits:
    async def test_deposit_detected_via_activities(self, db: Database) -> None:
        """Real CSD activity triggers deposit detection."""
        orchestrator = Orchestrator(db)
        await db.upsert_portfolio("A", cash=33_330.0, total_value=33_330.0)
        await db.upsert_portfolio("B", cash=66_670.0, total_value=66_670.0)

        alloc = _make_allocations(equity=100_000.0)

        # CSD activity for $1000, broker equity reflects the deposit + tracked
        orchestrator._executor = _mock_executor_with_activities(
            activities=[{"net_amount": "1000.00", "activity_type": "CSD"}],
            equity="101_000.0",
        )

        new_alloc = await orchestrator._detect_deposits(alloc, "default")

        assert new_alloc.initial_equity == 101_000.0
        assert new_alloc.portfolio_a_cash == pytest.approx(
            101_000.0 * 33.33 / 100,
            rel=1e-2,
        )
        assert new_alloc.portfolio_b_cash == pytest.approx(
            101_000.0 * 66.67 / 100,
            rel=1e-2,
        )

        # Verify portfolio cash was increased in DB
        port_a = await db.get_portfolio("A")
        assert port_a.cash > 33_330.0

    async def test_no_activities_no_deposit(self, db: Database) -> None:
        """No CSD activities → no deposit, even if equity delta is large."""
        orchestrator = Orchestrator(db)
        await db.upsert_portfolio("A", cash=33_330.0, total_value=33_330.0)
        await db.upsert_portfolio("B", cash=66_670.0, total_value=66_670.0)

        alloc = _make_allocations(equity=100_000.0)

        # No activities but broker equity is $395 higher (price movement)
        orchestrator._executor = _mock_executor_with_activities(
            activities=[],
            equity="100_395.22",
        )

        new_alloc = await orchestrator._detect_deposits(alloc, "default")

        # Must NOT detect a deposit — this was the original bug
        assert new_alloc.initial_equity == alloc.initial_equity

    async def test_empty_response_no_deposit(self, db: Database) -> None:
        """Empty/None API response is handled gracefully."""
        orchestrator = Orchestrator(db)
        alloc = _make_allocations()

        mock_client = MagicMock()
        mock_client.get.return_value = None
        orchestrator._executor = MagicMock()
        orchestrator._executor._client = mock_client

        new_alloc = await orchestrator._detect_deposits(alloc, "default")
        assert new_alloc is alloc

    async def test_small_deposit_below_threshold_ignored(self, db: Database) -> None:
        """CSD activity below threshold is ignored."""
        orchestrator = Orchestrator(db)
        await db.upsert_portfolio("A", cash=33_330.0, total_value=33_330.0)
        await db.upsert_portfolio("B", cash=66_670.0, total_value=66_670.0)

        alloc = _make_allocations(equity=100_000.0)

        orchestrator._executor = _mock_executor_with_activities(
            activities=[{"net_amount": "10.00", "activity_type": "CSD"}],
            equity="100_010.0",
        )

        new_alloc = await orchestrator._detect_deposits(alloc, "default")
        assert new_alloc.initial_equity == alloc.initial_equity

    async def test_negative_activity_amount_ignored(self, db: Database) -> None:
        """Withdrawal activities (negative amounts) are not counted."""
        orchestrator = Orchestrator(db)
        await db.upsert_portfolio("A", cash=33_330.0, total_value=33_330.0)
        await db.upsert_portfolio("B", cash=66_670.0, total_value=66_670.0)

        alloc = _make_allocations(equity=100_000.0)

        orchestrator._executor = _mock_executor_with_activities(
            activities=[{"net_amount": "-500.00", "activity_type": "CSW"}],
            equity="99_500.0",
        )

        new_alloc = await orchestrator._detect_deposits(alloc, "default")
        assert new_alloc.initial_equity == alloc.initial_equity

    async def test_no_executor_client(self, db: Database) -> None:
        """Graceful fallback when executor has no Alpaca client."""
        orchestrator = Orchestrator(db)
        alloc = _make_allocations()

        # PaperTrader has no _client attribute
        new_alloc = await orchestrator._detect_deposits(alloc, "default")
        assert new_alloc is alloc

    async def test_activities_fetch_failure(self, db: Database) -> None:
        """Graceful fallback when activities API call fails."""
        orchestrator = Orchestrator(db)
        alloc = _make_allocations()

        mock_client = MagicMock()
        mock_client.get.side_effect = Exception("Connection error")
        orchestrator._executor = MagicMock()
        orchestrator._executor._client = mock_client

        new_alloc = await orchestrator._detect_deposits(alloc, "default")
        assert new_alloc is alloc

    async def test_equity_fetch_failure_after_activities(self, db: Database) -> None:
        """Graceful fallback when equity fetch fails after activities found."""
        orchestrator = Orchestrator(db)
        alloc = _make_allocations()

        mock_client = MagicMock()
        mock_client.get.return_value = [
            {"net_amount": "500.00", "activity_type": "CSD"},
        ]
        mock_client.get_account.side_effect = Exception("API error")
        orchestrator._executor = MagicMock()
        orchestrator._executor._client = mock_client

        new_alloc = await orchestrator._detect_deposits(alloc, "default")
        assert new_alloc is alloc

    async def test_equity_gap_prevents_double_counting(self, db: Database) -> None:
        """If tracked totals already reflect deposit, skip processing."""
        orchestrator = Orchestrator(db)
        # Tracked totals already include the deposit (e.g. from a prior run)
        await db.upsert_portfolio("A", cash=33_663.0, total_value=33_663.0)
        await db.upsert_portfolio("B", cash=67_337.0, total_value=67_337.0)
        # tracked = 101,000

        alloc = _make_allocations(equity=101_000.0)

        # Activities show a $1000 deposit, but equity ≈ tracked → already processed
        orchestrator._executor = _mock_executor_with_activities(
            activities=[{"net_amount": "1000.00", "activity_type": "CSD"}],
            equity="101_000.0",  # matches tracked
        )

        new_alloc = await orchestrator._detect_deposits(alloc, "default")
        # Should NOT re-process — equity gap ≈ 0
        assert new_alloc.initial_equity == alloc.initial_equity

    async def test_multiple_activities_summed(self, db: Database) -> None:
        """Multiple CSD activities in the window are summed."""
        orchestrator = Orchestrator(db)
        await db.upsert_portfolio("A", cash=33_330.0, total_value=33_330.0)
        await db.upsert_portfolio("B", cash=66_670.0, total_value=66_670.0)

        alloc = _make_allocations(equity=100_000.0)

        orchestrator._executor = _mock_executor_with_activities(
            activities=[
                {"net_amount": "300.00", "activity_type": "CSD"},
                {"net_amount": "200.00", "activity_type": "JNLC"},
            ],
            equity="100_500.0",
        )

        new_alloc = await orchestrator._detect_deposits(alloc, "default")

        # $300 + $200 = $500 deposit
        assert new_alloc.initial_equity == 100_500.0

    async def test_single_dict_response_handled(self, db: Database) -> None:
        """API returning a single dict (instead of list) is handled."""
        orchestrator = Orchestrator(db)
        await db.upsert_portfolio("A", cash=33_330.0, total_value=33_330.0)
        await db.upsert_portfolio("B", cash=66_670.0, total_value=66_670.0)

        alloc = _make_allocations(equity=100_000.0)

        mock_client = MagicMock()
        # Single dict instead of list
        mock_client.get.return_value = {
            "net_amount": "500.00",
            "activity_type": "CSD",
        }
        mock_account = MagicMock()
        mock_account.equity = "100_500.0"
        mock_client.get_account.return_value = mock_account
        orchestrator._executor = MagicMock()
        orchestrator._executor._client = mock_client

        new_alloc = await orchestrator._detect_deposits(alloc, "default")
        assert new_alloc.initial_equity == 100_500.0

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

        orchestrator._executor = _mock_executor_with_activities(
            activities=[{"net_amount": "5000.00", "activity_type": "CSD"}],
            equity="105_000.0",
        )

        await orchestrator._detect_deposits(alloc, "t1")

        updated_tenant = await db.get_tenant("t1")
        assert updated_tenant.initial_equity == 105_000.0

    async def test_malformed_activity_skipped(self, db: Database) -> None:
        """Activities with bad net_amount are skipped gracefully."""
        orchestrator = Orchestrator(db)
        await db.upsert_portfolio("A", cash=33_330.0, total_value=33_330.0)
        await db.upsert_portfolio("B", cash=66_670.0, total_value=66_670.0)

        alloc = _make_allocations(equity=100_000.0)

        orchestrator._executor = _mock_executor_with_activities(
            activities=[
                {"net_amount": "not_a_number", "activity_type": "CSD"},
                {"net_amount": "500.00", "activity_type": "CSD"},
            ],
            equity="100_500.0",
        )

        new_alloc = await orchestrator._detect_deposits(alloc, "default")
        # Only the valid $500 is counted
        assert new_alloc.initial_equity == 100_500.0

    async def test_price_movement_with_no_activities(self, db: Database) -> None:
        """Reproduces the original bug: overnight price appreciation.

        Positions appreciated $395 overnight. Old code would falsely detect
        a deposit. New code queries activities and finds none → no deposit.
        """
        orchestrator = Orchestrator(db)
        # Tracked from last night's snapshot
        await db.upsert_portfolio("A", cash=5_000.0, total_value=33_000.0)
        await db.upsert_portfolio("B", cash=50_000.0, total_value=66_605.0)
        # tracked_total = $99,605

        alloc = _make_allocations(equity=100_000.0)

        # Broker reports $100,000 (positions worth more now)
        orchestrator._executor = _mock_executor_with_activities(
            activities=[],  # No actual deposit
            equity="100_000.0",  # $395 higher than tracked
        )

        new_alloc = await orchestrator._detect_deposits(alloc, "default")

        # MUST NOT inflate the baseline
        assert new_alloc.initial_equity == 100_000.0
        assert new_alloc is alloc


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

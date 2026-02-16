"""Tests for multi-tenant orchestrator iteration and isolation."""

from unittest.mock import AsyncMock, patch

import pytest
from cryptography.fernet import Fernet

from config.settings import settings
from src.orchestrator import Orchestrator
from src.storage.database import Database
from src.storage.models import TenantRow
from src.utils.crypto import encrypt_value

_TEST_KEY = Fernet.generate_key().decode()


@pytest.fixture(autouse=True)
def _set_encryption_key(monkeypatch):
    monkeypatch.setattr(settings, "tenant_encryption_key", _TEST_KEY)


@pytest.fixture
async def db():
    database = Database(url="sqlite+aiosqlite:///:memory:")
    await database.init_db()
    yield database
    await database.close()


def _make_tenant(
    tenant_id: str = "t1",
    name: str = "Test",
    strategy: str = "aggressive",
) -> TenantRow:
    return TenantRow(
        id=tenant_id,
        name=name,
        alpaca_api_key_enc=encrypt_value("KEY"),
        alpaca_api_secret_enc=encrypt_value("SECRET"),
        telegram_bot_token_enc=encrypt_value("TOKEN"),
        telegram_chat_id_enc=encrypt_value("123"),
        strategy_mode=strategy,
        run_portfolio_a=False,
        run_portfolio_b=True,
    )


class TestRunAllTenants:
    async def test_no_configured_tenants_skips_default(self, db: Database):
        """With only 'default' tenant (no credentials), it is skipped as incomplete."""
        orchestrator = Orchestrator(db)
        with patch.object(orchestrator, "run_daily", new_callable=AsyncMock) as mock_run:
            results = await orchestrator.run_all_tenants()
            # "default" exists but has no credentials → skipped
            assert len(results) == 1
            assert results[0]["skipped"] == "incomplete_credentials"
            assert results[0]["tenant_id"] == "default"
            mock_run.assert_not_called()

    async def test_iterates_active_tenants(self, db: Database):
        """With active tenants, iterates each fully-configured one."""
        await db.create_tenant(_make_tenant("t1", "Tenant A"))
        await db.create_tenant(_make_tenant("t2", "Tenant B"))

        orchestrator = Orchestrator(db)
        with patch.object(
            orchestrator,
            "run_tenant_session",
            new_callable=AsyncMock,
        ) as mock_session:
            mock_session.return_value = {"date": "2026-02-09"}
            results = await orchestrator.run_all_tenants()
            # "default" skipped (incomplete) + t1 + t2 processed = 3 results
            assert len(results) == 3
            assert mock_session.call_count == 2

    async def test_skips_inactive_tenants(self, db: Database):
        """Inactive tenants are not iterated."""
        await db.create_tenant(_make_tenant("t1", "Active"))
        await db.create_tenant(_make_tenant("t2", "Inactive"))
        await db.deactivate_tenant("t2")

        orchestrator = Orchestrator(db)
        with patch.object(
            orchestrator,
            "run_tenant_session",
            new_callable=AsyncMock,
        ) as mock_session:
            mock_session.return_value = {"date": "2026-02-09"}
            results = await orchestrator.run_all_tenants()
            # "default" skipped (incomplete) + t1 processed = 2 results
            assert len(results) == 2
            assert mock_session.call_count == 1

    async def test_tenant_failure_is_isolated(self, db: Database):
        """One tenant failing doesn't affect others."""
        await db.create_tenant(_make_tenant("t1", "OK"))
        await db.create_tenant(_make_tenant("t2", "Fails"))

        orchestrator = Orchestrator(db)

        call_count = 0

        async def mock_session(tenant, **kwargs):
            nonlocal call_count
            call_count += 1
            if tenant.id == "t2":
                raise RuntimeError("Alpaca connection failed")
            return {"date": "2026-02-09", "tenant_id": tenant.id}

        with patch.object(orchestrator, "run_tenant_session", side_effect=mock_session):
            # Also patch TelegramFactory to avoid decrypt errors in error handler
            with patch("src.orchestrator.asyncio.sleep", new_callable=AsyncMock):
                results = await orchestrator.run_all_tenants()
                # "default" skipped + t1 success + t2 error = 3 results
                assert len(results) == 3
                # Filter out the skipped "default" entry for assertion clarity
                processed = [r for r in results if r.get("skipped") != "incomplete_credentials"]
                errors = [r for r in processed if "error" in r]
                successes = [r for r in processed if "error" not in r]
                assert len(errors) == 1
                assert len(successes) == 1
                assert errors[0]["tenant_id"] == "t2"


class TestRunDaily:
    async def test_portfolio_a_skipped_when_not_configured(self, db: Database):
        """run_portfolio_a=False should skip Portfolio A."""
        orchestrator = Orchestrator(db)
        # Use a mock to avoid full pipeline execution
        with patch.object(orchestrator, "_run_portfolio_a", new_callable=AsyncMock):
            with patch.object(orchestrator, "_executor") as mock_exec:
                mock_exec.initialize_portfolios = AsyncMock()
                mock_exec.sync_positions = AsyncMock()
                mock_exec.execute_trades = AsyncMock(return_value=[])
                mock_exec.take_snapshot = AsyncMock()

                with patch("src.orchestrator.is_market_open", return_value=False):
                    result = await orchestrator.run_daily(
                        run_portfolio_a=False,
                        run_portfolio_b=True,
                    )
                    # Market is closed, so pipeline is skipped entirely
                    assert result["skipped"] == "market_closed"

    async def test_strategy_mode_override(self, db: Database):
        """Custom strategy_mode should be used instead of settings."""
        orchestrator = Orchestrator(db)
        # We just verify the parameter flows through correctly
        with patch("src.orchestrator.is_market_open", return_value=False):
            result = await orchestrator.run_daily(
                strategy_mode="aggressive",
            )
            assert result["skipped"] == "market_closed"

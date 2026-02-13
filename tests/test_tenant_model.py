"""Tests for tenant ORM model and database CRUD."""

import pytest
from cryptography.fernet import Fernet

from config.settings import settings
from src.storage.database import Database
from src.storage.models import TenantRow
from src.utils.crypto import encrypt_value

_TEST_KEY = Fernet.generate_key().decode()


@pytest.fixture(autouse=True)
def _set_encryption_key(monkeypatch):
    monkeypatch.setattr(settings, "tenant_encryption_key", _TEST_KEY)


@pytest.fixture
async def db():
    """In-memory database with tables created."""
    database = Database(url="sqlite+aiosqlite:///:memory:")
    await database.init_db()
    yield database
    await database.close()


def _make_tenant(
    tenant_id: str = "t1",
    name: str = "Test Tenant",
    strategy: str = "conservative",
) -> TenantRow:
    return TenantRow(
        id=tenant_id,
        name=name,
        alpaca_api_key_enc=encrypt_value("APCA-KEY"),
        alpaca_api_secret_enc=encrypt_value("APCA-SECRET"),
        telegram_bot_token_enc=encrypt_value("BOT-TOKEN"),
        telegram_chat_id_enc=encrypt_value("12345"),
        strategy_mode=strategy,
    )


class TestTenantCRUD:
    async def test_create_and_get(self, db: Database):
        tenant = _make_tenant()
        await db.create_tenant(tenant)

        fetched = await db.get_tenant("t1")
        assert fetched is not None
        assert fetched.name == "Test Tenant"
        assert fetched.strategy_mode == "conservative"
        assert fetched.is_active is True

    async def test_get_nonexistent(self, db: Database):
        assert await db.get_tenant("nope") is None

    async def test_get_active_tenants(self, db: Database):
        await db.create_tenant(_make_tenant("t1", "Active"))
        await db.create_tenant(_make_tenant("t2", "Inactive"))
        await db.deactivate_tenant("t2")

        active = await db.get_active_tenants()
        assert len(active) == 1
        assert active[0].id == "t1"

    async def test_get_all_tenants(self, db: Database):
        await db.create_tenant(_make_tenant("t1", "A"))
        await db.create_tenant(_make_tenant("t2", "B"))
        await db.deactivate_tenant("t2")

        all_tenants = await db.get_all_tenants()
        assert len(all_tenants) == 2

    async def test_update_tenant(self, db: Database):
        await db.create_tenant(_make_tenant())
        updated = await db.update_tenant("t1", {"strategy_mode": "aggressive"})
        assert updated.strategy_mode == "aggressive"

    async def test_update_nonexistent(self, db: Database):
        result = await db.update_tenant("nope", {"name": "x"})
        assert result is None

    async def test_deactivate_tenant(self, db: Database):
        await db.create_tenant(_make_tenant())
        assert await db.deactivate_tenant("t1") is True

        tenant = await db.get_tenant("t1")
        assert tenant.is_active is False

    async def test_deactivate_nonexistent(self, db: Database):
        assert await db.deactivate_tenant("nope") is False


class TestTenantRepr:
    def test_repr_masks_credentials(self):
        tenant = _make_tenant()
        r = repr(tenant)
        assert "APCA" not in r
        assert "BOT-TOKEN" not in r
        assert "Test Tenant" in r


class TestTenantIsolation:
    """Verify that tenant_id isolates portfolio data."""

    async def test_separate_portfolios(self, db: Database):
        await db.upsert_portfolio("A", cash=1000, total_value=1000, tenant_id="t1")
        await db.upsert_portfolio("A", cash=2000, total_value=2000, tenant_id="t2")

        p1 = await db.get_portfolio("A", tenant_id="t1")
        p2 = await db.get_portfolio("A", tenant_id="t2")

        assert p1.cash == 1000
        assert p2.cash == 2000

    async def test_separate_positions(self, db: Database):
        await db.upsert_position("B", "AAPL", 10, 150.0, tenant_id="t1")
        await db.upsert_position("B", "AAPL", 20, 160.0, tenant_id="t2")

        p1 = await db.get_positions("B", tenant_id="t1")
        p2 = await db.get_positions("B", tenant_id="t2")

        assert len(p1) == 1 and p1[0].shares == 10
        assert len(p2) == 1 and p2[0].shares == 20

    async def test_separate_trades(self, db: Database):
        await db.log_trade("B", "AAPL", "BUY", 10, 150.0, tenant_id="t1")
        await db.log_trade("B", "MSFT", "BUY", 5, 300.0, tenant_id="t2")

        t1 = await db.get_trades("B", tenant_id="t1")
        t2 = await db.get_trades("B", tenant_id="t2")

        assert len(t1) == 1 and t1[0].ticker == "AAPL"
        assert len(t2) == 1 and t2[0].ticker == "MSFT"

    async def test_separate_snapshots(self, db: Database):
        from datetime import date

        today = date(2026, 2, 9)
        await db.save_snapshot("B", today, 66000, 60000, 6000, tenant_id="t1")
        await db.save_snapshot("B", today, 77000, 70000, 7000, tenant_id="t2")

        s1 = await db.get_snapshots("B", tenant_id="t1")
        s2 = await db.get_snapshots("B", tenant_id="t2")

        assert len(s1) == 1 and s1[0].total_value == 66000
        assert len(s2) == 1 and s2[0].total_value == 77000

    async def test_separate_agent_memory(self, db: Database):
        await db.upsert_agent_memory(
            "short_term",
            "key1",
            "content-t1",
            tenant_id="t1",
        )
        await db.upsert_agent_memory(
            "short_term",
            "key1",
            "content-t2",
            tenant_id="t2",
        )

        m1 = await db.get_agent_memories("short_term", tenant_id="t1")
        m2 = await db.get_agent_memories("short_term", tenant_id="t2")

        assert len(m1) == 1 and m1[0].content == "content-t1"
        assert len(m2) == 1 and m2[0].content == "content-t2"

    async def test_default_tenant_backward_compat(self, db: Database):
        """Without tenant_id, everything uses 'default'."""
        await db.upsert_portfolio("A", cash=5000, total_value=5000)
        p = await db.get_portfolio("A")
        assert p is not None
        assert p.tenant_id == "default"

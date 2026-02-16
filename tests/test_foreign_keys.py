"""Tests for foreign key constraints on tenant_id columns."""

from datetime import datetime, timezone

import pytest
from sqlalchemy.exc import IntegrityError

from src.storage.database import Database
from src.storage.models import PortfolioRow, PositionRow, TenantRow, TradeRow


@pytest.fixture
async def db():
    """Create an in-memory test database with FK enforcement."""
    database = Database(url="sqlite+aiosqlite:///:memory:")
    await database.init_db()
    yield database
    await database.close()


class TestForeignKeyEnforcement:
    """Verify FK constraints reject invalid tenant_id values."""

    async def test_insert_with_nonexistent_tenant_raises(self, db: Database) -> None:
        """Inserting a row with a tenant_id that doesn't exist should fail."""
        async with db.session() as s:
            s.add(
                PortfolioRow(
                    tenant_id="nonexistent",
                    name="A",
                    cash=10000.0,
                    total_value=10000.0,
                    updated_at=datetime.now(timezone.utc),
                )
            )
            with pytest.raises(IntegrityError):
                await s.commit()

    async def test_insert_with_valid_tenant_succeeds(self, db: Database) -> None:
        """Inserting with 'default' tenant_id should succeed (seeded by init_db)."""
        await db.upsert_portfolio("A", cash=33000.0, total_value=33000.0)
        portfolio = await db.get_portfolio("A")
        assert portfolio is not None
        assert portfolio.tenant_id == "default"

    async def test_insert_with_custom_tenant_succeeds(self, db: Database) -> None:
        """Inserting with a real tenant should succeed."""
        async with db.session() as s:
            s.add(TenantRow(id="tenant-1", name="Tenant One"))
            await s.commit()

        async with db.session() as s:
            s.add(
                PortfolioRow(
                    tenant_id="tenant-1",
                    name="B",
                    cash=50000.0,
                    total_value=50000.0,
                    updated_at=datetime.now(timezone.utc),
                )
            )
            await s.commit()

        async with db.session() as s:
            from sqlalchemy import select

            result = await s.execute(select(PortfolioRow).where(PortfolioRow.tenant_id == "tenant-1"))
            row = result.scalar_one()
            assert row.name == "B"


class TestCascadeDelete:
    """Verify CASCADE delete propagation."""

    async def test_cascade_deletes_child_rows(self, db: Database) -> None:
        """Deleting a tenant should cascade-delete its child rows."""
        # Create a test tenant
        async with db.session() as s:
            s.add(TenantRow(id="doomed", name="Doomed Tenant"))
            await s.commit()

        # Create child rows
        async with db.session() as s:
            s.add(
                PortfolioRow(
                    tenant_id="doomed",
                    name="A",
                    cash=10000.0,
                    total_value=10000.0,
                    updated_at=datetime.now(timezone.utc),
                )
            )
            s.add(
                TradeRow(
                    tenant_id="doomed",
                    portfolio="A",
                    ticker="AAPL",
                    side="BUY",
                    shares=10.0,
                    price=150.0,
                    total=1500.0,
                    executed_at=datetime.now(timezone.utc),
                )
            )
            await s.commit()

        # Verify rows exist
        async with db.session() as s:
            from sqlalchemy import select

            portfolios = (
                (await s.execute(select(PortfolioRow).where(PortfolioRow.tenant_id == "doomed"))).scalars().all()
            )
            assert len(portfolios) == 1

            trades = (await s.execute(select(TradeRow).where(TradeRow.tenant_id == "doomed"))).scalars().all()
            assert len(trades) == 1

        # Delete the tenant
        async with db.session() as s:
            from sqlalchemy import delete

            await s.execute(delete(TenantRow).where(TenantRow.id == "doomed"))
            await s.commit()

        # Verify cascade — child rows should be gone
        async with db.session() as s:
            from sqlalchemy import select

            portfolios = (
                (await s.execute(select(PortfolioRow).where(PortfolioRow.tenant_id == "doomed"))).scalars().all()
            )
            assert len(portfolios) == 0

            trades = (await s.execute(select(TradeRow).where(TradeRow.tenant_id == "doomed"))).scalars().all()
            assert len(trades) == 0

    async def test_default_tenant_not_affected(self, db: Database) -> None:
        """Deleting another tenant should not affect 'default' tenant rows."""
        # Add a row for default tenant
        await db.upsert_portfolio("A", cash=33000.0, total_value=33000.0)

        # Create and delete another tenant
        async with db.session() as s:
            s.add(TenantRow(id="other", name="Other"))
            await s.commit()

        async with db.session() as s:
            from sqlalchemy import delete

            await s.execute(delete(TenantRow).where(TenantRow.id == "other"))
            await s.commit()

        # Default portfolio should still exist
        portfolio = await db.get_portfolio("A")
        assert portfolio is not None


class TestForeignKeyPragma:
    """Verify FK pragma is enabled on connections."""

    async def test_pragma_enabled(self, db: Database) -> None:
        """PRAGMA foreign_keys should be ON."""
        async with db.session() as s:
            result = await s.execute(__import__("sqlalchemy").text("PRAGMA foreign_keys"))
            row = result.fetchone()
            assert row[0] == 1

    async def test_init_db_seeds_default_tenant(self, db: Database) -> None:
        """init_db should create a default tenant row."""
        async with db.session() as s:
            tenant = await s.get(TenantRow, "default")
            assert tenant is not None
            assert tenant.name == "Default"


class TestMultipleTablesFK:
    """Verify FK works across several child tables."""

    async def test_position_fk_enforced(self, db: Database) -> None:
        """PositionRow should reject invalid tenant_id."""
        async with db.session() as s:
            s.add(
                PositionRow(
                    tenant_id="ghost",
                    portfolio="A",
                    ticker="AAPL",
                    shares=10.0,
                    avg_price=150.0,
                    updated_at=datetime.now(timezone.utc),
                )
            )
            with pytest.raises(IntegrityError):
                await s.commit()

    async def test_trade_fk_enforced(self, db: Database) -> None:
        """TradeRow should reject invalid tenant_id."""
        async with db.session() as s:
            s.add(
                TradeRow(
                    tenant_id="ghost",
                    portfolio="A",
                    ticker="AAPL",
                    side="BUY",
                    shares=10.0,
                    price=150.0,
                    total=1500.0,
                    executed_at=datetime.now(timezone.utc),
                )
            )
            with pytest.raises(IntegrityError):
                await s.commit()

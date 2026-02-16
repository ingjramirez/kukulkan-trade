"""Shared test fixtures."""

import pytest
from sqlalchemy import event
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from config.settings import settings
from src.storage.models import Base, TenantRow

# Disable Telegram during tests so no real messages are sent
settings.telegram.bot_token = ""
settings.telegram.chat_id = ""


@pytest.fixture
async def db_session():
    """Create an in-memory SQLite database for testing."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")

    @event.listens_for(engine.sync_engine, "connect")
    def _set_sqlite_fk(dbapi_connection, connection_record):  # noqa: ARG001
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    # Seed default tenant (FK anchor for tenant_id columns)
    async with async_session() as session:
        session.add(TenantRow(id="default", name="Default"))
        await session.commit()

    async with async_session() as session:
        yield session

    await engine.dispose()

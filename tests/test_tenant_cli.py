"""Tests for tenant CLI commands."""

import argparse
from unittest.mock import AsyncMock, patch

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
    database = Database(url="sqlite+aiosqlite:///:memory:")
    await database.init_db()
    yield database
    await database.close()


@pytest.fixture
def _patch_get_db(db: Database):
    """Patch _get_db to return the test db and prevent close from destroying it."""
    _original_close = db.close
    db.close = AsyncMock()

    async def _mock():
        return db

    with patch("src.cli.tenant_cli._get_db", new=_mock):
        yield

    db.close = _original_close


class TestSeedDefault:
    async def test_seed_creates_default_tenant(
        self, db: Database, _patch_get_db, monkeypatch,
    ):
        """seed_default should create a tenant from env vars when no tenants exist."""
        monkeypatch.setattr(settings.alpaca, "api_key", "APCA-KEY-LONG")
        monkeypatch.setattr(settings.alpaca, "secret_key", "APCA-SECRET-LONG")
        monkeypatch.setattr(settings.telegram, "bot_token", "BOT-TOKEN-LONG")
        monkeypatch.setattr(settings.telegram, "chat_id", "12345")

        from src.cli.tenant_cli import seed_default

        await seed_default(argparse.Namespace())

        tenants = await db.get_all_tenants()
        assert len(tenants) == 1
        assert tenants[0].id == "default"
        assert tenants[0].name == "Default"

    async def test_seed_skips_if_tenants_exist(
        self, db: Database, _patch_get_db, monkeypatch,
    ):
        """seed_default should not create duplicates."""
        await db.create_tenant(TenantRow(
            id="existing",
            name="Existing",
            alpaca_api_key_enc=encrypt_value("k"),
            alpaca_api_secret_enc=encrypt_value("s"),
            telegram_bot_token_enc=encrypt_value("t"),
            telegram_chat_id_enc=encrypt_value("c"),
        ))

        from src.cli.tenant_cli import seed_default

        await seed_default(argparse.Namespace())

        tenants = await db.get_all_tenants()
        assert len(tenants) == 1


class TestAddTenant:
    async def test_add_creates_tenant(self, db: Database, _patch_get_db):
        from src.cli.tenant_cli import add_tenant

        args = argparse.Namespace(
            name="Papa",
            alpaca_key="KEY",
            alpaca_secret="SECRET",
            alpaca_url="https://paper-api.alpaca.markets",
            telegram_token="TOKEN",
            telegram_chat_id="999",
            strategy="aggressive",
            portfolio_b_only=True,
            portfolio_a_cash=33000.0,
            portfolio_b_cash=66000.0,
            add_tickers="COIN,MSTR",
            remove_tickers=None,
            username=None,
            password=None,
        )

        await add_tenant(args)

        tenants = await db.get_all_tenants()
        assert len(tenants) == 1
        t = tenants[0]
        assert t.name == "Papa"
        assert t.strategy_mode == "aggressive"
        assert t.run_portfolio_a is False
        assert t.run_portfolio_b is True
        assert '"COIN"' in t.ticker_additions


class TestListTenants:
    async def test_list_empty(self, db: Database, _patch_get_db, capsys):
        from src.cli.tenant_cli import list_tenants

        await list_tenants(argparse.Namespace())

        output = capsys.readouterr().out
        assert "No tenants configured" in output

"""Tests for Alpaca client factory and Telegram factory."""

from unittest.mock import MagicMock, patch

import pytest
from cryptography.fernet import Fernet

from config.settings import settings

_TEST_KEY = Fernet.generate_key().decode()


@pytest.fixture(autouse=True)
def _set_encryption_key(monkeypatch):
    monkeypatch.setattr(settings, "tenant_encryption_key", _TEST_KEY)


@pytest.fixture(autouse=True)
def _clear_caches():
    """Ensure factory caches are clean before each test."""
    from src.execution.client_factory import AlpacaClientFactory
    from src.notifications.telegram_factory import TelegramFactory
    AlpacaClientFactory.clear_cache()
    TelegramFactory.clear_cache()
    yield
    AlpacaClientFactory.clear_cache()
    TelegramFactory.clear_cache()


def _mock_tenant(tenant_id: str = "t1") -> MagicMock:
    from src.utils.crypto import encrypt_value
    tenant = MagicMock()
    tenant.id = tenant_id
    tenant.alpaca_api_key_enc = encrypt_value("APCA-KEY")
    tenant.alpaca_api_secret_enc = encrypt_value("APCA-SECRET")
    tenant.alpaca_base_url = "https://paper-api.alpaca.markets"
    tenant.telegram_bot_token_enc = encrypt_value("BOT-TOKEN")
    tenant.telegram_chat_id_enc = encrypt_value("12345")
    return tenant


class TestAlpacaClientFactory:
    @patch("src.execution.client_factory.TradingClient")
    def test_creates_client(self, mock_tc):
        from src.execution.client_factory import AlpacaClientFactory
        tenant = _mock_tenant()
        client = AlpacaClientFactory.get_trading_client(tenant)
        mock_tc.assert_called_once_with(
            api_key="APCA-KEY",
            secret_key="APCA-SECRET",
            paper=True,
        )
        assert client == mock_tc.return_value

    @patch("src.execution.client_factory.TradingClient")
    def test_caches_client(self, mock_tc):
        from src.execution.client_factory import AlpacaClientFactory
        tenant = _mock_tenant()
        c1 = AlpacaClientFactory.get_trading_client(tenant)
        c2 = AlpacaClientFactory.get_trading_client(tenant)
        assert c1 is c2
        mock_tc.assert_called_once()

    @patch("src.execution.client_factory.TradingClient")
    def test_separate_clients_per_tenant(self, mock_tc):
        from src.execution.client_factory import AlpacaClientFactory
        t1 = _mock_tenant("t1")
        t2 = _mock_tenant("t2")
        c1 = AlpacaClientFactory.get_trading_client(t1)
        c2 = AlpacaClientFactory.get_trading_client(t2)
        assert mock_tc.call_count == 2

    @patch("src.execution.client_factory.TradingClient")
    def test_invalidate_evicts_from_cache(self, mock_tc):
        from src.execution.client_factory import AlpacaClientFactory
        tenant = _mock_tenant()
        AlpacaClientFactory.get_trading_client(tenant)
        AlpacaClientFactory.invalidate("t1")
        AlpacaClientFactory.get_trading_client(tenant)
        assert mock_tc.call_count == 2


class TestTelegramFactory:
    def test_creates_notifier(self):
        from src.notifications.telegram_factory import TelegramFactory
        tenant = _mock_tenant()
        notifier = TelegramFactory.get_notifier(tenant)
        assert notifier._token == "BOT-TOKEN"
        assert notifier._chat_id == "12345"

    def test_caches_notifier(self):
        from src.notifications.telegram_factory import TelegramFactory
        tenant = _mock_tenant()
        n1 = TelegramFactory.get_notifier(tenant)
        n2 = TelegramFactory.get_notifier(tenant)
        assert n1 is n2

    def test_invalidate_evicts(self):
        from src.notifications.telegram_factory import TelegramFactory
        tenant = _mock_tenant()
        n1 = TelegramFactory.get_notifier(tenant)
        TelegramFactory.invalidate("t1")
        n2 = TelegramFactory.get_notifier(tenant)
        assert n1 is not n2

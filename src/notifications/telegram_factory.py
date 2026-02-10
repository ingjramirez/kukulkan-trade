"""Dynamic Telegram bot factory for multi-tenant support.

Creates and caches TelegramNotifier instances per tenant, using
decrypted credentials from the tenant's encrypted config.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog

from src.notifications.telegram_bot import TelegramNotifier
from src.utils.crypto import decrypt_value

if TYPE_CHECKING:
    from src.storage.models import TenantRow

log = structlog.get_logger()


class TelegramFactory:
    """Creates and caches TelegramNotifier instances per tenant."""

    _cache: dict[str, TelegramNotifier] = {}

    @classmethod
    def get_notifier(cls, tenant: "TenantRow") -> TelegramNotifier:
        """Get or create a TelegramNotifier for the given tenant.

        Args:
            tenant: TenantRow with encrypted Telegram credentials.

        Returns:
            Cached or newly-created TelegramNotifier.
        """
        if tenant.id in cls._cache:
            return cls._cache[tenant.id]

        bot_token = decrypt_value(tenant.telegram_bot_token_enc)
        chat_id = decrypt_value(tenant.telegram_chat_id_enc)

        notifier = TelegramNotifier(bot_token=bot_token, chat_id=chat_id)
        cls._cache[tenant.id] = notifier
        log.info("telegram_notifier_created", tenant=tenant.id)
        return notifier

    @classmethod
    def invalidate(cls, tenant_id: str) -> None:
        """Remove a cached notifier (e.g. after credential rotation).

        Args:
            tenant_id: Tenant UUID to evict from cache.
        """
        cls._cache.pop(tenant_id, None)
        log.info("telegram_notifier_invalidated", tenant=tenant_id)

    @classmethod
    def clear_cache(cls) -> None:
        """Clear all cached notifiers."""
        cls._cache.clear()

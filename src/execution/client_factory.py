"""Dynamic Alpaca client factory for multi-tenant support.

Creates and caches TradingClient instances per tenant, using
decrypted credentials from the tenant's encrypted config.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog
from alpaca.trading.client import TradingClient

from src.utils.crypto import decrypt_value

if TYPE_CHECKING:
    from src.storage.models import TenantRow

log = structlog.get_logger()


class AlpacaClientFactory:
    """Creates and caches Alpaca TradingClient instances per tenant."""

    _cache: dict[str, TradingClient] = {}

    @classmethod
    def get_trading_client(cls, tenant: "TenantRow") -> TradingClient:
        """Get or create a TradingClient for the given tenant.

        Args:
            tenant: TenantRow with encrypted Alpaca credentials.

        Returns:
            Cached or newly-created TradingClient.
        """
        if tenant.id in cls._cache:
            return cls._cache[tenant.id]

        api_key = decrypt_value(tenant.alpaca_api_key_enc)
        api_secret = decrypt_value(tenant.alpaca_api_secret_enc)
        paper = "paper" in (tenant.alpaca_base_url or "paper")

        client = TradingClient(
            api_key=api_key,
            secret_key=api_secret,
            paper=paper,
        )
        cls._cache[tenant.id] = client
        log.info("alpaca_client_created", tenant=tenant.id, paper=paper)
        return client

    @classmethod
    def invalidate(cls, tenant_id: str) -> None:
        """Remove a cached client (e.g. after credential rotation).

        Args:
            tenant_id: Tenant UUID to evict from cache.
        """
        cls._cache.pop(tenant_id, None)
        log.info("alpaca_client_invalidated", tenant=tenant_id)

    @classmethod
    def clear_cache(cls) -> None:
        """Clear all cached clients."""
        cls._cache.clear()

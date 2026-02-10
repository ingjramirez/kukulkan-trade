"""Tenant-specific ticker universe resolution.

Resolves the effective universe for a tenant based on their
whitelist, additions, and exclusions configuration.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import structlog

from config.universe import PORTFOLIO_A_UNIVERSE, PORTFOLIO_B_UNIVERSE

if TYPE_CHECKING:
    from src.storage.models import TenantRow

log = structlog.get_logger()


def get_tenant_universe(tenant: "TenantRow", portfolio: str = "B") -> list[str]:
    """Resolve the effective ticker universe for a tenant.

    Priority:
    1. If ticker_whitelist is set, use ONLY those tickers (exclusive mode).
    2. Otherwise, start from the base universe and apply additions/exclusions.

    Args:
        tenant: TenantRow with ticker customization fields.
        portfolio: "A" or "B" — determines which base universe to start from.

    Returns:
        Sorted list of ticker symbols.
    """
    # Exclusive mode: whitelist overrides everything
    whitelist = _parse_json_list(tenant.ticker_whitelist)
    if whitelist:
        log.debug(
            "tenant_universe_whitelist",
            tenant=tenant.id,
            tickers=len(whitelist),
        )
        return sorted(set(whitelist))

    # Additive/subtractive mode
    base = set(
        PORTFOLIO_A_UNIVERSE if portfolio == "A" else PORTFOLIO_B_UNIVERSE
    )

    additions = _parse_json_list(tenant.ticker_additions)
    if additions:
        base |= set(additions)

    exclusions = _parse_json_list(tenant.ticker_exclusions)
    if exclusions:
        base -= set(exclusions)

    log.debug(
        "tenant_universe_resolved",
        tenant=tenant.id,
        portfolio=portfolio,
        tickers=len(base),
        additions=len(additions) if additions else 0,
        exclusions=len(exclusions) if exclusions else 0,
    )
    return sorted(base)


def _parse_json_list(value: str | None) -> list[str] | None:
    """Parse a JSON string into a list, or return None.

    Args:
        value: JSON string like '["AAPL","TSLA"]' or None.

    Returns:
        Parsed list or None if empty/invalid.
    """
    if not value:
        return None
    try:
        parsed = json.loads(value)
        if isinstance(parsed, list) and all(isinstance(t, str) for t in parsed):
            return [t.upper().strip() for t in parsed if t.strip()]
        return None
    except (json.JSONDecodeError, TypeError):
        log.warning("invalid_ticker_json", value=value[:50] if value else "")
        return None

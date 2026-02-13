"""Intraday snapshot collector.

Fetches live Alpaca positions every 15 minutes during market hours
and stores per-portfolio equity values for high-frequency charting.
"""

import asyncio
from datetime import datetime, timezone

import structlog

from src.storage.database import Database
from src.storage.models import TenantRow

log = structlog.get_logger()


async def collect_intraday_snapshot(
    db: Database,
    tenant: TenantRow,
) -> int:
    """Collect and store intraday snapshots for a tenant's enabled portfolios.

    Fetches live prices from Alpaca, sums per-portfolio position values using
    DB positions, and stores a snapshot row per portfolio.

    Args:
        db: Database instance.
        tenant: Active TenantRow with Alpaca credentials.

    Returns:
        Number of snapshots saved.
    """
    from src.execution.client_factory import AlpacaClientFactory

    client = AlpacaClientFactory.get_trading_client(tenant)

    # Fetch live positions from Alpaca
    try:
        alpaca_positions = await asyncio.to_thread(client.get_all_positions)
    except Exception as e:
        log.warning("intraday_alpaca_fetch_failed", tenant_id=tenant.id, error=str(e))
        return 0

    # Build live price map from broker positions
    live_prices: dict[str, float] = {}
    for pos in alpaca_positions:
        live_prices[pos.symbol] = float(pos.current_price)

    now = datetime.now(timezone.utc)
    # Truncate to the minute for consistent timestamps
    now = now.replace(second=0, microsecond=0)

    saved = 0
    portfolios = _enabled_portfolios(tenant)

    for pname in portfolios:
        portfolio = await db.get_portfolio(pname, tenant_id=tenant.id)
        if portfolio is None:
            continue

        positions = await db.get_positions(pname, tenant_id=tenant.id)
        positions_value = sum(
            p.shares * live_prices.get(p.ticker, p.avg_price)
            for p in positions
        )
        total_value = portfolio.cash + positions_value

        await db.save_intraday_snapshot(
            tenant_id=tenant.id,
            portfolio=pname,
            timestamp=now,
            total_value=total_value,
            cash=portfolio.cash,
            positions_value=positions_value,
        )
        saved += 1

    log.info(
        "intraday_snapshot_collected",
        tenant_id=tenant.id,
        portfolios=saved,
        timestamp=now.isoformat(),
    )
    return saved


def _enabled_portfolios(tenant: TenantRow) -> list[str]:
    """Return list of enabled portfolio names for a tenant."""
    portfolios: list[str] = []
    if tenant.run_portfolio_a:
        portfolios.append("A")
    if tenant.run_portfolio_b:
        portfolios.append("B")
    return portfolios

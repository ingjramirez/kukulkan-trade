"""Intraday snapshot collector.

Fetches live Alpaca positions every 15 minutes during market hours
and stores per-portfolio equity values for high-frequency charting.

Extended hours support: pre-market (7-9:30 ET) and after-hours (16-20 ET)
use yfinance extended prices instead of Alpaca live positions.
"""

import asyncio
from datetime import datetime

import structlog

from src.storage.database import Database
from src.storage.models import TenantRow
from src.utils.market_time import MarketPhase

log = structlog.get_logger()


async def collect_intraday_snapshot(
    db: Database,
    tenant: TenantRow,
    market_phase: MarketPhase = MarketPhase.MARKET,
) -> int:
    """Collect and store intraday snapshots for a tenant's enabled portfolios.

    During market hours, fetches live prices from Alpaca.
    During extended hours, uses yfinance extended prices.

    Args:
        db: Database instance.
        tenant: Active TenantRow with Alpaca credentials.
        market_phase: Current market phase (affects price source).

    Returns:
        Number of snapshots saved.
    """
    is_extended = market_phase in (MarketPhase.PREMARKET, MarketPhase.AFTERHOURS)

    if is_extended:
        live_prices = await _fetch_extended_prices(db, tenant)
    else:
        live_prices = await _fetch_alpaca_prices(tenant)

    if live_prices is None:
        return 0

    now = datetime.utcnow()
    now = now.replace(second=0, microsecond=0)

    saved = 0
    portfolios = _enabled_portfolios(tenant)

    for pname in portfolios:
        portfolio = await db.get_portfolio(pname, tenant_id=tenant.id)
        if portfolio is None:
            continue

        positions = await db.get_positions(pname, tenant_id=tenant.id)
        positions_value = sum(p.shares * live_prices.get(p.ticker, p.avg_price) for p in positions)
        total_value = portfolio.cash + positions_value

        await db.save_intraday_snapshot(
            tenant_id=tenant.id,
            portfolio=pname,
            timestamp=now,
            total_value=total_value,
            cash=portfolio.cash,
            positions_value=positions_value,
            is_extended_hours=is_extended,
            market_phase=market_phase.value,
        )

        # Publish SSE event
        try:
            from src.events.event_bus import Event, EventType, event_bus

            await event_bus.publish(
                Event(
                    type=EventType.INTRADAY_UPDATE,
                    tenant_id=tenant.id,
                    data={
                        "portfolio": pname,
                        "equity": total_value,
                        "cash": portfolio.cash,
                        "is_extended_hours": is_extended,
                        "market_phase": market_phase.value,
                    },
                )
            )
        except Exception:
            pass  # SSE is best-effort

        saved += 1

    log.info(
        "intraday_snapshot_collected",
        tenant_id=tenant.id,
        portfolios=saved,
        market_phase=market_phase.value,
        timestamp=now.isoformat(),
    )
    return saved


async def _fetch_alpaca_prices(tenant: TenantRow) -> dict[str, float] | None:
    """Fetch live prices from Alpaca (market hours)."""
    from src.execution.client_factory import AlpacaClientFactory

    client = AlpacaClientFactory.get_trading_client(tenant)
    try:
        alpaca_positions = await asyncio.to_thread(client.get_all_positions)
    except Exception as e:
        log.warning("intraday_alpaca_fetch_failed", tenant_id=tenant.id, error=str(e))
        return None

    return {pos.symbol: float(pos.current_price) for pos in alpaca_positions}


async def _fetch_extended_prices(db: Database, tenant: TenantRow) -> dict[str, float] | None:
    """Fetch extended hours prices via yfinance for all held tickers."""
    from src.data.market_data import get_extended_hours_prices

    portfolios = _enabled_portfolios(tenant)
    tickers: set[str] = set()
    for pname in portfolios:
        positions = await db.get_positions(pname, tenant_id=tenant.id)
        tickers.update(p.ticker for p in positions)

    if not tickers:
        return {}

    try:
        return await get_extended_hours_prices(list(tickers))
    except Exception as e:
        log.warning("intraday_extended_fetch_failed", tenant_id=tenant.id, error=str(e))
        return None


def _enabled_portfolios(tenant: TenantRow) -> list[str]:
    """Return list of enabled portfolio names for a tenant."""
    portfolios: list[str] = []
    if tenant.run_portfolio_a:
        portfolios.append("A")
    if tenant.run_portfolio_b:
        portfolios.append("B")
    return portfolios

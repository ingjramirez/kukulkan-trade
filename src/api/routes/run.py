"""POST /api/run — trigger a bot run for the authenticated tenant."""

from __future__ import annotations

import asyncio
import time

import structlog
from fastapi import APIRouter, Depends, HTTPException

from src.api.deps import get_authorized_tenant_id, get_current_user, get_db
from src.orchestrator import Orchestrator
from src.storage.database import Database

log = structlog.get_logger()

router = APIRouter(prefix="/api", tags=["run"])

# Per-tenant concurrency lock: only 1 active run at a time.
_running: dict[str, bool] = {}
# Per-tenant rate limit: timestamp of last trigger.
_last_trigger: dict[str, float] = {}

# Minimum seconds between triggers for the same tenant.
_RATE_LIMIT_SECONDS = 60.0


def _reset_run_state() -> None:
    """Clear concurrency and rate-limit state. Used in tests."""
    _running.clear()
    _last_trigger.clear()


@router.post("/run", status_code=202)
async def trigger_run(
    db: Database = Depends(get_db),
    _user: dict = Depends(get_current_user),
    tenant_id: str = Depends(get_authorized_tenant_id),
) -> dict:
    """Trigger the trading pipeline for the authenticated tenant.

    Returns 202 immediately; the pipeline runs in the background.
    """
    # Per-tenant rate limit
    now = time.monotonic()
    last = _last_trigger.get(tenant_id, 0.0)
    if now - last < _RATE_LIMIT_SECONDS:
        raise HTTPException(
            status_code=429,
            detail="Run already triggered recently. Try again later.",
        )

    # Concurrency guard
    if _running.get(tenant_id):
        raise HTTPException(
            status_code=409,
            detail="A run is already in progress for this tenant.",
        )

    # Fetch tenant and validate credentials
    tenant = await db.get_tenant(tenant_id)
    if tenant is None:
        raise HTTPException(status_code=404, detail="Tenant not found")

    if not Orchestrator._tenant_fully_configured(tenant):
        raise HTTPException(
            status_code=422,
            detail="Alpaca and Telegram credentials must be configured before running.",
        )

    # Acquire lock and record timestamp
    _running[tenant_id] = True
    _last_trigger[tenant_id] = now

    asyncio.create_task(_run_pipeline(db, tenant_id))

    return {"status": "triggered", "tenant_id": tenant_id}


async def _run_pipeline(db: Database, tenant_id: str) -> None:
    """Background task: create tenant-scoped orchestrator and run the pipeline."""
    try:
        tenant = await db.get_tenant(tenant_id)
        if tenant is None:
            log.error("run_pipeline_tenant_missing", tenant_id=tenant_id)
            return

        from src.execution.alpaca_executor import AlpacaExecutor
        from src.execution.client_factory import AlpacaClientFactory
        from src.notifications.telegram_factory import TelegramFactory

        notifier = TelegramFactory.get_notifier(tenant)
        client = AlpacaClientFactory.get_trading_client(tenant)
        executor = AlpacaExecutor(db, client)

        from src.agent.ticker_discovery import TickerDiscovery
        from src.utils.allocations import resolve_from_tenant
        from src.utils.tenant_universe import get_tenant_universe

        alloc = resolve_from_tenant(tenant)
        discovery = TickerDiscovery(db)
        discovered = await discovery.get_active_tickers(tenant_id=tenant_id)
        tenant_b_universe = get_tenant_universe(
            tenant,
            "B",
            discovered_tickers=discovered,
        )

        orchestrator = Orchestrator(db, notifier=notifier, executor=executor)
        await orchestrator.run_daily(
            tenant_id=tenant_id,
            session="Manual",
            strategy_mode=tenant.strategy_mode,
            run_portfolio_a=tenant.run_portfolio_a,
            run_portfolio_b=tenant.run_portfolio_b,
            allocations=alloc,
            portfolio_b_universe=tenant_b_universe,
        )
        log.info("manual_run_complete", tenant_id=tenant_id)
    except Exception as e:
        log.error("manual_run_failed", tenant_id=tenant_id, error=str(e))
        # Try to notify via Telegram
        try:
            from src.notifications.telegram_factory import TelegramFactory

            tenant = await db.get_tenant(tenant_id)
            if tenant:
                notifier = TelegramFactory.get_notifier(tenant)
                await notifier.send_message(f"Manual run failed for {tenant.name}: {e}")
        except Exception:
            pass
    finally:
        _running.pop(tenant_id, None)

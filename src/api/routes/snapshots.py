"""GET /api/snapshots — daily and intraday portfolio snapshots."""

from datetime import date, datetime, timedelta

from fastapi import APIRouter, Depends, Query

from src.api.deps import get_authorized_tenant_id, get_db
from src.api.schemas import IntradaySnapshotResponse, SnapshotResponse
from src.storage.database import Database

router = APIRouter(prefix="/api", tags=["snapshots"])

_PERIOD_DAYS = {"1d": 1, "3d": 3, "1w": 7, "1m": 30}


@router.get("/snapshots", response_model=list[SnapshotResponse])
async def list_snapshots(
    portfolio: str | None = Query(None),
    since: date | None = Query(None),
    tenant_id: str = Depends(get_authorized_tenant_id),
    db: Database = Depends(get_db),
) -> list[SnapshotResponse]:
    rows = await db.get_all_snapshots(
        portfolio=portfolio,
        since=since,
        tenant_id=tenant_id,
    )
    return [
        SnapshotResponse(
            portfolio=r.portfolio,
            date=r.date,
            total_value=r.total_value,
            cash=r.cash,
            positions_value=r.positions_value,
            daily_return_pct=r.daily_return_pct,
            cumulative_return_pct=r.cumulative_return_pct,
        )
        for r in rows
    ]


@router.get("/snapshots/intraday", response_model=list[IntradaySnapshotResponse])
async def list_intraday_snapshots(
    portfolio: str | None = Query(None),
    period: str = Query("1d", pattern=r"^(1d|3d|1w|1m)$"),
    tenant_id: str = Depends(get_authorized_tenant_id),
    db: Database = Depends(get_db),
) -> list[IntradaySnapshotResponse]:
    days = _PERIOD_DAYS[period]
    since = datetime.utcnow() - timedelta(days=days)
    rows = await db.get_intraday_snapshots(
        tenant_id=tenant_id,
        portfolio=portfolio,
        since=since,
    )
    return [
        IntradaySnapshotResponse(
            portfolio=r.portfolio,
            timestamp=r.timestamp,
            total_value=r.total_value,
            cash=r.cash,
            positions_value=r.positions_value,
            is_extended_hours=getattr(r, "is_extended_hours", False),
            market_phase=getattr(r, "market_phase", "market"),
        )
        for r in rows
    ]

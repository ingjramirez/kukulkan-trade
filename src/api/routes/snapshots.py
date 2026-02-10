"""GET /api/snapshots — daily portfolio snapshots."""

from datetime import date

from fastapi import APIRouter, Depends, Query

from src.api.deps import get_current_user, get_db
from src.api.schemas import SnapshotResponse
from src.storage.database import Database

router = APIRouter(prefix="/api", tags=["snapshots"])


@router.get("/snapshots", response_model=list[SnapshotResponse])
async def list_snapshots(
    portfolio: str | None = Query(None),
    since: date | None = Query(None),
    tenant_id: str = Query("default"),
    db: Database = Depends(get_db),
    _user: str = Depends(get_current_user),
) -> list[SnapshotResponse]:
    rows = await db.get_all_snapshots(
        portfolio=portfolio, since=since, tenant_id=tenant_id,
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

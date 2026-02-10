"""GET /api/momentum/rankings — latest momentum rankings."""

from fastapi import APIRouter, Depends, Query

from src.api.deps import get_current_user, get_db
from src.api.schemas import MomentumRankingResponse
from src.storage.database import Database

router = APIRouter(prefix="/api/momentum", tags=["momentum"])


@router.get("/rankings", response_model=list[MomentumRankingResponse])
async def list_rankings(
    tenant_id: str = Query("default"),  # noqa: ARG001 — accepted for FE compat
    db: Database = Depends(get_db),
    _user: dict = Depends(get_current_user),
) -> list[MomentumRankingResponse]:
    rows = await db.get_latest_momentum_rankings()
    return [
        MomentumRankingResponse(
            date=r.date,
            ticker=r.ticker,
            return_63d=r.return_63d,
            rank=r.rank,
        )
        for r in rows
    ]

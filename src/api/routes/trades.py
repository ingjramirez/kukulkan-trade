"""GET /api/trades — trade history."""

from fastapi import APIRouter, Depends, Query

from src.api.deps import get_current_user, get_db
from src.api.schemas import TradeResponse
from src.storage.database import Database

router = APIRouter(prefix="/api", tags=["trades"])


@router.get("/trades", response_model=list[TradeResponse])
async def list_trades(
    portfolio: str | None = Query(None),
    side: str | None = Query(None),
    limit: int = Query(100, ge=1, le=1000),
    db: Database = Depends(get_db),
    _user: str = Depends(get_current_user),
) -> list[TradeResponse]:
    rows = await db.get_all_trades(portfolio=portfolio, side=side, limit=limit)
    return [
        TradeResponse(
            id=r.id,
            portfolio=r.portfolio,
            ticker=r.ticker,
            side=r.side,
            shares=r.shares,
            price=r.price,
            total=r.total,
            reason=r.reason,
            executed_at=r.executed_at,
        )
        for r in rows
    ]

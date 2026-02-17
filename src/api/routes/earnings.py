"""Earnings calendar endpoint."""

from fastapi import APIRouter, Depends

from src.api.deps import get_authorized_tenant_id, get_db
from src.api.schemas import EarningsUpcomingResponse
from src.storage.database import Database

router = APIRouter(prefix="/api/earnings", tags=["earnings"])


@router.get("/upcoming", response_model=list[EarningsUpcomingResponse])
async def get_upcoming_earnings(
    days_ahead: int = 14,
    tenant_id: str = Depends(get_authorized_tenant_id),
    db: Database = Depends(get_db),
) -> list[EarningsUpcomingResponse]:
    """Get upcoming earnings for tickers in the tenant's universe."""
    # Get all positions across both portfolios for this tenant
    tickers: set[str] = set()
    for pname in ("A", "B"):
        positions = await db.get_positions(pname, tenant_id=tenant_id)
        tickers.update(p.ticker for p in positions)

    if not tickers:
        return []

    rows = await db.get_upcoming_earnings(list(tickers), days_ahead)
    return [
        EarningsUpcomingResponse(
            ticker=r.ticker,
            earnings_date=r.earnings_date.isoformat(),
            source=r.source,
        )
        for r in rows
    ]

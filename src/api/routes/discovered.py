"""Discovered tickers endpoint — tenant-scoped listing and approval."""

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from src.api.deps import get_authorized_tenant_id, get_db
from src.storage.database import Database

router = APIRouter(prefix="/api/discovered", tags=["discovered"])


class DiscoveredTickerResponse(BaseModel):
    ticker: str
    source: str
    rationale: str | None = None
    status: str
    proposed_at: str
    expires_at: str
    sector: str | None = None
    market_cap: float | None = None


class UpdateStatusRequest(BaseModel):
    status: str = Field(pattern=r"^(approved|rejected)$")


@router.get("")
async def list_discovered_tickers(
    status_filter: str | None = None,
    tenant_id: str = Depends(get_authorized_tenant_id),
    db: Database = Depends(get_db),
) -> list[DiscoveredTickerResponse]:
    """List discovered tickers for the authenticated tenant."""
    rows = await db.get_all_discovered_tickers(
        tenant_id=tenant_id,
        status=status_filter,
    )
    return [
        DiscoveredTickerResponse(
            ticker=r.ticker,
            source=r.source,
            rationale=r.rationale,
            status=r.status,
            proposed_at=r.proposed_at.isoformat(),
            expires_at=r.expires_at.isoformat(),
            sector=r.sector,
            market_cap=r.market_cap,
        )
        for r in rows
    ]


@router.patch("/{ticker}")
async def update_discovered_ticker(
    ticker: str,
    body: UpdateStatusRequest,
    tenant_id: str = Depends(get_authorized_tenant_id),
    db: Database = Depends(get_db),
) -> DiscoveredTickerResponse:
    """Approve or reject a proposed discovered ticker."""
    ticker = ticker.upper().strip()
    row = await db.get_discovered_ticker(ticker, tenant_id=tenant_id)
    if row is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Ticker {ticker} not found",
        )
    if row.status != "proposed":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Ticker {ticker} already {row.status}",
        )

    await db.update_discovered_ticker_status(
        ticker,
        body.status,
        tenant_id=tenant_id,
    )
    return DiscoveredTickerResponse(
        ticker=row.ticker,
        source=row.source,
        rationale=row.rationale,
        status=body.status,
        proposed_at=row.proposed_at.isoformat(),
        expires_at=row.expires_at.isoformat(),
        sector=row.sector,
        market_cap=row.market_cap,
    )

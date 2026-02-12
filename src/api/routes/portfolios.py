"""Portfolio endpoints — list, detail, positions, trailing stops, and watchlist."""

from fastapi import APIRouter, Depends, HTTPException

from src.api.deps import get_authorized_tenant_id, get_db
from src.api.schemas import PortfolioDetail, PortfolioSummary, PositionResponse
from src.storage.database import Database

router = APIRouter(prefix="/api/portfolios", tags=["portfolios"])


@router.get("", response_model=list[PortfolioSummary])
async def list_portfolios(
    tenant_id: str = Depends(get_authorized_tenant_id),
    db: Database = Depends(get_db),
) -> list[PortfolioSummary]:
    rows = await db.get_all_portfolios(tenant_id=tenant_id)
    return [
        PortfolioSummary(
            name=r.name,
            cash=r.cash,
            total_value=r.total_value,
            updated_at=r.updated_at,
        )
        for r in rows
    ]


@router.get("/{name}", response_model=PortfolioDetail)
async def get_portfolio(
    name: str,
    tenant_id: str = Depends(get_authorized_tenant_id),
    db: Database = Depends(get_db),
) -> PortfolioDetail:
    portfolio = await db.get_portfolio(name, tenant_id=tenant_id)
    if not portfolio:
        raise HTTPException(status_code=404, detail="Portfolio not found")
    positions = await db.get_positions(name, tenant_id=tenant_id)
    return PortfolioDetail(
        name=portfolio.name,
        cash=portfolio.cash,
        total_value=portfolio.total_value,
        updated_at=portfolio.updated_at,
        positions=[
            PositionResponse(
                portfolio=p.portfolio,
                ticker=p.ticker,
                shares=p.shares,
                avg_price=p.avg_price,
                current_price=p.current_price,
                market_value=p.market_value,
            )
            for p in positions
        ],
    )


@router.get("/{name}/positions", response_model=list[PositionResponse])
async def get_positions(
    name: str,
    tenant_id: str = Depends(get_authorized_tenant_id),
    db: Database = Depends(get_db),
) -> list[PositionResponse]:
    positions = await db.get_positions(name, tenant_id=tenant_id)
    return [
        PositionResponse(
            portfolio=p.portfolio,
            ticker=p.ticker,
            shares=p.shares,
            avg_price=p.avg_price,
            current_price=p.current_price,
            market_value=p.market_value,
        )
        for p in positions
    ]


@router.get("/{name}/trailing-stops")
async def get_trailing_stops(
    name: str,
    tenant_id: str = Depends(get_authorized_tenant_id),
    db: Database = Depends(get_db),
) -> list[dict]:
    """Get active trailing stops for a portfolio."""
    stops = await db.get_active_trailing_stops(tenant_id, name)
    return [
        {
            "ticker": s.ticker,
            "entry_price": s.entry_price,
            "peak_price": s.peak_price,
            "trail_pct": s.trail_pct,
            "stop_price": s.stop_price,
            "is_active": s.is_active,
        }
        for s in stops
    ]


@router.get("/{name}/watchlist")
async def get_watchlist(
    name: str,
    tenant_id: str = Depends(get_authorized_tenant_id),
    db: Database = Depends(get_db),
) -> list[dict]:
    """Get watchlist items for a portfolio."""
    items = await db.get_watchlist(tenant_id, name)
    return [
        {
            "ticker": w.ticker,
            "reason": w.reason,
            "conviction": w.conviction,
            "target_entry": w.target_entry,
            "added_date": str(w.added_date),
            "expires_at": str(w.expires_at),
        }
        for w in items
    ]

"""Portfolio endpoints — list, detail, and positions."""

from fastapi import APIRouter, Depends, HTTPException, Query

from src.api.deps import get_current_user, get_db
from src.api.schemas import PortfolioDetail, PortfolioSummary, PositionResponse
from src.storage.database import Database

router = APIRouter(prefix="/api/portfolios", tags=["portfolios"])


@router.get("", response_model=list[PortfolioSummary])
async def list_portfolios(
    tenant_id: str = Query("default"),
    db: Database = Depends(get_db),
    _user: str = Depends(get_current_user),
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
    tenant_id: str = Query("default"),
    db: Database = Depends(get_db),
    _user: str = Depends(get_current_user),
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
    tenant_id: str = Query("default"),
    db: Database = Depends(get_db),
    _user: str = Depends(get_current_user),
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

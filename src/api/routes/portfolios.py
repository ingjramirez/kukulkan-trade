"""Portfolio endpoints — list, detail, positions, trailing stops, watchlist, after-hours P&L, and gap risk."""

from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException

from src.api.deps import get_authorized_tenant_id, get_db
from src.api.schemas import PortfolioDetail, PortfolioSummary, PositionResponse
from src.storage.database import Database
from src.utils.market_time import ET, MarketPhase, get_market_phase

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


# --- Static paths MUST come before /{name} catch-all ---


@router.get("/after-hours-pnl")
async def get_after_hours_pnl(
    tenant_id: str = Depends(get_authorized_tenant_id),
    db: Database = Depends(get_db),
) -> dict:
    """Get after-hours or pre-market P&L vs. last market close snapshot."""
    phase = get_market_phase()
    if phase not in (MarketPhase.AFTERHOURS, MarketPhase.PREMARKET):
        return {"is_active": False, "market_phase": phase.value}

    close_snapshot = await db.get_last_market_hours_snapshot(tenant_id)
    if not close_snapshot:
        return {"is_active": False, "reason": "No market-hours snapshot found"}

    # Get all positions across portfolios
    positions_b = await db.get_positions("B", tenant_id=tenant_id)
    positions_a = await db.get_positions("A", tenant_id=tenant_id)
    all_positions = positions_b + positions_a
    tickers = [p.ticker for p in all_positions]

    if not tickers:
        return {
            "is_active": True,
            "market_phase": phase.value,
            "as_of": datetime.now(ET).isoformat(),
            "market_close_value": round(close_snapshot.total_value, 2),
            "current_value": round(close_snapshot.total_value, 2),
            "change": 0.0,
            "change_pct": 0.0,
            "movers": [],
        }

    from src.data.market_data import get_extended_hours_prices

    extended_prices = await get_extended_hours_prices(tickers)

    close_value = close_snapshot.total_value
    current_value = (
        sum(p.shares * extended_prices.get(p.ticker, p.current_price or p.avg_price) for p in all_positions)
        + close_snapshot.cash
    )

    change = current_value - close_value
    change_pct = (change / close_value * 100) if close_value > 0 else 0

    movers = []
    for p in all_positions:
        ext_price = extended_prices.get(p.ticker)
        if not ext_price:
            continue
        last = p.current_price or p.avg_price
        pos_change = (ext_price - last) / last * 100 if last else 0
        contribution = (ext_price - last) * p.shares
        movers.append(
            {
                "ticker": p.ticker,
                "close_price": round(last, 2),
                "current_price": round(ext_price, 2),
                "change_pct": round(pos_change, 2),
                "contribution": round(contribution, 2),
            }
        )
    movers.sort(key=lambda m: abs(m["contribution"]), reverse=True)

    return {
        "is_active": True,
        "market_phase": phase.value,
        "as_of": datetime.now(ET).isoformat(),
        "market_close_value": round(close_value, 2),
        "current_value": round(current_value, 2),
        "change": round(change, 2),
        "change_pct": round(change_pct, 2),
        "movers": movers[:10],
    }


@router.get("/overnight-risk")
async def get_overnight_risk(
    tenant_id: str = Depends(get_authorized_tenant_id),
    db: Database = Depends(get_db),
) -> dict:
    """Get overnight gap risk assessment for the portfolio."""
    from src.analysis.gap_risk import GapRiskAnalyzer

    analyzer = GapRiskAnalyzer()
    assessment = await analyzer.analyze(db, tenant_id)
    return {
        "as_of": datetime.now(ET).isoformat(),
        "aggregate_risk_score": assessment.aggregate_risk_score,
        "rating": assessment.rating,
        "earnings_tonight": assessment.earnings_tonight,
        "positions": [
            {
                "ticker": p.ticker,
                "weight_pct": p.weight_pct,
                "gap_risk_score": p.gap_risk_score,
                "reasons": p.reasons,
                "recommendation": p.recommendation,
            }
            for p in assessment.positions
        ],
    }


# --- Dynamic path parameter routes ---


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

    # Compute initial_value from tenant config
    initial_value: float | None = None
    return_pct: float | None = None
    tenant = await db.get_tenant(tenant_id)
    if tenant and tenant.initial_equity:
        pct = tenant.portfolio_a_pct if name == "A" else tenant.portfolio_b_pct
        initial_value = round(tenant.initial_equity * pct / 100, 2)
        if initial_value > 0:
            return_pct = round((portfolio.total_value - initial_value) / initial_value * 100, 2)

    return PortfolioDetail(
        name=portfolio.name,
        cash=portfolio.cash,
        total_value=portfolio.total_value,
        initial_value=initial_value,
        return_pct=return_pct,
        updated_at=portfolio.updated_at,
        positions=[
            PositionResponse(
                portfolio=p.portfolio,
                ticker=p.ticker,
                shares=p.shares,
                avg_price=p.avg_price,
                current_price=p.current_price or p.avg_price,
                market_value=p.market_value or (p.shares * p.avg_price),
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
            current_price=p.current_price or p.avg_price,
            market_value=p.market_value or (p.shares * p.avg_price),
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

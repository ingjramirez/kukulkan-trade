"""GET /api/account — live Alpaca account data."""

import asyncio

from fastapi import APIRouter, Depends, HTTPException, Query

from src.api.alpaca_client import get_live_account, get_portfolio_history
from src.api.deps import get_authorized_tenant_id, get_db
from src.api.schemas import AccountResponse, PortfolioHistoryResponse, PositionResponse
from src.storage.database import Database

router = APIRouter(prefix="/api", tags=["account"])


@router.get("/account", response_model=AccountResponse)
async def account(
    tenant_id: str = Depends(get_authorized_tenant_id),
    db: Database = Depends(get_db),
) -> AccountResponse:
    data = await _get_account_data(tenant_id, db)
    if data is None:
        raise HTTPException(status_code=503, detail="Alpaca unavailable")
    return AccountResponse(
        equity=data["equity"],
        last_equity=data["last_equity"],
        daily_pl=data["daily_pl"],
        daily_pl_pct=data["daily_pl_pct"],
        cash=data["cash"],
        buying_power=data["buying_power"],
        positions=[
            PositionResponse(
                ticker=p["ticker"],
                shares=p["shares"],
                avg_price=p["avg_price"],
                current_price=p["current_price"],
                market_value=p["market_value"],
                unrealized_pl=p["unrealized_pl"],
                unrealized_plpc=p["unrealized_plpc"],
            )
            for p in data["positions"]
        ],
    )


@router.get("/account/history", response_model=PortfolioHistoryResponse)
async def account_history(
    period: str = Query("1D", pattern=r"^(1D|1W|1M|3M|1A)$"),
    timeframe: str = Query("5Min", pattern=r"^(1Min|5Min|15Min|1H|1D)$"),
    extended_hours: bool = Query(False),
    tenant_id: str = Depends(get_authorized_tenant_id),
    db: Database = Depends(get_db),
) -> PortfolioHistoryResponse:
    data = await _get_portfolio_history(tenant_id, db, period, timeframe, extended_hours)
    if data is None:
        raise HTTPException(status_code=503, detail="Alpaca unavailable")
    return PortfolioHistoryResponse(**data)


async def _get_portfolio_history(
    tenant_id: str,
    db: Database,
    period: str,
    timeframe: str,
    extended_hours: bool,
) -> dict | None:
    """Fetch portfolio history — uses tenant-specific client or global default."""
    if tenant_id == "default":
        return await get_portfolio_history(period, timeframe, extended_hours)

    tenant = await db.get_tenant(tenant_id)
    if tenant is None:
        return None

    try:
        from src.execution.client_factory import AlpacaClientFactory

        client = AlpacaClientFactory.get_trading_client(tenant)

        from alpaca.trading.requests import GetPortfolioHistoryRequest

        req = GetPortfolioHistoryRequest(
            period=period,
            timeframe=timeframe,
            extended_hours=extended_hours,
        )
        result = await asyncio.to_thread(client.get_portfolio_history, req)

        return {
            "timestamps": result.timestamp or [],
            "equity": result.equity or [],
            "profit_loss": result.profit_loss or [],
            "profit_loss_pct": result.profit_loss_pct or [],
            "base_value": float(result.base_value) if result.base_value else 0.0,
            "timeframe": result.timeframe or timeframe,
        }
    except Exception:
        return None


async def _get_account_data(
    tenant_id: str,
    db: Database,
) -> dict | None:
    """Fetch account data — uses tenant-specific client or global default."""
    if tenant_id == "default":
        return await get_live_account()

    tenant = await db.get_tenant(tenant_id)
    if tenant is None:
        return None

    try:
        from src.execution.client_factory import AlpacaClientFactory

        client = AlpacaClientFactory.get_trading_client(tenant)
        account = await asyncio.to_thread(client.get_account)
        positions = await asyncio.to_thread(client.get_all_positions)

        equity = float(account.equity)
        last_equity = float(account.last_equity)
        daily_pl = equity - last_equity
        daily_pl_pct = (daily_pl / last_equity) * 100 if last_equity else 0.0

        return {
            "equity": equity,
            "last_equity": last_equity,
            "daily_pl": daily_pl,
            "daily_pl_pct": daily_pl_pct,
            "cash": float(account.cash),
            "buying_power": float(account.buying_power),
            "positions": [
                {
                    "ticker": p.symbol,
                    "shares": float(p.qty),
                    "avg_price": float(p.avg_entry_price),
                    "current_price": float(p.current_price),
                    "market_value": float(p.market_value),
                    "unrealized_pl": float(p.unrealized_pl),
                    "unrealized_plpc": float(p.unrealized_plpc) * 100,
                }
                for p in positions
            ],
        }
    except Exception:
        return None

"""GET /api/account — live Alpaca account data."""

from fastapi import APIRouter, Depends, HTTPException

from src.api.alpaca_client import get_live_account
from src.api.deps import get_current_user
from src.api.schemas import AccountResponse, PositionResponse

router = APIRouter(prefix="/api", tags=["account"])


@router.get("/account", response_model=AccountResponse)
async def account(_user: str = Depends(get_current_user)) -> AccountResponse:
    data = await get_live_account()
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

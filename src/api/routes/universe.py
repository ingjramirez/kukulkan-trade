"""Universe endpoints — base ticker universe grouped by sector."""

from fastapi import APIRouter

from config.universe import PORTFOLIO_B_UNIVERSE, SECTOR_MAP

router = APIRouter(prefix="/api/universe", tags=["universe"])


@router.get("/base")
async def get_base_universe() -> dict:
    """Return base universe tickers grouped by sector."""
    sectors: dict[str, list[str]] = {}
    for ticker in sorted(PORTFOLIO_B_UNIVERSE):
        sector = SECTOR_MAP.get(ticker, "Other")
        sectors.setdefault(sector, []).append(ticker)

    return {
        "total": len(PORTFOLIO_B_UNIVERSE),
        "sectors": {k: sorted(v) for k, v in sorted(sectors.items())},
    }

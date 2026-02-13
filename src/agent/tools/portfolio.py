"""Portfolio investigation tools for the agentic loop.

Tools for inspecting positions, P&L, and portfolio summary.
Data is pre-bound via functools.partial at registration time.
"""

from __future__ import annotations

from functools import partial

from config.universe import SECTOR_MAP
from src.agent.tools import ToolRegistry
from src.storage.database import Database


async def _get_current_positions(
    db: Database,
    tenant_id: str,
) -> list[dict]:
    """Get all Portfolio B positions with sector info."""
    positions = await db.get_positions("B", tenant_id=tenant_id)
    return [
        {
            "ticker": p.ticker,
            "shares": p.shares,
            "avg_price": round(p.avg_price, 2),
            "market_value": round(p.shares * p.avg_price, 2),
            "sector": SECTOR_MAP.get(p.ticker, "Unknown"),
        }
        for p in positions
    ]


async def _get_position_pnl(
    db: Database,
    tenant_id: str,
    current_prices: dict[str, float],
    ticker: str,
) -> dict:
    """Get P&L details for a specific position."""
    positions = await db.get_positions("B", tenant_id=tenant_id)
    pos = next((p for p in positions if p.ticker == ticker), None)
    if pos is None:
        return {"error": f"No position in {ticker}"}

    current_price = current_prices.get(ticker, pos.avg_price)
    pnl_pct = ((current_price - pos.avg_price) / pos.avg_price) * 100 if pos.avg_price > 0 else 0
    market_value = pos.shares * current_price

    # Check trailing stop status
    stops = await db.get_active_trailing_stops(tenant_id, "B")
    stop = next((s for s in stops if s.ticker == ticker), None)
    stop_info = None
    if stop:
        pct_from_stop = ((current_price - stop.stop_price) / current_price) * 100
        stop_info = {
            "stop_price": round(stop.stop_price, 2),
            "peak_price": round(stop.peak_price, 2),
            "trail_pct": stop.trail_pct,
            "pct_from_trigger": round(pct_from_stop, 1),
        }

    return {
        "ticker": ticker,
        "shares": pos.shares,
        "avg_price": round(pos.avg_price, 2),
        "current_price": round(current_price, 2),
        "market_value": round(market_value, 2),
        "pnl_pct": round(pnl_pct, 2),
        "sector": SECTOR_MAP.get(ticker, "Unknown"),
        "trailing_stop": stop_info,
    }


async def _get_portfolio_summary(
    db: Database,
    tenant_id: str,
    current_prices: dict[str, float],
) -> dict:
    """Get portfolio summary: cash, value, sector exposure."""
    portfolio = await db.get_portfolio("B", tenant_id=tenant_id)
    positions = await db.get_positions("B", tenant_id=tenant_id)

    cash = portfolio.cash if portfolio else 0
    sector_exposure: dict[str, float] = {}
    total_positions_value = 0.0

    for p in positions:
        price = current_prices.get(p.ticker, p.avg_price)
        value = p.shares * price
        total_positions_value += value
        sector = SECTOR_MAP.get(p.ticker, "Other")
        sector_exposure[sector] = sector_exposure.get(sector, 0) + value

    total_value = cash + total_positions_value

    # Convert to percentages
    sector_pct = {}
    if total_value > 0:
        for sector, val in sorted(sector_exposure.items(), key=lambda x: -x[1]):
            sector_pct[sector] = round(val / total_value * 100, 1)

    return {
        "cash": round(cash, 2),
        "cash_pct": round(cash / total_value * 100, 1) if total_value > 0 else 100.0,
        "positions_value": round(total_positions_value, 2),
        "total_value": round(total_value, 2),
        "position_count": len(positions),
        "sector_exposure": sector_pct,
    }


def register_portfolio_tools(
    registry: ToolRegistry,
    db: Database,
    tenant_id: str,
    current_prices: dict[str, float],
) -> None:
    """Register portfolio investigation tools with pre-bound context.

    Args:
        registry: ToolRegistry to register tools on.
        db: Database instance.
        tenant_id: Tenant UUID.
        current_prices: Dict of ticker → current price.
    """
    registry.register(
        name="get_current_positions",
        description="Get all Portfolio B positions with shares, avg price, market value, and sector.",
        input_schema={"type": "object", "properties": {}},
        handler=partial(_get_current_positions, db, tenant_id),
    )

    registry.register(
        name="get_position_pnl",
        description=(
            "Get detailed P&L for a specific position including current price, "
            "unrealized P&L %, and trailing stop status."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "ticker": {"type": "string", "description": "Ticker symbol to look up"},
            },
            "required": ["ticker"],
        },
        handler=partial(_get_position_pnl, db, tenant_id, current_prices),
    )

    registry.register(
        name="get_portfolio_summary",
        description="Get Portfolio B summary: cash, total value, position count, and sector exposure breakdown.",
        input_schema={"type": "object", "properties": {}},
        handler=partial(_get_portfolio_summary, db, tenant_id, current_prices),
    )

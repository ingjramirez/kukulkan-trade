"""Portfolio investigation tools for the agentic loop.

Phase 2 upgrade: 6 tools (3 upgraded + 3 new).
Old tool names kept as aliases for Phase 32 backward compatibility.
Data is pre-bound via functools.partial at registration time.
"""

from __future__ import annotations

from datetime import date, timedelta
from functools import partial

import pandas as pd

from config.universe import SECTOR_MAP
from src.agent.tools import ToolRegistry
from src.storage.database import Database

# ── 1. get_portfolio_state (upgrade of get_current_positions + get_portfolio_summary) ──


async def _get_portfolio_state(
    db: Database,
    tenant_id: str,
    current_prices: dict[str, float],
) -> dict:
    """Full Portfolio B state: positions with P&L, cash, sector exposure."""
    portfolio = await db.get_portfolio("B", tenant_id=tenant_id)
    positions = await db.get_positions("B", tenant_id=tenant_id)
    stops = await db.get_active_trailing_stops(tenant_id, "B")
    stop_map = {s.ticker: s for s in stops}

    cash = portfolio.cash if portfolio else 0
    sector_exposure: dict[str, float] = {}
    total_positions_value = 0.0
    pos_list = []

    for p in positions:
        price = current_prices.get(p.ticker, p.avg_price)
        value = p.shares * price
        total_positions_value += value
        sector = SECTOR_MAP.get(p.ticker, "Other")
        sector_exposure[sector] = sector_exposure.get(sector, 0) + value

        pnl_pct = ((price - p.avg_price) / p.avg_price) * 100 if p.avg_price > 0 else 0
        pos_entry: dict = {
            "ticker": p.ticker,
            "shares": p.shares,
            "avg_price": round(p.avg_price, 2),
            "current_price": round(price, 2),
            "market_value": round(value, 2),
            "pnl_pct": round(pnl_pct, 2),
            "sector": sector,
        }

        stop = stop_map.get(p.ticker)
        if stop:
            pct_from_stop = ((price - stop.stop_price) / price) * 100 if price > 0 else 0
            pos_entry["trailing_stop"] = {
                "stop_price": round(stop.stop_price, 2),
                "trail_pct": stop.trail_pct,
                "pct_from_trigger": round(pct_from_stop, 1),
            }

        pos_list.append(pos_entry)

    total_value = cash + total_positions_value

    # Sector percentages
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
        "positions": pos_list,
    }


# ── 2. get_position_detail (upgrade of get_position_pnl) ──


async def _get_position_detail(
    db: Database,
    tenant_id: str,
    current_prices: dict[str, float],
    ticker: str,
) -> dict:
    """Deep dive on a single position: P&L, trailing stop, recent trades."""
    positions = await db.get_positions("B", tenant_id=tenant_id)
    pos = next((p for p in positions if p.ticker == ticker), None)
    if pos is None:
        return {"error": f"No position in {ticker}"}

    current_price = current_prices.get(ticker, pos.avg_price)
    pnl_pct = ((current_price - pos.avg_price) / pos.avg_price) * 100 if pos.avg_price > 0 else 0
    market_value = pos.shares * current_price

    # Trailing stop
    stops = await db.get_active_trailing_stops(tenant_id, "B")
    stop = next((s for s in stops if s.ticker == ticker), None)
    stop_info = None
    if stop:
        pct_from_stop = ((current_price - stop.stop_price) / current_price) * 100 if current_price > 0 else 0
        stop_info = {
            "stop_price": round(stop.stop_price, 2),
            "peak_price": round(stop.peak_price, 2),
            "trail_pct": stop.trail_pct,
            "pct_from_trigger": round(pct_from_stop, 1),
        }

    # Recent trades for this ticker (last 30 days)
    since = date.today() - timedelta(days=30)
    trades = await db.get_trades("B", since=since, tenant_id=tenant_id)
    ticker_trades = [
        {
            "side": t.side,
            "shares": t.shares,
            "price": round(t.price, 2),
            "date": t.executed_at.strftime("%Y-%m-%d") if t.executed_at else "",
            "reason": (t.reason or "")[:100],
        }
        for t in trades
        if t.ticker == ticker
    ][:5]  # Last 5 trades

    return {
        "ticker": ticker,
        "shares": pos.shares,
        "avg_price": round(pos.avg_price, 2),
        "current_price": round(current_price, 2),
        "market_value": round(market_value, 2),
        "pnl_pct": round(pnl_pct, 2),
        "sector": SECTOR_MAP.get(ticker, "Unknown"),
        "trailing_stop": stop_info,
        "recent_trades": ticker_trades,
    }


# ── 3. get_portfolio_performance (new) ──


async def _get_portfolio_performance(
    db: Database,
    tenant_id: str,
    current_prices: dict[str, float],
    period: str = "30d",
) -> dict:
    """Portfolio performance metrics: return, drawdown, win rate."""
    # Parse period
    days_map = {"7d": 7, "14d": 14, "30d": 30, "60d": 60, "90d": 90}
    days = days_map.get(period, 30)
    since = date.today() - timedelta(days=days)

    snapshots = await db.get_snapshots("B", since=since, tenant_id=tenant_id)
    trades = await db.get_trades("B", since=since, tenant_id=tenant_id)

    # Current portfolio value
    portfolio = await db.get_portfolio("B", tenant_id=tenant_id)
    positions = await db.get_positions("B", tenant_id=tenant_id)
    cash = portfolio.cash if portfolio else 0
    positions_value = sum(p.shares * current_prices.get(p.ticker, p.avg_price) for p in positions)
    current_value = cash + positions_value

    result: dict = {
        "period": period,
        "days": days,
        "current_value": round(current_value, 2),
        "total_trades": len(trades),
    }

    if snapshots:
        start_value = snapshots[0].total_value
        if start_value > 0:
            result["period_return_pct"] = round((current_value - start_value) / start_value * 100, 2)

        # Drawdown: peak-to-trough from snapshots
        peak = start_value
        max_dd = 0.0
        for snap in snapshots:
            if snap.total_value > peak:
                peak = snap.total_value
            dd = (snap.total_value - peak) / peak * 100 if peak > 0 else 0
            if dd < max_dd:
                max_dd = dd
        result["max_drawdown_pct"] = round(max_dd, 2)

        # Daily returns
        returns = [s.daily_return_pct for s in snapshots if s.daily_return_pct is not None]
        if returns:
            result["avg_daily_return_pct"] = round(sum(returns) / len(returns), 3)
            result["best_day_pct"] = round(max(returns), 2)
            result["worst_day_pct"] = round(min(returns), 2)

    # Win rate from trades
    buys = [t for t in trades if t.side == "BUY"]
    sells = [t for t in trades if t.side == "SELL"]
    if sells:
        result["sells_count"] = len(sells)
    result["buys_count"] = len(buys)

    return result


# ── 4. get_historical_trades (new) ──


async def _get_historical_trades(
    db: Database,
    tenant_id: str,
    days: int = 30,
) -> dict:
    """Past trades with details. Returns last N days of trade history."""
    days = min(max(days, 1), 90)  # Clamp 1-90
    since = date.today() - timedelta(days=days)
    trades = await db.get_trades("B", since=since, tenant_id=tenant_id)

    trade_list = [
        {
            "ticker": t.ticker,
            "side": t.side,
            "shares": t.shares,
            "price": round(t.price, 2),
            "total": round(t.total, 2),
            "reason": (t.reason or "")[:100],
            "date": t.executed_at.strftime("%Y-%m-%d") if t.executed_at else "",
            "sector": SECTOR_MAP.get(t.ticker, "Unknown"),
        }
        for t in trades
    ]

    # Summary stats
    buy_total = sum(t.total for t in trades if t.side == "BUY")
    sell_total = sum(t.total for t in trades if t.side == "SELL")

    return {
        "days": days,
        "total_trades": len(trades),
        "trades": trade_list[:50],  # Cap at 50 most recent
        "buy_total_usd": round(buy_total, 2),
        "sell_total_usd": round(sell_total, 2),
        "net_flow_usd": round(sell_total - buy_total, 2),
    }


# ── 5. get_correlation_check (new) ──


async def _get_correlation_check(
    closes: pd.DataFrame,
    current_prices: dict[str, float],
    tickers: list[str] | None = None,
) -> dict:
    """Correlation matrix for specified tickers or all held positions."""
    if tickers is None or len(tickers) == 0:
        # Use all tickers that have current prices (i.e., in universe)
        tickers = [t for t in current_prices if t in closes.columns]

    # Filter to available columns
    available = [t for t in tickers if t in closes.columns]
    if len(available) < 2:
        return {"error": "Need at least 2 tickers with data for correlation check"}

    # Use last 60 trading days for correlation
    sub = closes[available].tail(60).dropna(axis=1, how="all")
    if sub.shape[1] < 2:
        return {"error": "Insufficient data for correlation after filtering"}

    # Compute returns correlation
    returns = sub.pct_change().dropna()
    if len(returns) < 10:
        return {"error": "Insufficient return data for correlation"}

    corr_matrix = returns.corr()

    # Find high correlations (>0.7)
    high_corrs = []
    cols = list(corr_matrix.columns)
    for i, t1 in enumerate(cols):
        for t2 in cols[i + 1 :]:
            val = corr_matrix.loc[t1, t2]
            if abs(val) > 0.7:
                high_corrs.append({"pair": f"{t1}/{t2}", "correlation": round(float(val), 3)})

    # Average pairwise correlation (diversification score)
    n = len(cols)
    if n > 1:
        total_corr = sum(abs(corr_matrix.loc[cols[i], cols[j]]) for i in range(n) for j in range(i + 1, n))
        pair_count = n * (n - 1) / 2
        avg_corr = total_corr / pair_count if pair_count > 0 else 0
    else:
        avg_corr = 0

    # Diversification score: 1.0 = perfectly uncorrelated, 0.0 = perfectly correlated
    diversification_score = round(1.0 - avg_corr, 3)

    return {
        "tickers": cols,
        "high_correlations": sorted(high_corrs, key=lambda x: -abs(x["correlation"])),
        "avg_pairwise_correlation": round(avg_corr, 3),
        "diversification_score": diversification_score,
    }


# ── 6. get_risk_assessment (new) ──


async def _get_risk_assessment(
    db: Database,
    tenant_id: str,
    current_prices: dict[str, float],
    closes: pd.DataFrame,
) -> dict:
    """Current risk exposure: sector concentration, stop distances, earnings risk."""
    portfolio = await db.get_portfolio("B", tenant_id=tenant_id)
    positions = await db.get_positions("B", tenant_id=tenant_id)
    stops = await db.get_active_trailing_stops(tenant_id, "B")
    stop_map = {s.ticker: s for s in stops}

    cash = portfolio.cash if portfolio else 0
    total_positions_value = 0.0
    sector_values: dict[str, float] = {}
    position_risks = []

    for p in positions:
        price = current_prices.get(p.ticker, p.avg_price)
        value = p.shares * price
        total_positions_value += value
        sector = SECTOR_MAP.get(p.ticker, "Other")
        sector_values[sector] = sector_values.get(sector, 0) + value

    total_value = cash + total_positions_value

    # Position-level risk
    for p in positions:
        price = current_prices.get(p.ticker, p.avg_price)
        value = p.shares * price
        weight = value / total_value * 100 if total_value > 0 else 0
        stop = stop_map.get(p.ticker)

        risk_entry: dict = {
            "ticker": p.ticker,
            "weight_pct": round(weight, 1),
            "has_trailing_stop": stop is not None,
        }
        if stop:
            pct_from_stop = ((price - stop.stop_price) / price) * 100 if price > 0 else 0
            risk_entry["pct_from_stop"] = round(pct_from_stop, 1)

        position_risks.append(risk_entry)

    # Sector concentration
    sector_pct = {}
    if total_value > 0:
        for sector, val in sorted(sector_values.items(), key=lambda x: -x[1]):
            sector_pct[sector] = round(val / total_value * 100, 1)

    # Portfolio volatility estimate (last 20 days)
    vol_estimate = None
    if positions and len(closes) >= 20:
        held_tickers = [p.ticker for p in positions if p.ticker in closes.columns]
        if held_tickers:
            sub = closes[held_tickers].tail(20).pct_change().dropna()
            if len(sub) >= 5:
                # Weight by position value
                weights = {}
                for p in positions:
                    if p.ticker in held_tickers:
                        price = current_prices.get(p.ticker, p.avg_price)
                        weights[p.ticker] = p.shares * price
                total_w = sum(weights.values())
                if total_w > 0:
                    weighted_var = 0.0
                    for t in held_tickers:
                        w = weights.get(t, 0) / total_w
                        if t in sub.columns:
                            weighted_var += (w * sub[t].std()) ** 2
                    vol_estimate = round(float(weighted_var**0.5) * (252**0.5) * 100, 1)

    # Positions without stops
    unprotected = [p.ticker for p in positions if p.ticker not in stop_map]

    return {
        "total_value": round(total_value, 2),
        "cash_pct": round(cash / total_value * 100, 1) if total_value > 0 else 100.0,
        "equity_invested_pct": round(total_positions_value / total_value * 100, 1) if total_value > 0 else 0.0,
        "position_count": len(positions),
        "sector_concentration": sector_pct,
        "largest_position": max(position_risks, key=lambda x: x["weight_pct"]) if position_risks else None,
        "positions_without_stops": unprotected,
        "annualized_vol_pct": vol_estimate,
        "position_risks": sorted(position_risks, key=lambda x: -x["weight_pct"]),
    }


# ── Legacy aliases (Phase 32 backward compatibility) ───────────────────────────


async def _get_current_positions(
    db: Database,
    tenant_id: str,
) -> list[dict]:
    """Get all Portfolio B positions with sector info. (Legacy alias)"""
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
    """Get P&L details for a specific position. (Legacy alias → delegates to get_position_detail)"""
    return await _get_position_detail(db, tenant_id, current_prices, ticker)


async def _get_portfolio_summary(
    db: Database,
    tenant_id: str,
    current_prices: dict[str, float],
) -> dict:
    """Get portfolio summary. (Legacy alias → wraps get_portfolio_state)"""
    state = await _get_portfolio_state(db, tenant_id, current_prices)
    # Return the Phase 32 format (without positions list)
    return {
        "cash": state["cash"],
        "cash_pct": state["cash_pct"],
        "positions_value": state["positions_value"],
        "total_value": state["total_value"],
        "position_count": state["position_count"],
        "sector_exposure": state["sector_exposure"],
    }


# ── Registration ──────────────────────────────────────────────────────────────


def register_portfolio_tools(
    registry: ToolRegistry,
    db: Database,
    tenant_id: str,
    current_prices: dict[str, float],
    closes: pd.DataFrame | None = None,
) -> None:
    """Register portfolio investigation tools with pre-bound context.

    Args:
        registry: ToolRegistry to register tools on.
        db: Database instance.
        tenant_id: Tenant UUID.
        current_prices: Dict of ticker -> current price.
        closes: Close price DataFrame (needed for correlation + risk tools).
    """
    # ── Phase 2 tools ────────────────────────────────────────────────────────
    registry.register(
        name="get_portfolio_state",
        description=(
            "Get full Portfolio B state: all positions with P&L and trailing stops, "
            "cash, total value, and sector exposure. Use this for a complete overview."
        ),
        input_schema={"type": "object", "properties": {}},
        handler=partial(_get_portfolio_state, db, tenant_id, current_prices),
    )

    registry.register(
        name="get_position_detail",
        description=(
            "Deep dive on a specific position: P&L, trailing stop status, and recent trade history for the ticker."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "ticker": {"type": "string", "description": "Ticker symbol to look up"},
            },
            "required": ["ticker"],
        },
        handler=partial(_get_position_detail, db, tenant_id, current_prices),
    )

    registry.register(
        name="get_portfolio_performance",
        description=(
            "Get portfolio performance metrics for a period: return, drawdown, daily return stats, and trade counts."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "period": {
                    "type": "string",
                    "enum": ["7d", "14d", "30d", "60d", "90d"],
                    "description": "Lookback period (default: 30d)",
                },
            },
        },
        handler=partial(_get_portfolio_performance, db, tenant_id, current_prices),
    )

    registry.register(
        name="get_historical_trades",
        description="Get past trade history with details. Returns last N days of Portfolio B trades.",
        input_schema={
            "type": "object",
            "properties": {
                "days": {
                    "type": "integer",
                    "description": "Number of days to look back (1-90, default: 30)",
                },
            },
        },
        handler=partial(_get_historical_trades, db, tenant_id),
    )

    if closes is not None:
        registry.register(
            name="get_correlation_check",
            description=(
                "Check correlation between tickers. Returns pairwise correlations, "
                "high-correlation pairs (>0.7), and a diversification score."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "tickers": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Tickers to check correlation for (default: all held positions)",
                    },
                },
            },
            handler=partial(_get_correlation_check, closes, current_prices),
        )

        registry.register(
            name="get_risk_assessment",
            description=(
                "Get portfolio risk assessment: sector concentration, position weights, "
                "trailing stop distances, unprotected positions, and volatility estimate."
            ),
            input_schema={"type": "object", "properties": {}},
            handler=partial(_get_risk_assessment, db, tenant_id, current_prices, closes),
        )

    # ── Phase 32 aliases (backward compatibility) ────────────────────────────
    registry.register(
        name="get_current_positions",
        description="[Alias for get_portfolio_state] Get all Portfolio B positions.",
        input_schema={"type": "object", "properties": {}},
        handler=partial(_get_current_positions, db, tenant_id),
    )

    registry.register(
        name="get_position_pnl",
        description="[Alias for get_position_detail] Get P&L for a specific position.",
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
        description="[Alias for get_portfolio_state] Get Portfolio B summary.",
        input_schema={"type": "object", "properties": {}},
        handler=partial(_get_portfolio_summary, db, tenant_id, current_prices),
    )

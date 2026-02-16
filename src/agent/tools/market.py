"""Market data investigation tools for the agentic loop.

Phase 2 upgrade: 4 tools (2 upgraded + 2 new).
Old tool names kept as aliases for Phase 32 backward compatibility.
Pre-fetched data is bound via functools.partial at registration.
"""

from __future__ import annotations

from functools import partial

import pandas as pd
import structlog

from config.universe import SECTOR_ETF_MAP, classify_instrument
from src.agent.tools import ToolRegistry
from src.analysis.technical import compute_all_indicators
from src.storage.database import Database

log = structlog.get_logger()

# ── 1. get_batch_technicals (replaces get_price_and_technicals, handles 5-20 tickers) ──


async def _get_batch_technicals(
    closes: pd.DataFrame,
    tickers: list[str],
) -> dict:
    """Bulk price and technicals for multiple tickers (5-20 per call)."""
    if not tickers:
        return {"error": "No tickers provided"}

    # Cap at 20 tickers per call
    tickers = tickers[:20]
    results = []

    for ticker in tickers:
        if ticker not in closes.columns:
            results.append({"ticker": ticker, "error": "Not found in market data"})
            continue

        series = closes[ticker].dropna()
        if len(series) < 2:
            results.append({"ticker": ticker, "error": "Insufficient data"})
            continue

        price = float(series.iloc[-1])
        pct_1d = ((series.iloc[-1] / series.iloc[-2]) - 1) * 100 if len(series) >= 2 else 0
        pct_5d = ((series.iloc[-1] / series.iloc[-6]) - 1) * 100 if len(series) >= 6 else 0
        pct_20d = ((series.iloc[-1] / series.iloc[-21]) - 1) * 100 if len(series) >= 21 else 0

        entry: dict = {
            "ticker": ticker,
            "price": round(price, 2),
            "change_1d_pct": round(pct_1d, 2),
            "change_5d_pct": round(pct_5d, 2),
            "change_20d_pct": round(pct_20d, 2),
            "instrument_type": classify_instrument(ticker).value,
        }

        # Compute technicals if enough data
        if len(series) >= 50:
            try:
                ind = compute_all_indicators(series)
                latest = ind.iloc[-1]
                entry["rsi_14"] = round(float(latest["rsi_14"]), 1) if pd.notna(latest["rsi_14"]) else None
                entry["macd"] = round(float(latest["macd"]), 2) if pd.notna(latest["macd"]) else None
                entry["sma_20"] = round(float(latest["sma_20"]), 2) if pd.notna(latest["sma_20"]) else None
                entry["sma_50"] = round(float(latest["sma_50"]), 2) if pd.notna(latest["sma_50"]) else None
            except (ValueError, KeyError, IndexError) as e:
                log.debug("batch_technicals_indicator_failed", ticker=ticker, error=str(e))

        results.append(entry)

    return {"tickers_requested": len(tickers), "results": results}


# ── 2. get_sector_heatmap (extracted from get_market_context, more detail) ──


async def _get_sector_heatmap(
    closes: pd.DataFrame,
) -> dict:
    """Full sector rotation signals: 1d/5d/20d returns and RSI for each sector ETF."""
    sectors = {}

    for sector, etf in SECTOR_ETF_MAP.items():
        if etf not in closes.columns:
            continue
        s = closes[etf].dropna()
        if len(s) < 2:
            continue

        entry: dict = {"etf": etf}
        entry["price"] = round(float(s.iloc[-1]), 2)

        if len(s) >= 2:
            entry["change_1d_pct"] = round(((s.iloc[-1] / s.iloc[-2]) - 1) * 100, 2)
        if len(s) >= 6:
            entry["change_5d_pct"] = round(((s.iloc[-1] / s.iloc[-6]) - 1) * 100, 2)
        if len(s) >= 21:
            entry["change_20d_pct"] = round(((s.iloc[-1] / s.iloc[-21]) - 1) * 100, 2)

        # RSI for the sector ETF
        if len(s) >= 50:
            try:
                ind = compute_all_indicators(s)
                rsi = ind["rsi_14"].iloc[-1]
                entry["rsi_14"] = round(float(rsi), 1) if pd.notna(rsi) else None
            except (ValueError, KeyError, IndexError) as e:
                log.debug("sector_heatmap_rsi_failed", sector=sector, etf=etf, error=str(e))

        sectors[sector] = entry

    # Sort by 5-day return (best performers first)
    sorted_sectors = dict(sorted(sectors.items(), key=lambda x: x[1].get("change_5d_pct", 0), reverse=True))

    return {"sectors": sorted_sectors, "sector_count": len(sorted_sectors)}


# ── 3. get_market_overview (rename of get_market_context) ──


async def _get_market_overview(
    closes: pd.DataFrame,
    vix: float | None,
    yield_curve: float | None,
    regime: str | None,
) -> dict:
    """Broad market overview: SPY, VIX, yield curve, regime, and sector heatmap."""
    result: dict = {
        "regime": regime or "Unknown",
        "vix": round(vix, 1) if vix is not None else None,
        "yield_curve_10y_2y": round(yield_curve, 2) if yield_curve is not None else None,
    }

    # SPY performance
    if "SPY" in closes.columns:
        spy = closes["SPY"].dropna()
        if len(spy) >= 2:
            result["spy_price"] = round(float(spy.iloc[-1]), 2)
            result["spy_1d_pct"] = round(((spy.iloc[-1] / spy.iloc[-2]) - 1) * 100, 2)
        if len(spy) >= 6:
            result["spy_5d_pct"] = round(((spy.iloc[-1] / spy.iloc[-6]) - 1) * 100, 2)
        if len(spy) >= 21:
            result["spy_20d_pct"] = round(((spy.iloc[-1] / spy.iloc[-21]) - 1) * 100, 2)

    # Sector heatmap (1-week returns for sector ETFs)
    sector_returns: dict[str, float] = {}
    for sector, etf in SECTOR_ETF_MAP.items():
        if etf in closes.columns:
            s = closes[etf].dropna()
            if len(s) >= 6:
                ret = ((s.iloc[-1] / s.iloc[-6]) - 1) * 100
                sector_returns[sector] = round(ret, 2)

    if sector_returns:
        result["sector_heatmap_1w"] = dict(sorted(sector_returns.items(), key=lambda x: -x[1]))

    return result


# ── 4. get_earnings_calendar (new) ──


async def _get_earnings_calendar(
    db: Database,
    held_tickers: list[str],
    tickers: list[str] | None = None,
    days_ahead: int = 14,
) -> dict:
    """Upcoming earnings for specified tickers or held positions."""
    lookup_tickers = tickers if tickers else held_tickers
    if not lookup_tickers:
        return {"earnings": [], "message": "No tickers to check"}

    days_ahead = min(max(days_ahead, 1), 30)  # Clamp 1-30
    earnings = await db.get_upcoming_earnings(lookup_tickers, days_ahead=days_ahead)

    entries = []
    for e in earnings:
        entry: dict = {
            "ticker": e.ticker,
            "earnings_date": e.earnings_date.isoformat() if e.earnings_date else "",
            "is_held": e.ticker in held_tickers,
        }
        # Days until
        if e.earnings_date:
            from datetime import date

            days_until = (e.earnings_date - date.today()).days
            entry["days_until"] = days_until
        entries.append(entry)

    return {
        "days_ahead": days_ahead,
        "earnings_count": len(entries),
        "earnings": sorted(entries, key=lambda x: x.get("days_until", 999)),
    }


# ── Legacy aliases (Phase 32 backward compatibility) ─────────────────────────


async def _get_price_and_technicals(
    closes: pd.DataFrame,
    ticker: str,
) -> dict:
    """Get price and technicals for a single ticker. (Legacy alias → delegates to batch.)"""
    result = await _get_batch_technicals(closes, [ticker])
    if result.get("results"):
        return result["results"][0]
    return {"error": f"Ticker {ticker} not found"}


async def _get_market_context(
    closes: pd.DataFrame,
    vix: float | None,
    yield_curve: float | None,
    regime: str | None,
) -> dict:
    """Get broad market context. (Legacy alias → delegates to get_market_overview.)"""
    return await _get_market_overview(closes, vix, yield_curve, regime)


# ── Registration ──────────────────────────────────────────────────────────────


def register_market_tools(
    registry: ToolRegistry,
    closes: pd.DataFrame,
    vix: float | None = None,
    yield_curve: float | None = None,
    regime: str | None = None,
    db: Database | None = None,
    held_tickers: list[str] | None = None,
) -> None:
    """Register market data tools with pre-fetched data.

    Args:
        registry: ToolRegistry to register tools on.
        closes: Close price DataFrame.
        vix: Current VIX value.
        yield_curve: 10Y-2Y yield curve spread.
        regime: Current market regime string.
        db: Database instance (needed for earnings calendar).
        held_tickers: List of currently held tickers (for earnings context).
    """
    # ── Phase 2 tools ────────────────────────────────────────────────────────
    registry.register(
        name="get_batch_technicals",
        description=(
            "Bulk price and technical indicators for 1-20 tickers: price, 1d/5d/20d returns, RSI, MACD, SMA20/50."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "tickers": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of ticker symbols (1-20)",
                },
            },
            "required": ["tickers"],
        },
        handler=partial(_get_batch_technicals, closes),
    )

    registry.register(
        name="get_sector_heatmap",
        description=(
            "Full sector rotation signals: 1d/5d/20d returns and RSI for each sector ETF. Sorted by 5-day performance."
        ),
        input_schema={"type": "object", "properties": {}},
        handler=partial(_get_sector_heatmap, closes),
    )

    registry.register(
        name="get_market_overview",
        description="Broad market overview: regime, VIX, yield curve, SPY stats, and sector 1-week heatmap.",
        input_schema={"type": "object", "properties": {}},
        handler=partial(_get_market_overview, closes, vix, yield_curve, regime),
    )

    if db is not None:
        registry.register(
            name="get_earnings_calendar",
            description=(
                "Check upcoming earnings dates. Defaults to held positions. "
                "Shows days until earnings and whether ticker is held."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "tickers": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Tickers to check (default: held positions)",
                    },
                    "days_ahead": {
                        "type": "integer",
                        "description": "Days ahead to look (1-30, default: 14)",
                    },
                },
            },
            handler=partial(_get_earnings_calendar, db, held_tickers or []),
        )

    # ── Phase 32 aliases (backward compatibility) ────────────────────────────
    registry.register(
        name="get_price_and_technicals",
        description="[Alias for get_batch_technicals] Get technicals for a single ticker.",
        input_schema={
            "type": "object",
            "properties": {
                "ticker": {"type": "string", "description": "Ticker symbol"},
            },
            "required": ["ticker"],
        },
        handler=partial(_get_price_and_technicals, closes),
    )

    registry.register(
        name="get_market_context",
        description="[Alias for get_market_overview] Get broad market overview.",
        input_schema={"type": "object", "properties": {}},
        handler=partial(_get_market_context, closes, vix, yield_curve, regime),
    )

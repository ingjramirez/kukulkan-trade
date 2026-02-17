"""Market data investigation tools for the agentic loop.

Phase 2 upgrade: 4 tools (2 upgraded + 2 new).
Old tool names kept as aliases for Phase 32 backward compatibility.
Pre-fetched data is bound via functools.partial at registration.
"""

from __future__ import annotations

import asyncio
from functools import partial

import pandas as pd
import structlog
import yfinance as yf

from config.universe import FULL_UNIVERSE, SECTOR_ETF_MAP, classify_instrument
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


# ── 5. search_ticker_info (research a ticker outside the universe) ────────────


def _yf_lookup(ticker: str) -> dict:
    """Synchronous yfinance lookup — called via asyncio.to_thread."""
    yf_ticker = yf.Ticker(ticker)
    info = yf_ticker.info

    if not info or info.get("regularMarketPrice") is None:
        return {"found": False}

    # Fetch 3 months of history for price changes + RSI
    try:
        history = yf_ticker.history(period="3mo")
    except Exception:
        history = None

    return {"found": True, "info": info, "history": history}


async def _search_ticker_info(
    db: Database | None,
    tenant_id: str,
    ticker: str,
) -> dict:
    """Quick research lookup for a ticker the agent is considering.

    READ-ONLY — does NOT propose the ticker. Use discover_ticker for that.
    Fetches live data from yfinance: price, market cap, volume, sector, RSI,
    price changes, and checks universe/discovery status.
    """
    from src.agent.ticker_discovery import MIN_AVG_VOLUME, MIN_MARKET_CAP

    ticker = ticker.upper().strip()
    if not ticker:
        return {"ticker": "", "valid": False, "error": "No ticker provided"}

    in_universe = ticker in FULL_UNIVERSE

    # Check discovery status in DB
    previously_discovered = False
    discovery_status = None
    if db is not None:
        existing = await db.get_discovered_ticker(ticker, tenant_id=tenant_id)
        if existing:
            previously_discovered = True
            discovery_status = existing.status

    # Fetch from yfinance in a thread (sync SDK)
    try:
        data = await asyncio.to_thread(_yf_lookup, ticker)
    except Exception as e:
        return {"ticker": ticker, "valid": False, "error": f"yfinance lookup failed: {e}"}

    if not data.get("found"):
        return {"ticker": ticker, "valid": False, "error": "Ticker not found on yfinance"}

    info = data["info"]
    market_cap = info.get("marketCap", 0) or 0
    avg_volume = info.get("averageVolume", 0) or 0

    # Minimum checks (same thresholds as TickerDiscovery)
    meets_minimums = market_cap >= MIN_MARKET_CAP and avg_volume >= MIN_AVG_VOLUME
    disqualify_reason = None
    if not meets_minimums:
        reasons = []
        if market_cap < MIN_MARKET_CAP:
            reasons.append(f"Market cap ${market_cap / 1e9:.1f}B < ${MIN_MARKET_CAP / 1e9:.0f}B min")
        if avg_volume < MIN_AVG_VOLUME:
            reasons.append(f"Avg volume {avg_volume:,.0f} < {MIN_AVG_VOLUME:,.0f} min")
        disqualify_reason = "; ".join(reasons)

    result: dict = {
        "ticker": ticker,
        "valid": True,
        "name": info.get("shortName", ticker),
        "sector": info.get("sector", "Unknown"),
        "industry": info.get("industry", "Unknown"),
        "market_cap": market_cap,
        "market_cap_display": f"${market_cap / 1e9:.1f}B" if market_cap else "N/A",
        "avg_volume": avg_volume,
        "price": info.get("regularMarketPrice"),
        "pe_ratio": info.get("trailingPE"),
        "52w_high": info.get("fiftyTwoWeekHigh"),
        "52w_low": info.get("fiftyTwoWeekLow"),
        "in_universe": in_universe,
        "previously_discovered": previously_discovered,
        "discovery_status": discovery_status,
        "meets_minimums": meets_minimums,
        "disqualify_reason": disqualify_reason,
    }

    # Price changes + RSI from history
    history = data.get("history")
    if history is not None and len(history) >= 2:
        closes_series = history["Close"]
        if len(closes_series) >= 2:
            result["change_1d_pct"] = round(((closes_series.iloc[-1] / closes_series.iloc[-2]) - 1) * 100, 2)
        if len(closes_series) >= 6:
            result["change_5d_pct"] = round(((closes_series.iloc[-1] / closes_series.iloc[-6]) - 1) * 100, 2)
        if len(closes_series) >= 21:
            result["change_20d_pct"] = round(((closes_series.iloc[-1] / closes_series.iloc[-21]) - 1) * 100, 2)

        # RSI + technicals via existing compute_all_indicators
        if len(closes_series) >= 50:
            try:
                ind = compute_all_indicators(closes_series)
                latest = ind.iloc[-1]
                result["rsi_14"] = round(float(latest["rsi_14"]), 1) if pd.notna(latest["rsi_14"]) else None
            except (ValueError, KeyError, IndexError):
                pass

        # Distance from 52-week high
        if result.get("52w_high") and result.get("price"):
            result["distance_from_52w_high_pct"] = round(
                ((result["price"] - result["52w_high"]) / result["52w_high"]) * 100, 1
            )

    return result


# ── Registration ──────────────────────────────────────────────────────────────


def register_market_tools(
    registry: ToolRegistry,
    closes: pd.DataFrame,
    vix: float | None = None,
    yield_curve: float | None = None,
    regime: str | None = None,
    db: Database | None = None,
    held_tickers: list[str] | None = None,
    tenant_id: str = "default",
) -> None:
    """Register market data tools with pre-fetched data.

    Args:
        registry: ToolRegistry to register tools on.
        closes: Close price DataFrame.
        vix: Current VIX value.
        yield_curve: 10Y-2Y yield curve spread.
        regime: Current market regime string.
        db: Database instance (needed for earnings calendar + discovery).
        held_tickers: List of currently held tickers (for earnings context).
        tenant_id: Tenant UUID (for discovery status checks).
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

    # ── Discovery tools ──────────────────────────────────────────────────────
    registry.register(
        name="search_ticker_info",
        description=(
            "Research a ticker outside the current universe. Returns price, market cap, "
            "volume, sector, RSI, price changes, and whether it meets discovery minimums. "
            "READ-ONLY — does not propose the ticker. Use discover_ticker to propose."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "ticker": {"type": "string", "description": "Ticker symbol to research (e.g., ANET, PLTR)"},
            },
            "required": ["ticker"],
        },
        handler=partial(_search_ticker_info, db, tenant_id),
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

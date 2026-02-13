"""Market data investigation tools for the agentic loop.

Tools for checking price/technicals and market context.
Pre-fetched data is bound via functools.partial at registration.
"""

from __future__ import annotations

from functools import partial

import pandas as pd

from config.universe import SECTOR_ETF_MAP
from src.agent.tools import ToolRegistry
from src.analysis.technical import compute_all_indicators


async def _get_price_and_technicals(
    closes: pd.DataFrame,
    ticker: str,
) -> dict:
    """Get price, returns, and technical indicators for a ticker."""
    if ticker not in closes.columns:
        return {"error": f"Ticker {ticker} not found in market data"}

    series = closes[ticker].dropna()
    if len(series) < 2:
        return {"error": f"Insufficient data for {ticker}"}

    price = float(series.iloc[-1])
    pct_1d = ((series.iloc[-1] / series.iloc[-2]) - 1) * 100 if len(series) >= 2 else 0
    pct_5d = ((series.iloc[-1] / series.iloc[-6]) - 1) * 100 if len(series) >= 6 else 0
    pct_20d = ((series.iloc[-1] / series.iloc[-21]) - 1) * 100 if len(series) >= 21 else 0

    result: dict = {
        "ticker": ticker,
        "price": round(price, 2),
        "change_1d_pct": round(pct_1d, 2),
        "change_5d_pct": round(pct_5d, 2),
        "change_20d_pct": round(pct_20d, 2),
    }

    # Compute technicals if enough data
    if len(series) >= 50:
        try:
            ind = compute_all_indicators(series)
            latest = ind.iloc[-1]
            result["rsi_14"] = round(float(latest["rsi_14"]), 1) if pd.notna(latest["rsi_14"]) else None
            result["macd"] = round(float(latest["macd"]), 2) if pd.notna(latest["macd"]) else None
            result["sma_20"] = round(float(latest["sma_20"]), 2) if pd.notna(latest["sma_20"]) else None
            result["sma_50"] = round(float(latest["sma_50"]), 2) if pd.notna(latest["sma_50"]) else None
        except Exception:
            pass

    return result


async def _get_market_context(
    closes: pd.DataFrame,
    vix: float | None,
    yield_curve: float | None,
    regime: str | None,
) -> dict:
    """Get broad market context: SPY, VIX, yield curve, sector heatmap."""
    result: dict = {
        "regime": regime or "Unknown",
        "vix": round(vix, 1) if vix is not None else None,
        "yield_curve_10y_2y": round(yield_curve, 2) if yield_curve is not None else None,
    }

    # SPY performance
    if "SPY" in closes.columns:
        spy = closes["SPY"].dropna()
        if len(spy) >= 6:
            result["spy_price"] = round(float(spy.iloc[-1]), 2)
            result["spy_1d_pct"] = round(((spy.iloc[-1] / spy.iloc[-2]) - 1) * 100, 2)
            result["spy_5d_pct"] = round(((spy.iloc[-1] / spy.iloc[-6]) - 1) * 100, 2)

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


def register_market_tools(
    registry: ToolRegistry,
    closes: pd.DataFrame,
    vix: float | None = None,
    yield_curve: float | None = None,
    regime: str | None = None,
) -> None:
    """Register market data tools with pre-fetched data.

    Args:
        registry: ToolRegistry to register tools on.
        closes: Close price DataFrame.
        vix: Current VIX value.
        yield_curve: 10Y-2Y yield curve spread.
        regime: Current market regime string.
    """
    registry.register(
        name="get_price_and_technicals",
        description="Get current price, 1d/5d/20d returns, RSI, MACD, SMA20/50 for a specific ticker.",
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
        description="Get broad market overview: SPY performance, VIX, yield curve, regime, and sector 1-week heatmap.",
        input_schema={"type": "object", "properties": {}},
        handler=partial(_get_market_context, closes, vix, yield_curve, regime),
    )

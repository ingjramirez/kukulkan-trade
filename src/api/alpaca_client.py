"""Cached async wrapper around the sync Alpaca trading client."""

import asyncio
import time
from typing import Any

import structlog

from config.settings import settings

log = structlog.get_logger()

_cache: dict = {}
_CACHE_TTL = 30  # seconds

# Separate cache for portfolio history (keyed by params)
_history_cache: dict[str, Any] = {}
_history_cache_ts: dict[str, float] = {}


async def get_live_account() -> dict | None:
    """Fetch account + positions from Alpaca with 30s TTL cache."""
    if not settings.alpaca.api_key:
        return None

    now = time.monotonic()
    if "account" in _cache and now - _cache["ts"] < _CACHE_TTL:
        return _cache["account"]

    try:
        data = await asyncio.to_thread(_fetch_from_alpaca)
        _cache["account"] = data
        _cache["ts"] = now
        return data
    except Exception as e:
        log.warning("alpaca_account_fetch_failed", error=str(e))
        return _cache.get("account")


def _fetch_from_alpaca() -> dict:
    """Sync Alpaca calls (run in thread)."""
    from alpaca.trading.client import TradingClient

    client = TradingClient(
        api_key=settings.alpaca.api_key,
        secret_key=settings.alpaca.secret_key,
        paper=settings.alpaca.paper,
    )
    account = client.get_account()
    positions = client.get_all_positions()

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


async def get_portfolio_history(
    period: str = "1D",
    timeframe: str = "5Min",
    extended_hours: bool = False,
) -> dict | None:
    """Fetch portfolio history from Alpaca with 30s TTL cache.

    Args:
        period: Time period (1D, 1W, 1M, 3M, 1A).
        timeframe: Bar size (1Min, 5Min, 15Min, 1H, 1D).
        extended_hours: Include pre/post market data.

    Returns:
        Dict with timestamps, equity, profit_loss, profit_loss_pct,
        base_value, and timeframe — or None on failure.
    """
    if not settings.alpaca.api_key:
        return None

    cache_key = f"{period}:{timeframe}:{extended_hours}"
    now = time.monotonic()
    if cache_key in _history_cache and now - _history_cache_ts.get(cache_key, 0) < _CACHE_TTL:
        return _history_cache[cache_key]

    try:
        data = await asyncio.to_thread(
            _fetch_portfolio_history,
            period,
            timeframe,
            extended_hours,
        )
        _history_cache[cache_key] = data
        _history_cache_ts[cache_key] = now
        return data
    except Exception as e:
        log.warning("alpaca_history_fetch_failed", period=period, timeframe=timeframe, error=str(e))
        return _history_cache.get(cache_key)


def _fetch_portfolio_history(
    period: str,
    timeframe: str,
    extended_hours: bool,
) -> dict:
    """Sync Alpaca portfolio history call (run in thread)."""
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import GetPortfolioHistoryRequest

    client = TradingClient(
        api_key=settings.alpaca.api_key,
        secret_key=settings.alpaca.secret_key,
        paper=settings.alpaca.paper,
    )
    req = GetPortfolioHistoryRequest(
        period=period,
        timeframe=timeframe,
        extended_hours=extended_hours,
    )
    result = client.get_portfolio_history(req)

    return {
        "timestamps": result.timestamp or [],
        "equity": result.equity or [],
        "profit_loss": result.profit_loss or [],
        "profit_loss_pct": result.profit_loss_pct or [],
        "base_value": float(result.base_value) if result.base_value else 0.0,
        "timeframe": result.timeframe or timeframe,
    }

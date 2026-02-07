"""Cached async wrapper around the sync Alpaca trading client."""

import asyncio
import time

from config.settings import settings

_cache: dict = {}
_CACHE_TTL = 30  # seconds


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
    except Exception:
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

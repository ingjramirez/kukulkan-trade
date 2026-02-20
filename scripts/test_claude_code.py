#!/usr/bin/env python3
"""End-to-end test of Claude Code CLI trading session.

Usage (on VPS):
    cd /opt/kukulkan-trade
    .venv/bin/python scripts/test_claude_code.py

Creates real session-state.json + context.md from the database,
then invokes `claude -p` with MCP tools.
"""
from __future__ import annotations

import asyncio
import json
import sys
from datetime import date
from pathlib import Path

# Ensure project root on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import structlog

from src.agent.claude_invoker import ClaudeInvoker, write_context_file, write_session_state
from src.storage.database import Database

log = structlog.get_logger()

WORKSPACE = Path(__file__).resolve().parent.parent / "data" / "agent-workspace"
DB_URL = "sqlite+aiosqlite:///data/kukulkan.db"


async def build_state_from_db() -> dict:
    """Fetch real data from database to build session state."""
    db = Database(DB_URL)
    await db.init_db()

    tenant_id = "default"
    today = date.today()

    # Get positions
    positions_raw = await db.get_positions("B", tenant_id=tenant_id)
    positions = []
    held_tickers: list[str] = []
    for p in positions_raw:
        if p.shares > 0:
            positions.append({
                "ticker": p.ticker,
                "shares": float(p.shares),
                "avg_price": float(p.avg_price),
                "market_value": float(p.market_value) if p.market_value else 0.0,
            })
            held_tickers.append(p.ticker)

    # Get latest snapshot for cash/total value
    snapshots = await db.get_snapshots("B", tenant_id=tenant_id)
    cash = 34000.0  # fallback
    total_value = 66800.0
    if snapshots:
        latest = snapshots[-1]
        cash = float(latest.cash)
        total_value = float(latest.total_value)

    # Get closes from yfinance (lightweight — last 60 days for a few tickers)
    closes_dict: dict = {}
    closes_index: list[str] = []
    current_prices: dict[str, float] = {}
    try:
        import yfinance as yf

        tickers = held_tickers + ["SPY", "QQQ", "GLD", "BTC-USD"]
        tickers = list(set(tickers))
        log.info("fetching_market_data", tickers=tickers)
        data = yf.download(tickers, period="60d", progress=False)
        if "Close" in data.columns.get_level_values(0) if data.columns.nlevels > 1 else "Close" in data.columns:
            closes = data["Close"] if data.columns.nlevels > 1 else data[["Close"]]
            closes_dict = {
                col: {str(k): v for k, v in closes[col].dropna().to_dict().items()}
                for col in closes.columns
            }
            closes_index = [str(idx) for idx in closes.index]
            for col in closes.columns:
                last = closes[col].dropna()
                if len(last) > 0:
                    current_prices[col] = float(last.iloc[-1])
    except Exception as e:
        log.warning("yfinance_failed", error=str(e))

    # Get VIX
    vix = None
    try:
        import yfinance as yf

        vix_data = yf.download("^VIX", period="5d", progress=False)
        if len(vix_data) > 0:
            vix = float(vix_data["Close"].iloc[-1].item())
    except Exception:
        pass

    # Fear & Greed
    fear_greed = None
    try:
        fg_row = await db.get_latest_sentiment(tenant_id, "fear_greed_index")
        if fg_row:
            fear_greed = {"value": fg_row.value, "classification": fg_row.classification}
    except Exception:
        pass

    # Get signal rankings
    signal_text = None
    try:
        signal_rows = await db.get_latest_signals(tenant_id)
        if signal_rows:
            from src.analysis.signal_engine import db_rows_to_signals, format_signals_for_agent

            signals = db_rows_to_signals(signal_rows)
            held_set = set(held_tickers)
            signal_text = format_signals_for_agent(signals, held_set)
    except Exception as e:
        log.debug("signal_fetch_failed", error=str(e))

    # Posture
    posture = "balanced"
    try:
        posture_row = await db.get_current_posture(tenant_id)
        if posture_row:
            posture = posture_row.effective_posture
    except Exception:
        pass

    # Regime
    regime = "unknown"
    try:
        from src.analysis.regime import RegimeClassifier

        import pandas as pd

        classifier = RegimeClassifier()
        if closes_dict and "SPY" in closes_dict:
            spy_series = pd.Series(closes_dict["SPY"], dtype=float)
            spy_series.index = pd.to_datetime(spy_series.index)
            regime_result = classifier.classify(spy_series, vix=vix)
            regime = regime_result.regime.value if hasattr(regime_result, "regime") else str(regime_result)
    except Exception as e:
        log.debug("regime_detection_failed", error=str(e))

    return {
        "tenant_id": tenant_id,
        "today": today,
        "closes_dict": closes_dict,
        "closes_index": closes_index,
        "current_prices": current_prices,
        "held_tickers": held_tickers,
        "positions": positions,
        "cash": cash,
        "total_value": total_value,
        "vix": vix,
        "yield_curve": None,
        "regime": regime,
        "fear_greed": fear_greed,
        "signal_text": signal_text,
        "posture": posture,
    }


async def main():
    print("=" * 60)
    print("  Kukulkan — Claude Code End-to-End Test")
    print("=" * 60)

    # 1. Build state from database
    print("\n[1/4] Fetching data from database + yfinance...")
    state = await build_state_from_db()

    print(f"  Positions: {len(state['positions'])} held")
    print(f"  Tickers in closes: {len(state['closes_dict'])}")
    print(f"  Current prices: {len(state['current_prices'])}")
    print(f"  VIX: {state['vix']}")
    print(f"  Regime: {state['regime']}")
    print(f"  Cash: ${state['cash']:,.2f}")
    print(f"  Total Value: ${state['total_value']:,.2f}")

    # 2. Write session-state.json
    print("\n[2/4] Writing session-state.json...")
    write_session_state(
        workspace=WORKSPACE,
        tenant_id=state["tenant_id"],
        closes_dict=state["closes_dict"],
        closes_index=state["closes_index"],
        current_prices=state["current_prices"],
        held_tickers=state["held_tickers"],
        vix=state["vix"],
        yield_curve=state["yield_curve"],
        regime=state["regime"],
        fear_greed=state["fear_greed"],
    )
    print(f"  Written: {WORKSPACE / 'session-state.json'}")

    # 3. Write context.md
    print("\n[3/4] Writing context.md...")
    pinned = f"## Current Posture: {state['posture'].capitalize()}\n"
    write_context_file(
        workspace=WORKSPACE,
        session_type="manual",
        today=state["today"],
        regime=state["regime"],
        vix=state["vix"],
        yield_curve=state["yield_curve"],
        cash=state["cash"],
        total_value=state["total_value"],
        positions=state["positions"],
        signal_text=state["signal_text"],
        fear_greed=state["fear_greed"],
        pinned_context=pinned,
    )
    print(f"  Written: {WORKSPACE / 'context.md'}")

    # 4. Invoke Claude Code
    print("\n[4/4] Invoking Claude Code CLI...")
    print("  (This may take 1-3 minutes)")
    print("-" * 60)

    invoker = ClaudeInvoker(
        workspace=WORKSPACE,
        timeout=300,  # 5 min for test
        max_turns=15,
        model="claude-sonnet-4-6",
    )

    result = await invoker.invoke(session_type="manual", today=state["today"])

    print("-" * 60)

    if result.error:
        print(f"\n  ERROR: {result.error}")
        return 1

    print(f"\n  Session ID: {result.session_id}")
    print(f"  Trades: {len(result.trades)}")
    print(f"  Posture: {result.posture}")
    print(f"  Trailing Stops: {len(result.trailing_stop_requests)}")

    # Print the full response
    print("\n" + "=" * 60)
    print("  FULL RESPONSE:")
    print("=" * 60)
    print(json.dumps(result.response, indent=2, default=str))

    if result.accumulated:
        print("\n" + "=" * 60)
        print("  ACCUMULATED STATE (from MCP):")
        print("=" * 60)
        print(json.dumps(result.accumulated, indent=2, default=str))

    print("\n  End-to-end test complete!")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

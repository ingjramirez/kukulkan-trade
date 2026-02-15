"""Action tools for the agentic loop — accumulate trade proposals and state.

Phase 2 upgrade: 5 action tools (2 upgraded + 3 new).
Uses an ActionState class (instantiated per run) to avoid module-level
globals that would contaminate multi-tenant runs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial
from typing import Any

from src.agent.tools import ToolRegistry


@dataclass
class ActionState:
    """Per-run state container for accumulated agent actions."""

    proposed_trades: list[dict] = field(default_factory=list)
    watchlist_updates: list[dict] = field(default_factory=list)
    memory_notes: list[dict] = field(default_factory=list)
    executed_trades: list[dict] = field(default_factory=list)
    trailing_stop_requests: list[dict] = field(default_factory=list)
    declared_posture: str | None = None

    def get_accumulated_state(self) -> dict:
        """Return all accumulated actions as a dict."""
        return {
            "trades": list(self.proposed_trades),
            "watchlist_updates": list(self.watchlist_updates),
            "memory_notes": list(self.memory_notes),
            "executed_trades": list(self.executed_trades),
            "trailing_stop_requests": list(self.trailing_stop_requests),
            "declared_posture": self.declared_posture,
        }

    def reset(self) -> None:
        """Clear all accumulated state."""
        self.proposed_trades.clear()
        self.watchlist_updates.clear()
        self.memory_notes.clear()
        self.executed_trades.clear()
        self.trailing_stop_requests.clear()
        self.declared_posture = None


# ── 1. execute_trade (new — direct execution with risk check) ────────────────


async def _execute_trade(
    state: ActionState,
    ticker: str,
    side: str,
    shares: float,
    reason: str = "",
    conviction: str = "medium",
) -> dict:
    """Submit a trade for execution.

    In the current implementation, this accumulates the trade for the orchestrator
    to execute after the agent loop completes. The orchestrator handles:
    1. RiskManager pre-trade checks
    2. Telegram approval for >10% of portfolio
    3. Alpaca execution + fill polling
    4. Result is returned in the next session (or via Telegram)

    Future: direct execution within the tool call with fill result.
    """
    if not ticker or not side:
        return {"error": "ticker and side are required"}

    side_upper = side.upper()
    if side_upper not in ("BUY", "SELL"):
        return {"error": f"Invalid side: {side}. Must be BUY or SELL."}

    if shares <= 0:
        return {"error": "shares must be positive"}

    trade = {
        "ticker": ticker.upper(),
        "side": side_upper,
        "shares": float(shares),
        "conviction": conviction,
        "reason": reason[:200],
    }

    state.executed_trades.append(trade)
    # Also add to proposed_trades for backward compatibility with orchestrator merge
    state.proposed_trades.append(
        {
            "ticker": trade["ticker"],
            "side": trade["side"],
            "weight": 0.10,  # Default weight — orchestrator computes actual from shares
            "conviction": trade["conviction"],
            "reason": trade["reason"],
            "shares_requested": trade["shares"],
        }
    )

    return {
        "status": "submitted",
        "ticker": trade["ticker"],
        "side": trade["side"],
        "shares": trade["shares"],
        "message": f"Trade submitted: {trade['side']} {trade['shares']} {trade['ticker']}. "
        f"Will execute after risk check and approval (if required).",
        "trades_accumulated": len(state.executed_trades),
    }


# ── 2. set_trailing_stop (new — explicit stop management) ───────────────────


async def _set_trailing_stop(
    state: ActionState,
    ticker: str,
    trail_pct: float,
    reason: str = "",
) -> dict:
    """Set or update a trailing stop for a position.

    The stop is accumulated and applied by the orchestrator after the agent loop.

    Args:
        state: ActionState instance.
        ticker: Ticker to set stop for.
        trail_pct: Trailing stop percentage (0.03 to 0.20, e.g., 0.07 = 7%).
        reason: Reason for the stop level.
    """
    if not ticker:
        return {"error": "ticker is required"}

    if trail_pct < 0.03 or trail_pct > 0.20:
        return {"error": f"trail_pct must be 0.03-0.20, got {trail_pct}"}

    request = {
        "ticker": ticker.upper(),
        "trail_pct": trail_pct,
        "reason": reason[:200],
    }
    state.trailing_stop_requests.append(request)

    return {
        "status": "ok",
        "ticker": request["ticker"],
        "trail_pct": trail_pct,
        "message": f"Trailing stop set: {request['ticker']} at {trail_pct * 100:.0f}%",
        "stops_accumulated": len(state.trailing_stop_requests),
    }


# ── 3. get_order_status (new — check recent order fills) ────────────────────


async def _get_order_status(
    state: ActionState,
    db: Any,
    tenant_id: str,
    ticker: str | None = None,
) -> dict:
    """Check status of recently submitted trades.

    If ticker is provided, shows status for that ticker.
    Otherwise shows all trades submitted in this session.
    """
    # Session trades (from this agent run)
    session_trades = state.executed_trades
    if ticker:
        ticker_upper = ticker.upper()
        session_trades = [t for t in session_trades if t["ticker"] == ticker_upper]

    # Also check recent DB trades (last 1 day) for fill info
    from datetime import date, timedelta

    since = date.today() - timedelta(days=1)
    recent_db_trades = await db.get_trades("B", since=since, tenant_id=tenant_id)

    recent_fills = []
    for t in recent_db_trades[:10]:
        if ticker and t.ticker != ticker.upper():
            continue
        recent_fills.append(
            {
                "ticker": t.ticker,
                "side": t.side,
                "shares": t.shares,
                "price": round(t.price, 2),
                "date": t.executed_at.strftime("%Y-%m-%d %H:%M") if t.executed_at else "",
            }
        )

    return {
        "pending_this_session": len(session_trades),
        "session_trades": [{"ticker": t["ticker"], "side": t["side"], "shares": t["shares"]} for t in session_trades],
        "recent_fills": recent_fills,
    }


# ── 4. save_observation (rename of save_memory_note) ────────────────────────


async def _save_observation(
    state: ActionState,
    key: str,
    content: str,
) -> dict:
    """Save an observation to persist across sessions.

    Use for theses, lessons, correlations, or any insight worth remembering.

    Args:
        state: ActionState instance.
        key: Note key (e.g. "thesis-tech", "lesson-timing", "correlation-xlk-nvda").
        content: Note content (max ~50 words).

    Returns:
        Confirmation.
    """
    state.memory_notes.append({"key": key, "content": content[:200]})
    return {
        "status": "ok",
        "key": key,
        "memory_notes_accumulated": len(state.memory_notes),
    }


# ── 5. declare_posture (new — risk posture declaration) ──────────────────────

_VALID_POSTURES = {"balanced", "defensive", "crisis", "aggressive"}


async def _declare_posture(
    state: ActionState,
    posture: str,
    reason: str = "",
) -> dict:
    """Declare the risk posture for this session.

    Valid postures: balanced, defensive, crisis, aggressive.
    Aggressive requires track record gate (50+ trades, >55% WR, positive alpha).

    Args:
        state: ActionState instance.
        posture: Posture level to declare.
        reason: Reason for the posture choice.
    """
    posture_lower = posture.lower().strip()
    if posture_lower not in _VALID_POSTURES:
        return {"error": f"Invalid posture: {posture}. Must be one of: {', '.join(sorted(_VALID_POSTURES))}"}

    state.declared_posture = posture_lower
    return {
        "status": "ok",
        "posture": posture_lower,
        "reason": reason[:200],
        "message": f"Posture set to {posture_lower}. Risk limits will be adjusted accordingly.",
        "note": "Aggressive posture requires track record gate — system will verify.",
    }


# ── Legacy functions (kept for backward compat) ─────────────────────────────


async def _propose_trades(
    state: ActionState,
    trades: list[dict],
) -> dict:
    """Accumulate trade proposals. (Legacy — use execute_trade instead.)"""
    for trade in trades:
        if not trade.get("ticker") or not trade.get("side"):
            continue
        state.proposed_trades.append(
            {
                "ticker": trade["ticker"].upper(),
                "side": trade["side"].upper(),
                "weight": trade.get("weight", 0.10),
                "conviction": trade.get("conviction", "medium"),
                "reason": trade.get("reason", ""),
            }
        )
    return {
        "status": "ok",
        "trades_accumulated": len(state.proposed_trades),
        "latest": [t["ticker"] for t in state.proposed_trades[-3:]],
    }


async def _update_watchlist(
    state: ActionState,
    updates: list[dict],
) -> dict:
    """Accumulate watchlist updates."""
    for update in updates:
        if not update.get("ticker") or not update.get("action"):
            continue
        state.watchlist_updates.append(
            {
                "action": update["action"].lower(),
                "ticker": update["ticker"].upper(),
                "reason": update.get("reason", ""),
                "conviction": update.get("conviction", "medium"),
                "target_entry": update.get("target_entry"),
            }
        )
    return {
        "status": "ok",
        "watchlist_updates_accumulated": len(state.watchlist_updates),
    }


async def _save_memory_note(
    state: ActionState,
    key: str,
    content: str,
) -> dict:
    """Accumulate a memory note. (Legacy alias → delegates to save_observation.)"""
    return await _save_observation(state, key, content)


# ── Registration ──────────────────────────────────────────────────────────────


def register_action_tools(
    registry: ToolRegistry,
    state: ActionState,
    db: Any | None = None,
    tenant_id: str = "default",
) -> None:
    """Register action tools with a per-run state container.

    Args:
        registry: ToolRegistry to register tools on.
        state: ActionState instance (created fresh per run).
        db: Database instance (for order status lookups).
        tenant_id: Tenant UUID.
    """
    # ── Phase 2 tools ────────────────────────────────────────────────────────
    registry.register(
        name="execute_trade",
        description=(
            "Submit a trade for execution. Specify ticker, side (BUY/SELL), shares, and reason. "
            "Trade goes through RiskManager checks. Large trades (>10% of portfolio) require Telegram approval."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "ticker": {"type": "string", "description": "Ticker symbol"},
                "side": {"type": "string", "enum": ["BUY", "SELL"], "description": "Buy or sell"},
                "shares": {"type": "number", "description": "Number of shares"},
                "reason": {"type": "string", "description": "Reason for the trade"},
                "conviction": {"type": "string", "enum": ["high", "medium", "low"]},
            },
            "required": ["ticker", "side", "shares"],
        },
        handler=partial(_execute_trade, state),
    )

    registry.register(
        name="set_trailing_stop",
        description=("Set or update a trailing stop for a position. Specify trail_pct as decimal (e.g., 0.07 for 7%)."),
        input_schema={
            "type": "object",
            "properties": {
                "ticker": {"type": "string", "description": "Ticker symbol"},
                "trail_pct": {
                    "type": "number",
                    "description": "Trailing stop % as decimal (0.03-0.20, e.g., 0.07 = 7%)",
                },
                "reason": {"type": "string", "description": "Reason for the stop level"},
            },
            "required": ["ticker", "trail_pct"],
        },
        handler=partial(_set_trailing_stop, state),
    )

    if db is not None:
        registry.register(
            name="get_order_status",
            description="Check status of recent trades. Shows pending session trades and recent fills.",
            input_schema={
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "Optional ticker to filter (default: all)",
                    },
                },
            },
            handler=partial(_get_order_status, state, db, tenant_id),
        )

    registry.register(
        name="save_observation",
        description="Save an insight or observation to persist across sessions. Use for theses, lessons, correlations.",
        input_schema={
            "type": "object",
            "properties": {
                "key": {"type": "string", "description": "Note key like 'thesis-tech' or 'lesson-timing'"},
                "content": {"type": "string", "description": "Note content (max ~50 words)"},
            },
            "required": ["key", "content"],
        },
        handler=partial(_save_observation, state),
    )

    registry.register(
        name="declare_posture",
        description=(
            "Declare your risk posture for this session. "
            "Options: balanced (default), defensive (reduce exposure), crisis (minimal risk), "
            "aggressive (max exposure — gated behind track record)."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "posture": {
                    "type": "string",
                    "enum": ["balanced", "defensive", "crisis", "aggressive"],
                    "description": "Risk posture level",
                },
                "reason": {"type": "string", "description": "Reason for the posture choice"},
            },
            "required": ["posture"],
        },
        handler=partial(_declare_posture, state),
    )

    registry.register(
        name="update_watchlist",
        description="Add or remove tickers from the watchlist for future monitoring.",
        input_schema={
            "type": "object",
            "properties": {
                "updates": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "action": {"type": "string", "enum": ["add", "remove"]},
                            "ticker": {"type": "string"},
                            "reason": {"type": "string"},
                            "conviction": {"type": "string", "enum": ["high", "medium", "low"]},
                            "target_entry": {"type": "number"},
                        },
                        "required": ["action", "ticker"],
                    },
                },
            },
            "required": ["updates"],
        },
        handler=partial(_update_watchlist, state),
    )

    # ── Phase 32 aliases (backward compatibility) ────────────────────────────
    registry.register(
        name="propose_trades",
        description="[Alias for execute_trade] Submit trade proposals (legacy format).",
        input_schema={
            "type": "object",
            "properties": {
                "trades": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "ticker": {"type": "string"},
                            "side": {"type": "string", "enum": ["BUY", "SELL"]},
                            "weight": {"type": "number"},
                            "conviction": {"type": "string", "enum": ["high", "medium", "low"]},
                            "reason": {"type": "string"},
                        },
                        "required": ["ticker", "side"],
                    },
                },
            },
            "required": ["trades"],
        },
        handler=partial(_propose_trades, state),
    )

    registry.register(
        name="save_memory_note",
        description="[Alias for save_observation] Save an observation to persist across sessions.",
        input_schema={
            "type": "object",
            "properties": {
                "key": {"type": "string", "description": "Note key"},
                "content": {"type": "string", "description": "Note content"},
            },
            "required": ["key", "content"],
        },
        handler=partial(_save_memory_note, state),
    )

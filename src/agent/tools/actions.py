"""Action tools for the agentic loop — accumulate trade proposals and state.

Phase 2 upgrade: 5 action tools (2 upgraded + 3 new).
Uses an ActionState class (instantiated per run) to avoid module-level
globals that would contaminate multi-tenant runs.

When executor and risk_manager are provided (MCP server / chat context),
execute_trade and set_trailing_stop execute directly instead of accumulating.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from functools import partial
from typing import Any

import structlog

from src.agent.tools import ToolRegistry

log = structlog.get_logger()


@dataclass
class ActionState:
    """Per-run state container for accumulated agent actions."""

    proposed_trades: list[dict] = field(default_factory=list)
    watchlist_updates: list[dict] = field(default_factory=list)
    memory_notes: list[dict] = field(default_factory=list)
    executed_trades: list[dict] = field(default_factory=list)
    trailing_stop_requests: list[dict] = field(default_factory=list)
    discovery_proposals: list[dict] = field(default_factory=list)
    declared_posture: str | None = None

    def get_accumulated_state(self) -> dict:
        """Return all accumulated actions as a dict."""
        return {
            "trades": list(self.proposed_trades),
            "watchlist_updates": list(self.watchlist_updates),
            "memory_notes": list(self.memory_notes),
            "executed_trades": list(self.executed_trades),
            "trailing_stop_requests": list(self.trailing_stop_requests),
            "discovery_proposals": list(self.discovery_proposals),
            "declared_posture": self.declared_posture,
        }

    def reset(self) -> None:
        """Clear all accumulated state."""
        self.proposed_trades.clear()
        self.watchlist_updates.clear()
        self.memory_notes.clear()
        self.executed_trades.clear()
        self.trailing_stop_requests.clear()
        self.discovery_proposals.clear()
        self.declared_posture = None


# ── Shared helper ─────────────────────────────────────────────────────────────


async def _fetch_current_price(ticker: str) -> float | None:
    """Fetch latest price via yfinance fast_info (thread-safe)."""
    import yfinance as yf

    def _get() -> float | None:
        t = yf.Ticker(ticker)
        return getattr(t.fast_info, "last_price", None) or getattr(t.fast_info, "previous_close", None)

    try:
        return await asyncio.to_thread(_get)
    except Exception:
        return None


# ── 1. execute_trade — direct execution when executor available ──────────────


async def _execute_trade(
    state: ActionState,
    executor: Any | None,
    risk_manager: Any | None,
    db: Any | None,
    tenant_id: str,
    current_prices: dict[str, float],
    ticker: str,
    side: str,
    shares: float,
    reason: str = "",
    conviction: str = "medium",
) -> dict:
    """Submit a trade for execution.

    When executor is provided (chat/MCP context), executes directly via
    Alpaca or PaperTrader with risk checks. Otherwise accumulates for the
    orchestrator to process post-agent-loop.
    """
    if not ticker or not side:
        return {"error": "ticker and side are required"}

    side_upper = side.upper()
    if side_upper not in ("BUY", "SELL"):
        return {"error": f"Invalid side: {side}. Must be BUY or SELL."}

    if shares <= 0:
        return {"error": "shares must be positive"}

    ticker_upper = ticker.upper()

    # ── Fallback: accumulate-only (orchestrator path) ─────────────────────
    if executor is None:
        trade = {
            "ticker": ticker_upper,
            "side": side_upper,
            "shares": float(shares),
            "conviction": conviction,
            "reason": reason[:200],
        }
        state.executed_trades.append(trade)
        state.proposed_trades.append(
            {
                "ticker": trade["ticker"],
                "side": trade["side"],
                "weight": 0.10,
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

    # ── Direct execution (chat/MCP path) ──────────────────────────────────
    from src.storage.models import OrderSide, PortfolioName, TradeSchema

    # 1. Resolve price
    price = current_prices.get(ticker_upper)
    if not price:
        price = await _fetch_current_price(ticker_upper)
    if not price:
        return {"error": f"Cannot determine current price for {ticker_upper}"}

    # 2. Build TradeSchema
    trade_schema = TradeSchema(
        portfolio=PortfolioName.B,
        ticker=ticker_upper,
        side=OrderSide(side_upper),
        shares=float(shares),
        price=price,
        reason=reason[:200] if reason else f"Chat trade: {side_upper} {ticker_upper}",
    )

    # 3. Risk check
    if risk_manager is not None and db is not None:
        positions = await db.get_positions("B", tenant_id=tenant_id)
        position_map = {p.ticker: float(p.shares) for p in positions}
        portfolio = await db.get_portfolio("B", tenant_id=tenant_id)
        portfolio_value = portfolio.total_value if portfolio else 0
        cash = portfolio.cash if portfolio else 0

        verdict = risk_manager.check_pre_trade([trade_schema], "B", position_map, current_prices, portfolio_value, cash)
        if verdict.blocked:
            blocked_reason = verdict.blocked[0][1]
            return {"status": "blocked", "ticker": ticker_upper, "reason": blocked_reason}

    # 3b. Cash guard for BUYs — prevent spending more than Portfolio B's allocated cash
    if side_upper == "BUY" and db is not None:
        portfolio_b = await db.get_portfolio("B", tenant_id=tenant_id)
        b_cash = portfolio_b.cash if portfolio_b else 0
        estimated_cost = float(shares) * price
        if estimated_cost > b_cash:
            return {
                "status": "blocked",
                "ticker": ticker_upper,
                "reason": f"Insufficient Portfolio B cash: need ${estimated_cost:,.0f}, have ${b_cash:,.0f}",
            }

    # 4. Execute
    try:
        executed = await executor.execute_trades([trade_schema], tenant_id=tenant_id)
    except Exception as exc:
        log.error("chat_trade_execution_failed", ticker=ticker_upper, error=str(exc))
        return {"status": "error", "ticker": ticker_upper, "message": f"Execution failed: {exc}"}

    if not executed:
        return {
            "status": "rejected",
            "ticker": ticker_upper,
            "message": "Executor rejected the trade (insufficient cash or position).",
        }

    fill = executed[0]

    # 5. Update current_price/market_value so the frontend shows correct data immediately
    if db is not None:
        try:
            await db.update_position_prices(
                "B",
                {ticker_upper: float(fill.price)},
                tenant_id=tenant_id,
            )
        except Exception as exc:
            log.warning("chat_trade_price_update_failed", ticker=ticker_upper, error=str(exc))

    # 5b. Deactivate trailing stops when selling (mirrors orchestrator Step 7.2)
    if side_upper == "SELL" and db is not None:
        try:
            await db.deactivate_trailing_stops_for_ticker(tenant_id, "B", ticker_upper)
            log.info("chat_trade_trailing_stops_deactivated", ticker=ticker_upper)
        except Exception as exc:
            log.warning("chat_trade_trailing_stop_deactivation_failed", ticker=ticker_upper, error=str(exc))

    # 6. Record in state for session-results.json
    state.executed_trades.append(
        {
            "ticker": ticker_upper,
            "side": side_upper,
            "shares": float(fill.shares),
            "price": float(fill.price),
            "conviction": conviction,
            "reason": reason[:200],
            "status": "filled",
        }
    )

    log.info("chat_trade_executed", ticker=ticker_upper, side=side_upper, shares=fill.shares, price=fill.price)

    return {
        "status": "filled",
        "ticker": ticker_upper,
        "side": side_upper,
        "shares": float(fill.shares),
        "price": round(float(fill.price), 2),
        "total": round(float(fill.shares) * float(fill.price), 2),
        "message": f"Trade executed: {side_upper} {fill.shares} {ticker_upper} @ ${fill.price:.2f}",
    }


# ── 2. set_trailing_stop — direct DB write when db available ──────────────────


async def _set_trailing_stop(
    state: ActionState,
    db: Any | None,
    tenant_id: str,
    current_prices: dict[str, float],
    ticker: str,
    trail_pct: float,
    reason: str = "",
) -> dict:
    """Set or update a trailing stop for a position.

    When db is provided (chat/MCP context), creates the stop directly in the DB.
    Otherwise accumulates for the orchestrator to apply post-agent-loop.
    """
    if not ticker:
        return {"error": "ticker is required"}

    if trail_pct < 0.03 or trail_pct > 0.20:
        return {"error": f"trail_pct must be 0.03-0.20, got {trail_pct}"}

    ticker_upper = ticker.upper()

    # ── Fallback: accumulate-only (orchestrator path) ─────────────────────
    if db is None:
        request = {
            "ticker": ticker_upper,
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

    # ── Direct creation (chat/MCP path) ───────────────────────────────────

    # Validate position exists
    positions = await db.get_positions("B", tenant_id=tenant_id)
    pos = next((p for p in positions if p.ticker == ticker_upper), None)
    if not pos:
        return {"error": f"No position in {ticker_upper} to set trailing stop for"}

    # Resolve price for peak
    price = current_prices.get(ticker_upper)
    if not price:
        price = await _fetch_current_price(ticker_upper)
    if not price:
        price = float(pos.avg_price)  # Last resort: use avg entry

    try:
        await db.create_trailing_stop(
            tenant_id=tenant_id,
            portfolio="B",
            ticker=ticker_upper,
            entry_price=float(pos.avg_price),
            trail_pct=trail_pct,
        )
    except Exception as exc:
        log.error("chat_trailing_stop_failed", ticker=ticker_upper, error=str(exc))
        return {"status": "error", "ticker": ticker_upper, "message": f"Failed to create trailing stop: {exc}"}

    stop_price = round(price * (1 - trail_pct), 2)

    # Record for session-results.json
    state.trailing_stop_requests.append({"ticker": ticker_upper, "trail_pct": trail_pct, "reason": reason[:200]})

    log.info("chat_trailing_stop_created", ticker=ticker_upper, trail_pct=trail_pct, stop_price=stop_price)

    return {
        "status": "created",
        "ticker": ticker_upper,
        "trail_pct": trail_pct,
        "stop_price": stop_price,
        "message": f"Trailing stop active: {ticker_upper} at {trail_pct * 100:.0f}% (stop=${stop_price})",
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
        "note": "All postures available immediately — paper trading mode.",
    }


# ── 6. discover_ticker (propose a new ticker for the universe) ─────────────


async def _discover_ticker(
    state: ActionState,
    ticker_discovery: object | None,
    tenant_id: str,
    ticker: str,
    reason: str,
    conviction: str = "medium",
    sector_rationale: str = "",
) -> dict:
    """Propose a new ticker for addition to the trading universe.

    Validates via yfinance (market cap, volume) and saves as 'proposed'.
    Owner approves/rejects via Telegram after the session.
    Use search_ticker_info FIRST to research before proposing.
    """
    if ticker_discovery is None:
        return {"success": False, "ticker": ticker, "status": "error", "message": "Discovery system not available"}

    ticker = ticker.upper().strip()
    if not ticker:
        return {"success": False, "ticker": "", "status": "error", "message": "No ticker provided"}
    if not reason:
        return {"success": False, "ticker": ticker, "status": "error", "message": "Reason is required"}

    # Check if already in effective universe (base + tenant additions + discovered)
    from config.universe import FULL_UNIVERSE

    already_in = ticker in FULL_UNIVERSE
    if not already_in:
        from src.storage.database import Database

        if hasattr(ticker_discovery, "_db") and isinstance(ticker_discovery._db, Database):
            from src.agent.tools.market import _is_in_tenant_universe

            already_in = await _is_in_tenant_universe(ticker_discovery._db, tenant_id, ticker)
    if already_in:
        return {
            "success": False,
            "ticker": ticker,
            "status": "already_in_universe",
            "message": f"{ticker} is already in the active trading universe",
        }

    # Check if already discovered for this tenant
    from src.storage.database import Database

    if hasattr(ticker_discovery, "_db") and isinstance(ticker_discovery._db, Database):
        existing = await ticker_discovery._db.get_discovered_ticker(ticker, tenant_id=tenant_id)
        if existing:
            if existing.status == "proposed":
                days_ago = ""
                if existing.proposed_at:
                    from datetime import date

                    delta = (date.today() - existing.proposed_at).days
                    days_ago = f" (proposed {delta} days ago)"
                return {
                    "success": False,
                    "ticker": ticker,
                    "status": "already_pending",
                    "message": f"{ticker} is already pending approval{days_ago}",
                }
            elif existing.status == "approved":
                return {
                    "success": False,
                    "ticker": ticker,
                    "status": "already_approved",
                    "message": f"{ticker} is already approved and in the dynamic universe",
                }

    # Build combined rationale
    full_rationale = reason
    if conviction and conviction != "medium":
        full_rationale = f"[{conviction}] {full_rationale}"
    if sector_rationale:
        full_rationale = f"{full_rationale} | Sector: {sector_rationale}"

    # Use existing TickerDiscovery.propose_ticker() for validation + DB write
    row = await ticker_discovery.propose_ticker(
        ticker=ticker,
        rationale=full_rationale,
        source="agent_tool",
        tenant_id=tenant_id,
    )

    if row is None:
        # Validation failed or limit reached — get reason from validate_ticker
        validation = ticker_discovery.validate_ticker(ticker)
        if not validation.valid:
            return {
                "success": False,
                "ticker": ticker,
                "status": "rejected",
                "message": validation.reason,
            }
        return {
            "success": False,
            "ticker": ticker,
            "status": "rejected",
            "message": "Dynamic ticker limit reached or ticker already exists",
        }

    # Track in ActionState for post-loop Telegram approval
    state.discovery_proposals.append(
        {
            "ticker": ticker,
            "reason": reason,
            "conviction": conviction,
            "source": "agent_tool",
        }
    )

    return {
        "success": True,
        "ticker": ticker,
        "status": "pending_approval",
        "message": "Validated and submitted. Owner will approve/reject via Telegram after this session.",
        "validation": {
            "market_cap_ok": True,
            "volume_ok": True,
            "sector": row.sector or "Unknown",
            "market_cap": f"${row.market_cap / 1e9:.1f}B" if row.market_cap else "N/A",
        },
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
    ticker_discovery: Any | None = None,
    executor: Any | None = None,
    risk_manager: Any | None = None,
    current_prices: dict[str, float] | None = None,
) -> None:
    """Register action tools with a per-run state container.

    Args:
        registry: ToolRegistry to register tools on.
        state: ActionState instance (created fresh per run).
        db: Database instance (for order status lookups and direct stop creation).
        tenant_id: Tenant UUID.
        ticker_discovery: TickerDiscovery instance (for discover_ticker tool).
        executor: Trade executor (PaperTrader or AlpacaExecutor). When provided,
            execute_trade runs directly instead of accumulating.
        risk_manager: RiskManager instance. When provided, pre-trade risk checks
            run before direct execution.
        current_prices: Latest prices dict. Used for risk checks and price lookup.
    """
    _prices = current_prices or {}
    # ── Phase 2 tools ────────────────────────────────────────────────────────
    registry.register(
        name="execute_trade",
        description=(
            "Submit a trade for execution. Specify ticker, side (BUY/SELL), shares, and reason. "
            "Trade goes through RiskManager checks. No approval delays — paper trading mode."
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
        handler=partial(_execute_trade, state, executor, risk_manager, db, tenant_id, _prices),
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
        handler=partial(_set_trailing_stop, state, db, tenant_id, _prices),
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
            "aggressive (max exposure). All postures available — no gate in paper trading."
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
        description=(
            "Add or remove tickers from the AI watchlist for future monitoring. "
            "Use this proactively when: (1) news signals a potential opportunity but risk/timing isn't right yet, "
            "(2) a ticker shows strong momentum but is overextended — waiting for a pullback, "
            "(3) a sector rotation thesis is forming and you want to track candidates, "
            "(4) earnings are upcoming and you want to revisit after results. "
            "The watchlist is injected into every session so you remember what to follow up on."
        ),
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

    # ── Discovery tools ──────────────────────────────────────────────────────
    registry.register(
        name="discover_ticker",
        description=(
            "Propose a new ticker for the trading universe. Validates via yfinance "
            "(market cap >$1B, volume >100K) and submits for owner approval. "
            "Use search_ticker_info FIRST to research, then call this to propose."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "ticker": {"type": "string", "description": "Ticker symbol to propose (e.g., ANET)"},
                "reason": {"type": "string", "description": "Why this ticker should be added (your thesis)"},
                "conviction": {
                    "type": "string",
                    "enum": ["high", "medium", "low"],
                    "description": "Confidence level (default: medium)",
                },
                "sector_rationale": {
                    "type": "string",
                    "description": "How this fits the portfolio's sector strategy",
                },
            },
            "required": ["ticker", "reason"],
        },
        handler=partial(_discover_ticker, state, ticker_discovery, tenant_id),
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

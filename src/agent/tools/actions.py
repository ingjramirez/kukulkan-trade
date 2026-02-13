"""Action tools for the agentic loop — accumulate trade proposals and state.

Uses an ActionState class (instantiated per run) to avoid module-level
globals that would contaminate multi-tenant runs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial

from src.agent.tools import ToolRegistry


@dataclass
class ActionState:
    """Per-run state container for accumulated agent actions."""

    proposed_trades: list[dict] = field(default_factory=list)
    watchlist_updates: list[dict] = field(default_factory=list)
    memory_notes: list[dict] = field(default_factory=list)

    def get_accumulated_state(self) -> dict:
        """Return all accumulated actions as a dict."""
        return {
            "trades": list(self.proposed_trades),
            "watchlist_updates": list(self.watchlist_updates),
            "memory_notes": list(self.memory_notes),
        }

    def reset(self) -> None:
        """Clear all accumulated state."""
        self.proposed_trades.clear()
        self.watchlist_updates.clear()
        self.memory_notes.clear()


async def _propose_trades(
    state: ActionState,
    trades: list[dict],
) -> dict:
    """Accumulate trade proposals.

    Args:
        state: ActionState instance.
        trades: List of trade dicts with ticker, side, weight, conviction, reason.

    Returns:
        Confirmation with current trade count.
    """
    for trade in trades:
        # Validate required fields
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
    """Accumulate watchlist updates.

    Args:
        state: ActionState instance.
        updates: List of update dicts with action, ticker, reason, etc.

    Returns:
        Confirmation.
    """
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
    """Accumulate a memory note.

    Args:
        state: ActionState instance.
        key: Note key (e.g. "thesis-tech").
        content: Note content (max ~50 words).

    Returns:
        Confirmation.
    """
    state.memory_notes.append({"key": key, "content": content[:200]})
    return {
        "status": "ok",
        "memory_notes_accumulated": len(state.memory_notes),
    }


def register_action_tools(
    registry: ToolRegistry,
    state: ActionState,
) -> None:
    """Register action tools with a per-run state container.

    Args:
        registry: ToolRegistry to register tools on.
        state: ActionState instance (created fresh per run).
    """
    registry.register(
        name="propose_trades",
        description="Submit trade proposals. Can be called multiple times to accumulate trades.",
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
                            "weight": {"type": "number", "description": "Target portfolio weight 0.0-0.30"},
                            "conviction": {"type": "string", "enum": ["high", "medium", "low"]},
                            "reason": {"type": "string"},
                        },
                        "required": ["ticker", "side"],
                    },
                    "description": "List of trade proposals",
                },
            },
            "required": ["trades"],
        },
        handler=partial(_propose_trades, state),
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

    registry.register(
        name="save_memory_note",
        description="Save an observation to persist across sessions. Use for theses, lessons, correlations.",
        input_schema={
            "type": "object",
            "properties": {
                "key": {"type": "string", "description": "Note key like 'thesis-tech' or 'lesson-timing'"},
                "content": {"type": "string", "description": "Note content (max ~50 words)"},
            },
            "required": ["key", "content"],
        },
        handler=partial(_save_memory_note, state),
    )

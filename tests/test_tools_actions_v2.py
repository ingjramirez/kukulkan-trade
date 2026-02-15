"""Tests for Phase 2 action tools — execute_trade, set_trailing_stop,
get_order_status, save_observation, and legacy aliases.
"""

import pytest

from src.agent.tools import ToolRegistry
from src.agent.tools.actions import (
    ActionState,
    _execute_trade,
    _get_order_status,
    _save_memory_note,
    _save_observation,
    _set_trailing_stop,
    register_action_tools,
)
from src.storage.database import Database


@pytest.fixture
def state():
    return ActionState()


@pytest.fixture
async def db():
    database = Database(url="sqlite+aiosqlite:///:memory:")
    await database.init_db()
    yield database
    await database.close()


# ── execute_trade ────────────────────────────────────────────────────────────


async def test_execute_trade_basic(state: ActionState):
    """execute_trade accumulates a trade with correct format."""
    result = await _execute_trade(state, "NVDA", "BUY", 50, reason="AI thesis", conviction="high")
    assert result["status"] == "submitted"
    assert result["ticker"] == "NVDA"
    assert result["side"] == "BUY"
    assert result["shares"] == 50
    assert result["trades_accumulated"] == 1

    # Should also be in proposed_trades for backward compat
    assert len(state.proposed_trades) == 1
    assert state.proposed_trades[0]["ticker"] == "NVDA"
    assert state.proposed_trades[0]["shares_requested"] == 50


async def test_execute_trade_sell(state: ActionState):
    """execute_trade handles SELL side."""
    result = await _execute_trade(state, "XLK", "SELL", 100, reason="Take profit")
    assert result["side"] == "SELL"
    assert len(state.executed_trades) == 1


async def test_execute_trade_case_insensitive(state: ActionState):
    """execute_trade normalizes ticker and side to uppercase."""
    result = await _execute_trade(state, "nvda", "buy", 50)
    assert result["ticker"] == "NVDA"
    assert result["side"] == "BUY"


async def test_execute_trade_invalid_side(state: ActionState):
    """execute_trade rejects invalid side."""
    result = await _execute_trade(state, "NVDA", "SHORT", 50)
    assert "error" in result


async def test_execute_trade_zero_shares(state: ActionState):
    """execute_trade rejects zero shares."""
    result = await _execute_trade(state, "NVDA", "BUY", 0)
    assert "error" in result


async def test_execute_trade_missing_ticker(state: ActionState):
    """execute_trade rejects empty ticker."""
    result = await _execute_trade(state, "", "BUY", 50)
    assert "error" in result


async def test_execute_trade_multiple(state: ActionState):
    """execute_trade accumulates multiple trades."""
    await _execute_trade(state, "NVDA", "BUY", 50)
    await _execute_trade(state, "XLK", "SELL", 100)
    assert len(state.executed_trades) == 2
    assert len(state.proposed_trades) == 2


# ── set_trailing_stop ────────────────────────────────────────────────────────


async def test_set_trailing_stop_basic(state: ActionState):
    """set_trailing_stop accumulates a stop request."""
    result = await _set_trailing_stop(state, "NVDA", 0.07, reason="Standard 7% stop")
    assert result["status"] == "ok"
    assert result["ticker"] == "NVDA"
    assert result["trail_pct"] == 0.07
    assert result["stops_accumulated"] == 1


async def test_set_trailing_stop_validation(state: ActionState):
    """set_trailing_stop rejects out-of-range trail_pct."""
    result = await _set_trailing_stop(state, "NVDA", 0.01)
    assert "error" in result

    result = await _set_trailing_stop(state, "NVDA", 0.50)
    assert "error" in result


async def test_set_trailing_stop_missing_ticker(state: ActionState):
    """set_trailing_stop rejects empty ticker."""
    result = await _set_trailing_stop(state, "", 0.07)
    assert "error" in result


async def test_set_trailing_stop_multiple(state: ActionState):
    """set_trailing_stop accumulates multiple requests."""
    await _set_trailing_stop(state, "NVDA", 0.07)
    await _set_trailing_stop(state, "XLK", 0.05)
    assert len(state.trailing_stop_requests) == 2


# ── get_order_status ─────────────────────────────────────────────────────────


async def test_order_status_empty(state: ActionState, db: Database):
    """get_order_status with no trades shows empty."""
    result = await _get_order_status(state, db, "default")
    assert result["pending_this_session"] == 0
    assert result["session_trades"] == []
    assert result["recent_fills"] == []


async def test_order_status_with_session_trades(state: ActionState, db: Database):
    """get_order_status shows trades submitted this session."""
    await _execute_trade(state, "NVDA", "BUY", 50)
    result = await _get_order_status(state, db, "default")
    assert result["pending_this_session"] == 1
    assert result["session_trades"][0]["ticker"] == "NVDA"


async def test_order_status_with_fills(state: ActionState, db: Database):
    """get_order_status shows recent fills from DB."""
    await db.upsert_portfolio("B", cash=66000.0, total_value=66000.0)
    await db.log_trade("B", "NVDA", "BUY", 50, 118.0, reason="test", tenant_id="default")

    result = await _get_order_status(state, db, "default")
    assert len(result["recent_fills"]) == 1
    assert result["recent_fills"][0]["ticker"] == "NVDA"


async def test_order_status_filter_by_ticker(state: ActionState, db: Database):
    """get_order_status filters by ticker when specified."""
    await _execute_trade(state, "NVDA", "BUY", 50)
    await _execute_trade(state, "XLK", "SELL", 100)

    result = await _get_order_status(state, db, "default", ticker="NVDA")
    assert result["pending_this_session"] == 1
    assert result["session_trades"][0]["ticker"] == "NVDA"


# ── save_observation ─────────────────────────────────────────────────────────


async def test_save_observation_basic(state: ActionState):
    """save_observation accumulates a memory note."""
    result = await _save_observation(state, "thesis-tech", "AI spending accelerating")
    assert result["status"] == "ok"
    assert result["key"] == "thesis-tech"
    assert result["memory_notes_accumulated"] == 1
    assert state.memory_notes[0]["key"] == "thesis-tech"


async def test_save_observation_truncates(state: ActionState):
    """save_observation truncates long content."""
    long_content = "x" * 500
    await _save_observation(state, "test", long_content)
    assert len(state.memory_notes[0]["content"]) == 200


# ── Legacy aliases ───────────────────────────────────────────────────────────


async def test_save_memory_note_delegates(state: ActionState):
    """Legacy save_memory_note delegates to save_observation."""
    result = await _save_memory_note(state, "legacy-key", "legacy content")
    assert result["status"] == "ok"
    assert state.memory_notes[0]["key"] == "legacy-key"


# ── ActionState ──────────────────────────────────────────────────────────────


async def test_action_state_get_accumulated(state: ActionState):
    """get_accumulated_state includes all new fields."""
    await _execute_trade(state, "NVDA", "BUY", 50)
    await _set_trailing_stop(state, "NVDA", 0.07)

    accumulated = state.get_accumulated_state()
    assert "executed_trades" in accumulated
    assert "trailing_stop_requests" in accumulated
    assert len(accumulated["executed_trades"]) == 1
    assert len(accumulated["trailing_stop_requests"]) == 1


async def test_action_state_reset(state: ActionState):
    """reset() clears all accumulated state including new fields."""
    await _execute_trade(state, "NVDA", "BUY", 50)
    await _set_trailing_stop(state, "NVDA", 0.07)
    state.reset()
    assert len(state.executed_trades) == 0
    assert len(state.trailing_stop_requests) == 0
    assert len(state.proposed_trades) == 0


# ── Registration ─────────────────────────────────────────────────────────────


async def test_registration_all_tools(db: Database):
    """register_action_tools registers all Phase 2 + legacy tools."""
    state = ActionState()
    registry = ToolRegistry()
    register_action_tools(registry, state, db=db, tenant_id="default")

    names = registry.tool_names
    # Phase 2 tools
    assert "execute_trade" in names
    assert "set_trailing_stop" in names
    assert "get_order_status" in names
    assert "save_observation" in names
    assert "update_watchlist" in names
    # Phase 32 aliases
    assert "propose_trades" in names
    assert "save_memory_note" in names


async def test_registration_without_db():
    """register_action_tools without db skips get_order_status."""
    state = ActionState()
    registry = ToolRegistry()
    register_action_tools(registry, state)

    names = registry.tool_names
    assert "execute_trade" in names
    assert "get_order_status" not in names

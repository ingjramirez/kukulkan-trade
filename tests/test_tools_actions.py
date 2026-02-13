"""Tests for action tools — trade proposals, watchlist, memory notes."""

import pytest

from src.agent.tools import ToolRegistry
from src.agent.tools.actions import (
    ActionState,
    _propose_trades,
    _save_memory_note,
    _update_watchlist,
    register_action_tools,
)


@pytest.fixture
def state():
    return ActionState()


@pytest.mark.asyncio
async def test_propose_accumulates(state):
    await _propose_trades(state, [{"ticker": "XLK", "side": "BUY", "weight": 0.15}])
    await _propose_trades(state, [{"ticker": "XLE", "side": "SELL", "weight": 0.0}])
    assert len(state.proposed_trades) == 2
    assert state.proposed_trades[0]["ticker"] == "XLK"
    assert state.proposed_trades[1]["ticker"] == "XLE"


@pytest.mark.asyncio
async def test_watchlist_accumulates(state):
    await _update_watchlist(state, [{"action": "add", "ticker": "AAPL", "reason": "test"}])
    await _update_watchlist(state, [{"action": "remove", "ticker": "MSFT"}])
    assert len(state.watchlist_updates) == 2


@pytest.mark.asyncio
async def test_memory_note(state):
    await _save_memory_note(state, "thesis-tech", "Tech rotation confirmed")
    assert len(state.memory_notes) == 1
    assert state.memory_notes[0]["key"] == "thesis-tech"


@pytest.mark.asyncio
async def test_get_accumulated_state(state):
    await _propose_trades(state, [{"ticker": "XLK", "side": "BUY"}])
    await _update_watchlist(state, [{"action": "add", "ticker": "AAPL"}])
    await _save_memory_note(state, "key", "value")
    accumulated = state.get_accumulated_state()
    assert len(accumulated["trades"]) == 1
    assert len(accumulated["watchlist_updates"]) == 1
    assert len(accumulated["memory_notes"]) == 1


@pytest.mark.asyncio
async def test_reset(state):
    await _propose_trades(state, [{"ticker": "XLK", "side": "BUY"}])
    await _save_memory_note(state, "k", "v")
    state.reset()
    assert len(state.proposed_trades) == 0
    assert len(state.memory_notes) == 0


@pytest.mark.asyncio
async def test_registration():
    state = ActionState()
    registry = ToolRegistry()
    register_action_tools(registry, state)
    assert "propose_trades" in registry.tool_names
    assert "update_watchlist" in registry.tool_names
    assert "save_memory_note" in registry.tool_names

    # Execute through registry
    result = await registry.execute("propose_trades", {"trades": [{"ticker": "XLK", "side": "BUY"}]})
    assert result["trades_accumulated"] == 1

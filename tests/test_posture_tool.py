"""Tests for the declare_posture tool in src/agent/tools/actions.py."""

import pytest

from src.agent.tools.actions import ActionState, _declare_posture


@pytest.fixture
def state() -> ActionState:
    return ActionState()


# ── declare_posture ───────────────────────────────────────────────────────────


async def test_declare_posture_valid(state: ActionState) -> None:
    """Valid posture 'defensive' sets state and returns ok."""
    result = await _declare_posture(state, "defensive", "High VIX")
    assert result["status"] == "ok"
    assert result["posture"] == "defensive"
    assert state.declared_posture == "defensive"


async def test_declare_posture_invalid(state: ActionState) -> None:
    """Invalid posture returns an error."""
    result = await _declare_posture(state, "unknown", "Bad posture")
    assert "error" in result
    assert state.declared_posture is None


async def test_declare_posture_aggressive(state: ActionState) -> None:
    """Aggressive posture is accepted with a note about the gate."""
    result = await _declare_posture(state, "aggressive", "Strong momentum")
    assert result["status"] == "ok"
    assert result["posture"] == "aggressive"
    assert "gate" in result["note"].lower()
    assert state.declared_posture == "aggressive"


# ── ActionState integration ───────────────────────────────────────────────────


async def test_action_state_accumulated_includes_posture(state: ActionState) -> None:
    """get_accumulated_state includes declared_posture."""
    await _declare_posture(state, "crisis", "Market crash")
    accumulated = state.get_accumulated_state()
    assert accumulated["declared_posture"] == "crisis"


async def test_action_state_reset_clears_posture(state: ActionState) -> None:
    """reset() clears declared_posture back to None."""
    await _declare_posture(state, "defensive", "Caution")
    assert state.declared_posture == "defensive"
    state.reset()
    assert state.declared_posture is None

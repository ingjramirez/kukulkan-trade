"""Tests for the PostureManager system in src/agent/posture.py."""

import pytest

from src.agent.posture import (
    POSTURE_CONFIGS,
    PostureLevel,
    PostureManager,
)


@pytest.fixture
def manager() -> PostureManager:
    return PostureManager()


# ── get_limits ────────────────────────────────────────────────────────────────


def test_get_limits_balanced(manager: PostureManager) -> None:
    limits = manager.get_limits(PostureLevel.BALANCED)
    assert limits.max_single_position_pct == 0.35
    assert limits.max_sector_concentration == 0.50
    assert limits.max_equity_pct == 0.80


def test_get_limits_defensive(manager: PostureManager) -> None:
    limits = manager.get_limits(PostureLevel.DEFENSIVE)
    assert limits.max_single_position_pct == 0.25
    assert limits.max_sector_concentration == 0.35
    assert limits.max_equity_pct == 0.50


def test_get_limits_crisis(manager: PostureManager) -> None:
    limits = manager.get_limits(PostureLevel.CRISIS)
    assert limits.max_single_position_pct == 0.15
    assert limits.max_sector_concentration == 0.25
    assert limits.max_equity_pct == 0.30


def test_get_limits_aggressive(manager: PostureManager) -> None:
    limits = manager.get_limits(PostureLevel.AGGRESSIVE)
    assert limits.max_single_position_pct == 0.35
    assert limits.max_sector_concentration == 0.50
    assert limits.max_equity_pct == 0.95


# ── validate_aggressive ──────────────────────────────────────────────────────


def test_validate_aggressive_all_criteria_met(manager: PostureManager) -> None:
    allowed, reason = manager.validate_aggressive(total_trades=60, win_rate_pct=60.0, avg_alpha_vs_spy=2.0)
    assert allowed is True
    assert reason == "Aggressive mode approved"


def test_validate_aggressive_insufficient_trades(manager: PostureManager) -> None:
    allowed, reason = manager.validate_aggressive(total_trades=30, win_rate_pct=60.0, avg_alpha_vs_spy=2.0)
    assert allowed is False
    assert "50+" in reason
    assert "30" in reason


def test_validate_aggressive_low_win_rate(manager: PostureManager) -> None:
    allowed, reason = manager.validate_aggressive(total_trades=60, win_rate_pct=50.0, avg_alpha_vs_spy=2.0)
    assert allowed is False
    assert "win rate" in reason.lower()


def test_validate_aggressive_negative_alpha(manager: PostureManager) -> None:
    allowed, reason = manager.validate_aggressive(total_trades=60, win_rate_pct=60.0, avg_alpha_vs_spy=-1.0)
    assert allowed is False
    assert "alpha" in reason.lower()


def test_validate_aggressive_none_alpha(manager: PostureManager) -> None:
    """None alpha is treated as negative (-1.0), so aggressive is blocked."""
    allowed, reason = manager.validate_aggressive(total_trades=60, win_rate_pct=60.0, avg_alpha_vs_spy=None)
    assert allowed is False
    assert "alpha" in reason.lower()


# ── resolve_effective_limits ──────────────────────────────────────────────────


def test_resolve_balanced_returns_balanced_limits(manager: PostureManager) -> None:
    limits, effective = manager.resolve_effective_limits(PostureLevel.BALANCED)
    assert effective == PostureLevel.BALANCED
    assert limits == POSTURE_CONFIGS[PostureLevel.BALANCED]


def test_resolve_defensive_returns_defensive_limits(manager: PostureManager) -> None:
    limits, effective = manager.resolve_effective_limits(PostureLevel.DEFENSIVE)
    assert effective == PostureLevel.DEFENSIVE
    assert limits == POSTURE_CONFIGS[PostureLevel.DEFENSIVE]


def test_resolve_aggressive_gate_fails_falls_back_to_balanced(manager: PostureManager) -> None:
    """Aggressive with insufficient trades falls back to balanced."""
    limits, effective = manager.resolve_effective_limits(
        PostureLevel.AGGRESSIVE,
        total_trades=10,
        win_rate_pct=40.0,
        avg_alpha_vs_spy=None,
    )
    assert effective == PostureLevel.BALANCED
    assert limits == POSTURE_CONFIGS[PostureLevel.BALANCED]


def test_resolve_aggressive_gate_passes(manager: PostureManager) -> None:
    """Aggressive with all criteria met returns aggressive limits."""
    limits, effective = manager.resolve_effective_limits(
        PostureLevel.AGGRESSIVE,
        total_trades=60,
        win_rate_pct=60.0,
        avg_alpha_vs_spy=2.0,
    )
    assert effective == PostureLevel.AGGRESSIVE
    assert limits == POSTURE_CONFIGS[PostureLevel.AGGRESSIVE]

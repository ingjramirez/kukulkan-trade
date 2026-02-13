"""Tests for TokenTracker — cost tracking and budget enforcement."""

from src.agent.token_tracker import TokenTracker


def test_record_and_cost():
    tracker = TokenTracker(session_budget_usd=1.00)
    tracker.record("claude-sonnet-4-5-20250929", input_tokens=1000, output_tokens=500, turn=1)
    # Cost: (1000 * 3.0 + 500 * 15.0) / 1_000_000 = (3000 + 7500) / 1M = 0.0105
    assert abs(tracker.total_cost_usd - 0.0105) < 0.0001


def test_opus_pricing():
    tracker = TokenTracker(session_budget_usd=1.00)
    tracker.record("claude-opus-4-6", input_tokens=1000, output_tokens=500, turn=1)
    # Cost: (1000 * 15.0 + 500 * 75.0) / 1_000_000 = (15000 + 37500) / 1M = 0.0525
    assert abs(tracker.total_cost_usd - 0.0525) < 0.0001


def test_haiku_pricing():
    tracker = TokenTracker(session_budget_usd=1.00)
    tracker.record("claude-haiku-4-5-20251001", input_tokens=10000, output_tokens=1000, turn=1)
    # Cost: (10000 * 0.80 + 1000 * 4.0) / 1_000_000 = (8000 + 4000) / 1M = 0.012
    assert abs(tracker.total_cost_usd - 0.012) < 0.0001


def test_budget_exceeded():
    tracker = TokenTracker(session_budget_usd=0.01)
    tracker.record("claude-opus-4-6", input_tokens=10000, output_tokens=5000, turn=1)
    # Cost: (10000*15 + 5000*75)/1M = (150000+375000)/1M = 0.525
    assert tracker.budget_exceeded is True


def test_budget_not_exceeded():
    tracker = TokenTracker(session_budget_usd=1.00)
    tracker.record("claude-haiku-4-5-20251001", input_tokens=100, output_tokens=50, turn=1)
    assert tracker.budget_exceeded is False


def test_budget_remaining():
    tracker = TokenTracker(session_budget_usd=0.50)
    tracker.record("claude-sonnet-4-5-20250929", input_tokens=1000, output_tokens=500, turn=1)
    remaining = tracker.budget_remaining_usd
    assert remaining < 0.50
    assert remaining > 0.0


def test_unknown_model_fallback():
    tracker = TokenTracker(session_budget_usd=1.00)
    tracker.record("unknown-model-xyz", input_tokens=1000, output_tokens=500, turn=1)
    # Uses default pricing (same as sonnet)
    assert tracker.total_cost_usd > 0


def test_summary():
    tracker = TokenTracker(session_budget_usd=0.50)
    tracker.record("claude-sonnet-4-5-20250929", input_tokens=1000, output_tokens=500, turn=1)
    tracker.record("claude-sonnet-4-5-20250929", input_tokens=2000, output_tokens=800, turn=2)

    s = tracker.summary()
    assert s["total_input_tokens"] == 3000
    assert s["total_output_tokens"] == 1300
    assert s["turns"] == 2
    assert s["budget_usd"] == 0.50
    assert len(s["entries"]) == 2


def test_total_tokens():
    tracker = TokenTracker()
    tracker.record("claude-sonnet-4-5-20250929", input_tokens=1000, output_tokens=500, turn=1)
    tracker.record("claude-sonnet-4-5-20250929", input_tokens=2000, output_tokens=300, turn=2)
    assert tracker.total_input_tokens == 3000
    assert tracker.total_output_tokens == 800

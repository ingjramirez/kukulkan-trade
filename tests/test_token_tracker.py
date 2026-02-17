"""Tests for TokenTracker — pricing, cache-aware cost, and budget tracking."""

from src.agent.token_tracker import (
    CACHE_READ_MULTIPLIER,
    CACHE_WRITE_MULTIPLIER,
    MODEL_PRICING,
    TokenTracker,
)

# ── Corrected pricing ────────────────────────────────────────────────────────


def test_opus_pricing_corrected():
    assert MODEL_PRICING["claude-opus-4-6"] == (5.0, 25.0)


def test_sonnet_pricing():
    assert MODEL_PRICING["claude-sonnet-4-6"] == (3.0, 15.0)


def test_haiku_pricing_corrected():
    assert MODEL_PRICING["claude-haiku-4-5-20251001"] == (1.0, 5.0)


def test_cache_multipliers():
    assert CACHE_WRITE_MULTIPLIER == 1.25
    assert CACHE_READ_MULTIPLIER == 0.10


# ── Basic cost (no cache) ────────────────────────────────────────────────────


def test_record_and_cost_sonnet():
    tracker = TokenTracker(session_budget_usd=1.00)
    tracker.record("claude-sonnet-4-6", input_tokens=1000, output_tokens=500, turn=1)
    # Cost: (1000 * 3.0 + 500 * 15.0) / 1M = 0.0105
    assert abs(tracker.total_cost_usd - 0.0105) < 1e-8


def test_record_and_cost_opus():
    tracker = TokenTracker(session_budget_usd=1.00)
    tracker.record("claude-opus-4-6", input_tokens=1000, output_tokens=500, turn=1)
    # Cost: (1000 * 5.0 + 500 * 25.0) / 1M = 0.0175
    assert abs(tracker.total_cost_usd - 0.0175) < 1e-8


def test_record_and_cost_haiku():
    tracker = TokenTracker(session_budget_usd=1.00)
    tracker.record("claude-haiku-4-5-20251001", input_tokens=10000, output_tokens=1000, turn=1)
    # Cost: (10000 * 1.0 + 1000 * 5.0) / 1M = 0.015
    assert abs(tracker.total_cost_usd - 0.015) < 1e-8


def test_backward_compat_no_cache_params():
    """record() without cache params still works (default 0)."""
    tracker = TokenTracker(session_budget_usd=1.0)
    tracker.record(model="claude-sonnet-4-6", input_tokens=1000, output_tokens=500, turn=1)
    assert len(tracker.entries) == 1
    assert tracker.entries[0].cache_creation_tokens == 0
    assert tracker.entries[0].cache_read_tokens == 0


# ── Budget tracking ──────────────────────────────────────────────────────────


def test_budget_exceeded():
    tracker = TokenTracker(session_budget_usd=0.01)
    tracker.record("claude-opus-4-6", input_tokens=10000, output_tokens=5000, turn=1)
    assert tracker.budget_exceeded is True


def test_budget_not_exceeded():
    tracker = TokenTracker(session_budget_usd=1.00)
    tracker.record("claude-haiku-4-5-20251001", input_tokens=100, output_tokens=50, turn=1)
    assert tracker.budget_exceeded is False


def test_budget_remaining():
    tracker = TokenTracker(session_budget_usd=0.50)
    tracker.record("claude-sonnet-4-6", input_tokens=1000, output_tokens=500, turn=1)
    assert 0.0 < tracker.budget_remaining_usd < 0.50


def test_unknown_model_fallback():
    tracker = TokenTracker(session_budget_usd=1.00)
    tracker.record("unknown-model-xyz", input_tokens=1000, output_tokens=500, turn=1)
    # Uses default pricing (3.0, 15.0) — same as Sonnet
    assert abs(tracker.total_cost_usd - 0.0105) < 1e-8


# ── Cache-aware cost ─────────────────────────────────────────────────────────


def test_cache_creation_cost():
    tracker = TokenTracker()
    # 500 input * 3.0 + 500 cache_write * 3.0 * 1.25 + 200 output * 15.0
    # = 1500 + 1875 + 3000 = 6375  →  6375 / 1M = 0.006375
    tracker.record(
        model="claude-sonnet-4-6",
        input_tokens=500,
        output_tokens=200,
        turn=1,
        cache_creation_tokens=500,
    )
    assert abs(tracker.total_cost_usd - 0.006375) < 1e-8


def test_cache_read_cost():
    tracker = TokenTracker()
    # 500 * 3.0 + 2000 * 3.0 * 0.10 + 200 * 15.0 = 1500 + 600 + 3000 = 5100 / 1M = 0.0051
    tracker.record(
        model="claude-sonnet-4-6",
        input_tokens=500,
        output_tokens=200,
        turn=1,
        cache_read_tokens=2000,
    )
    assert abs(tracker.total_cost_usd - 0.0051) < 1e-8


def test_mixed_cache_and_normal():
    tracker = TokenTracker()
    tracker.record(
        model="claude-sonnet-4-6",
        input_tokens=1000,
        output_tokens=500,
        turn=1,
        cache_creation_tokens=2000,
    )
    tracker.record(
        model="claude-sonnet-4-6",
        input_tokens=200,
        output_tokens=500,
        turn=2,
        cache_read_tokens=2000,
    )
    assert tracker.total_cache_creation_tokens == 2000
    assert tracker.total_cache_read_tokens == 2000
    assert len(tracker.entries) == 2


def test_cache_savings():
    tracker = TokenTracker()
    # 2000 cache_read with Sonnet (3.0):
    # full = 2000 * 3.0 / 1M = 0.006
    # cached = 2000 * 3.0 * 0.10 / 1M = 0.0006
    # savings = 0.0054
    tracker.record(
        model="claude-sonnet-4-6",
        input_tokens=500,
        output_tokens=200,
        turn=1,
        cache_read_tokens=2000,
    )
    assert abs(tracker.cache_savings_usd - 0.0054) < 1e-8


def test_no_cache_savings_without_reads():
    tracker = TokenTracker()
    tracker.record(model="claude-sonnet-4-6", input_tokens=1000, output_tokens=500, turn=1)
    assert tracker.cache_savings_usd == 0.0


# ── Summary ──────────────────────────────────────────────────────────────────


def test_summary_includes_cache_fields():
    tracker = TokenTracker(session_budget_usd=0.50)
    tracker.record(
        model="claude-sonnet-4-6",
        input_tokens=1000,
        output_tokens=500,
        turn=1,
        cache_creation_tokens=300,
        cache_read_tokens=200,
    )
    s = tracker.summary()
    assert s["total_cache_creation_tokens"] == 300
    assert s["total_cache_read_tokens"] == 200
    assert "cache_savings_usd" in s


def test_summary_entries_include_cache_fields():
    tracker = TokenTracker()
    tracker.record(
        model="claude-opus-4-6",
        input_tokens=500,
        output_tokens=200,
        turn=1,
        cache_creation_tokens=100,
        cache_read_tokens=50,
    )
    entry = tracker.summary()["entries"][0]
    assert entry["cache_creation_tokens"] == 100
    assert entry["cache_read_tokens"] == 50


def test_summary_basic():
    tracker = TokenTracker(session_budget_usd=0.50)
    tracker.record("claude-sonnet-4-6", input_tokens=1000, output_tokens=500, turn=1)
    tracker.record("claude-sonnet-4-6", input_tokens=2000, output_tokens=800, turn=2)
    s = tracker.summary()
    assert s["total_input_tokens"] == 3000
    assert s["total_output_tokens"] == 1300
    assert s["turns"] == 2
    assert s["budget_usd"] == 0.50
    assert len(s["entries"]) == 2


def test_total_tokens():
    tracker = TokenTracker()
    tracker.record("claude-sonnet-4-6", input_tokens=1000, output_tokens=500, turn=1)
    tracker.record("claude-sonnet-4-6", input_tokens=2000, output_tokens=300, turn=2)
    assert tracker.total_input_tokens == 3000
    assert tracker.total_output_tokens == 800

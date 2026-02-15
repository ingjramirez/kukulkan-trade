"""Tests for ContextManager — context building for persistent agent."""

from datetime import datetime, timezone

from src.agent.context_manager import ContextManager


def make_manager() -> ContextManager:
    return ContextManager()


def test_build_system_prompt_includes_pinned_context():
    """System prompt includes the pinned context section."""
    cm = make_manager()
    pinned = "## Active Theses\n- NVDA: Half position at $118"
    prompt = cm.build_system_prompt(pinned_context=pinned)
    assert "NVDA" in prompt
    assert "Half position at $118" in prompt
    assert "AI portfolio manager" in prompt


def test_build_system_prompt_with_strategy_directive():
    """System prompt includes strategy directive when provided."""
    cm = make_manager()
    prompt = cm.build_system_prompt(
        pinned_context="",
        strategy_directive="Focus on defensive positions in this regime.",
    )
    assert "Focus on defensive positions" in prompt


def test_build_system_prompt_without_pinned():
    """System prompt works with empty pinned context."""
    cm = make_manager()
    prompt = cm.build_system_prompt(pinned_context="")
    assert "AI portfolio manager" in prompt


def test_build_messages_with_summaries_and_recent():
    """Messages array includes summaries preamble + recent sessions + trigger."""
    cm = make_manager()
    summaries = [
        {
            "session_id": "s1",
            "trigger_type": "morning",
            "summary": "Bought NVDA at $118.",
            "created_at": datetime(2026, 2, 10, 15, 0, tzinfo=timezone.utc),
        },
    ]
    recent_sessions = [
        {
            "session_id": "s2",
            "trigger_type": "midday",
            "messages": [
                {"role": "user", "content": "Midday check."},
                {"role": "assistant", "content": "All positions holding."},
            ],
            "created_at": datetime(2026, 2, 14, 17, 30, tzinfo=timezone.utc),
        },
    ]
    trigger = "Good morning. VIX 18.2."

    messages = cm.build_messages(summaries, recent_sessions, trigger)

    # Structure: summary preamble (user) + ack (assistant) + recent messages + trigger
    assert len(messages) == 5
    assert messages[0]["role"] == "user"
    assert "Bought NVDA at $118" in messages[0]["content"]
    assert messages[1]["role"] == "assistant"
    assert "context from my prior" in messages[1]["content"]
    assert messages[2]["role"] == "user"
    assert messages[2]["content"] == "Midday check."
    assert messages[3]["role"] == "assistant"
    assert messages[4]["role"] == "user"
    assert messages[4]["content"] == trigger


def test_build_messages_empty_history():
    """First session ever — just the trigger message."""
    cm = make_manager()
    messages = cm.build_messages([], [], "Good morning. Markets open.")
    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    assert messages[0]["content"] == "Good morning. Markets open."


def test_build_messages_only_summaries_no_recent():
    """Only compressed summaries, no recent full sessions."""
    cm = make_manager()
    summaries = [
        {
            "session_id": "s1",
            "trigger_type": "morning",
            "summary": "Held positions, no trades.",
            "created_at": datetime(2026, 2, 10, 15, 0, tzinfo=timezone.utc),
        },
    ]
    messages = cm.build_messages(summaries, [], "Midday update.")
    # summary preamble + ack + trigger = 3
    assert len(messages) == 3
    assert "Held positions" in messages[0]["content"]


def test_build_messages_only_recent_no_summaries():
    """Only recent sessions, no compressed summaries."""
    cm = make_manager()
    recent = [
        {
            "session_id": "s1",
            "trigger_type": "morning",
            "messages": [
                {"role": "user", "content": "Morning trigger."},
                {"role": "assistant", "content": "Checking portfolio."},
            ],
            "created_at": datetime(2026, 2, 14, 15, 0, tzinfo=timezone.utc),
        },
    ]
    messages = cm.build_messages([], recent, "Midday update.")
    # 2 recent messages + trigger = 3
    assert len(messages) == 3
    assert messages[0]["content"] == "Morning trigger."
    assert messages[2]["content"] == "Midday update."


def test_trigger_message_morning_format():
    """Morning trigger includes regime, VIX, SPY, and portfolio info."""
    cm = make_manager()
    msg = cm.build_trigger_message(
        trigger_type="morning",
        market_data={"regime": "BULL", "vix": 15.3, "spy_change_pct": 0.5},
        portfolio_summary={"total_value": 66000, "cash": 25000, "positions_count": 4},
    )
    assert "Good morning" in msg
    assert "BULL" in msg
    assert "15.3" in msg
    assert "66000" in msg


def test_trigger_message_midday_format():
    """Midday trigger includes VIX and portfolio summary."""
    cm = make_manager()
    msg = cm.build_trigger_message(
        trigger_type="midday",
        market_data={"vix": 17.0},
        portfolio_summary={"total_value": 65000, "cash": 24000},
    )
    assert "Midday" in msg
    assert "17.0" in msg


def test_trigger_message_close_format():
    """Close trigger mentions overnight risk."""
    cm = make_manager()
    msg = cm.build_trigger_message(
        trigger_type="close",
        market_data={"vix": 18.5},
        portfolio_summary={"total_value": 65500, "cash": 24000},
    )
    assert "closing" in msg.lower()
    assert "overnight" in msg.lower()


def test_trigger_message_event_format():
    """Event trigger includes event type and detail."""
    cm = make_manager()
    msg = cm.build_trigger_message(
        trigger_type="event",
        market_data={"event_type": "VIX spike", "event_detail": "VIX crossed 28."},
    )
    assert "ALERT" in msg
    assert "VIX spike" in msg
    assert "VIX crossed 28" in msg


def test_trigger_message_weekly_review_format():
    """Weekly review trigger includes outcomes."""
    cm = make_manager()
    msg = cm.build_trigger_message(
        trigger_type="weekly_review",
        market_data={"outcomes_summary": "3 wins, 1 loss. +$450 net."},
    )
    assert "Weekly review" in msg
    assert "3 wins, 1 loss" in msg


def test_estimate_tokens_approximate():
    """Token estimation is roughly 4 chars per token."""
    cm = make_manager()
    messages = [
        {"role": "user", "content": "A" * 400},  # ~100 tokens
        {"role": "assistant", "content": "B" * 200},  # ~50 tokens
    ]
    tokens = cm.estimate_tokens(messages)
    assert 100 <= tokens <= 200  # Approximate


def test_estimate_tokens_with_tool_blocks():
    """Token estimation handles tool use blocks in content."""
    cm = make_manager()
    messages = [
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "X" * 100},
                {"type": "tool_use", "id": "t1", "name": "get_price", "input": {"ticker": "NVDA"}},
            ],
        },
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "Y" * 200}],
        },
    ]
    tokens = cm.estimate_tokens(messages)
    assert tokens > 0


def test_should_compress_threshold():
    """should_compress returns True above threshold."""
    cm = make_manager()
    assert cm.should_compress(200_000) is True
    assert cm.should_compress(100_000) is False
    assert cm.should_compress(150_001) is True
    assert cm.should_compress(150_000) is False


def test_build_pinned_context_with_theses():
    """Pinned context includes active theses."""
    cm = make_manager()
    pinned = cm.build_pinned_context(
        active_theses=[
            {"ticker": "NVDA", "description": "Half position at $118, adding if holds $115", "entered": "Feb 10"},
        ],
        key_learnings=["Tech in CONSOLIDATION: 35% win rate — reduce sizing"],
        current_posture="Defensive",
        track_record_summary="Overall: 23W-15L (60%). Alpha vs SPY: +1.2%.",
    )
    assert "NVDA" in pinned
    assert "Half position at $118" in pinned
    assert "Feb 10" in pinned
    assert "Tech in CONSOLIDATION" in pinned
    assert "Defensive" in pinned
    assert "23W-15L" in pinned


def test_build_pinned_context_empty_theses():
    """Pinned context handles no active theses gracefully."""
    cm = make_manager()
    pinned = cm.build_pinned_context(
        active_theses=[],
        current_posture="Balanced",
    )
    assert "No active theses" in pinned
    assert "Balanced" in pinned


def test_recent_sessions_replayed_as_conversation():
    """Multiple recent sessions are replayed sequentially."""
    cm = make_manager()
    recent = [
        {
            "session_id": "s1",
            "trigger_type": "morning",
            "messages": [
                {"role": "user", "content": "Morning 1."},
                {"role": "assistant", "content": "Response 1."},
            ],
            "created_at": datetime(2026, 2, 13, 15, 0, tzinfo=timezone.utc),
        },
        {
            "session_id": "s2",
            "trigger_type": "midday",
            "messages": [
                {"role": "user", "content": "Midday 2."},
                {"role": "assistant", "content": "Response 2."},
            ],
            "created_at": datetime(2026, 2, 13, 17, 30, tzinfo=timezone.utc),
        },
    ]
    messages = cm.build_messages([], recent, "Close trigger.")
    # 2 + 2 + 1 = 5
    assert len(messages) == 5
    assert messages[0]["content"] == "Morning 1."
    assert messages[1]["content"] == "Response 1."
    assert messages[2]["content"] == "Midday 2."
    assert messages[3]["content"] == "Response 2."
    assert messages[4]["content"] == "Close trigger."


def test_context_budget_within_limits():
    """Constant budget values are reasonable."""
    cm = make_manager()
    total_budget = cm.SYSTEM_PROMPT_BUDGET + cm.PINNED_CONTEXT_BUDGET + cm.SUMMARY_BUDGET + cm.RECENT_HISTORY_BUDGET
    assert total_budget < cm.COMPRESSION_THRESHOLD
    assert total_budget < 200_000  # Stay within Anthropic context window

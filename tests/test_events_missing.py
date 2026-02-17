"""Tests for the 4 newly wired SSE event producers."""

from __future__ import annotations

import pytest

from src.events.event_bus import Event, EventType, event_bus


@pytest.fixture(autouse=True)
def _clean_bus():
    event_bus.clear()
    yield
    event_bus.clear()


# --- TRADE_REJECTED ---


async def test_trade_rejected_event():
    sub_id, queue = event_bus.subscribe(tenant_id="default")
    event_bus.publish(
        Event(
            type=EventType.TRADE_REJECTED,
            tenant_id="default",
            data={
                "ticker": "AAPL",
                "side": "BUY",
                "shares": 400,
                "reason": "AAPL would be 40% of portfolio (limit 35%)",
                "portfolio": "B",
            },
        )
    )
    evt = queue.get_nowait()
    assert evt.type == EventType.TRADE_REJECTED
    assert evt.data["ticker"] == "AAPL"
    assert "40%" in evt.data["reason"]
    event_bus.unsubscribe(sub_id)


async def test_trade_rejected_includes_all_fields():
    sub_id, queue = event_bus.subscribe(tenant_id="default")
    event_bus.publish(
        Event(
            type=EventType.TRADE_REJECTED,
            tenant_id="default",
            data={
                "ticker": "TSLA",
                "side": "BUY",
                "shares": 200,
                "reason": "Tech sector concentration exceeded",
                "portfolio": "B",
            },
        )
    )
    evt = queue.get_nowait()
    assert evt.data["side"] == "BUY"
    assert evt.data["shares"] == 200
    assert evt.data["portfolio"] == "B"
    event_bus.unsubscribe(sub_id)


# --- WATCHLIST_UPDATED ---


async def test_watchlist_updated_event():
    sub_id, queue = event_bus.subscribe(tenant_id="default")
    event_bus.publish(
        Event(
            type=EventType.WATCHLIST_UPDATED,
            tenant_id="default",
            data={"additions": 2, "removals": 1},
        )
    )
    evt = queue.get_nowait()
    assert evt.type == EventType.WATCHLIST_UPDATED
    assert evt.data["additions"] == 2
    assert evt.data["removals"] == 1
    event_bus.unsubscribe(sub_id)


async def test_watchlist_updated_zero_counts():
    """Edge case: all updates are invalid actions → 0/0."""
    sub_id, queue = event_bus.subscribe(tenant_id="default")
    event_bus.publish(
        Event(
            type=EventType.WATCHLIST_UPDATED,
            tenant_id="default",
            data={"additions": 0, "removals": 0},
        )
    )
    evt = queue.get_nowait()
    assert evt.data["additions"] == 0
    assert evt.data["removals"] == 0
    event_bus.unsubscribe(sub_id)


# --- SYSTEM_ERROR ---


async def test_system_error_event():
    sub_id, queue = event_bus.subscribe(tenant_id="default")
    event_bus.publish(
        Event(
            type=EventType.SYSTEM_ERROR,
            tenant_id="default",
            data={"message": "Connection refused", "step": "Morning"},
        )
    )
    evt = queue.get_nowait()
    assert evt.type == EventType.SYSTEM_ERROR
    assert "Connection refused" in evt.data["message"]
    assert evt.data["step"] == "Morning"
    event_bus.unsubscribe(sub_id)


async def test_system_error_truncated_message():
    """Message should be truncated to 200 chars at the producer."""
    long_msg = "x" * 300
    sub_id, queue = event_bus.subscribe(tenant_id="default")
    event_bus.publish(
        Event(
            type=EventType.SYSTEM_ERROR,
            tenant_id="default",
            data={"message": long_msg[:200], "step": "Close"},
        )
    )
    evt = queue.get_nowait()
    assert len(evt.data["message"]) == 200
    event_bus.unsubscribe(sub_id)


# --- IMPROVEMENT_REPORT ---


async def test_improvement_report_event():
    sub_id, queue = event_bus.subscribe(tenant_id="default")
    event_bus.publish(
        Event(
            type=EventType.IMPROVEMENT_REPORT,
            tenant_id="default",
            data={
                "changes_applied": 2,
                "proposals_total": 3,
                "summary": "Solid week with improving win rate",
            },
        )
    )
    evt = queue.get_nowait()
    assert evt.type == EventType.IMPROVEMENT_REPORT
    assert evt.data["changes_applied"] == 2
    assert evt.data["proposals_total"] == 3
    event_bus.unsubscribe(sub_id)


async def test_improvement_report_includes_summary():
    sub_id, queue = event_bus.subscribe(tenant_id="default")
    event_bus.publish(
        Event(
            type=EventType.IMPROVEMENT_REPORT,
            tenant_id="default",
            data={
                "changes_applied": 0,
                "proposals_total": 0,
                "summary": "No changes needed this week",
            },
        )
    )
    evt = queue.get_nowait()
    assert "No changes" in evt.data["summary"]
    event_bus.unsubscribe(sub_id)

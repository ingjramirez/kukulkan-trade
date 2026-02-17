"""Tests for alert event types (circuit breaker, session lifecycle, posture)."""

from __future__ import annotations

import pytest

from src.events.event_bus import Event, EventType, event_bus


@pytest.fixture(autouse=True)
def _clean_bus():
    event_bus.clear()
    yield
    event_bus.clear()


async def test_circuit_breaker_event():
    sub_id, queue = event_bus.subscribe(tenant_id="default")
    event_bus.publish(
        Event(
            type=EventType.CIRCUIT_BREAKER_TRIGGERED,
            tenant_id="default",
            data={"portfolio": "A", "reason": "3-day drawdown > 8%"},
        )
    )
    evt = queue.get_nowait()
    assert evt.type == EventType.CIRCUIT_BREAKER_TRIGGERED
    assert evt.data["portfolio"] == "A"
    event_bus.unsubscribe(sub_id)


async def test_session_started_event():
    sub_id, queue = event_bus.subscribe(tenant_id="default")
    event_bus.publish(
        Event(
            type=EventType.SESSION_STARTED,
            tenant_id="default",
            data={"trigger": "morning", "session": "Morning"},
        )
    )
    evt = queue.get_nowait()
    assert evt.type == EventType.SESSION_STARTED
    assert evt.data["trigger"] == "morning"
    event_bus.unsubscribe(sub_id)


async def test_session_skipped_event():
    sub_id, queue = event_bus.subscribe(tenant_id="default")
    event_bus.publish(
        Event(
            type=EventType.SESSION_SKIPPED,
            tenant_id="default",
            data={"reason": "daily_budget_exhausted", "spent": 3.5},
        )
    )
    evt = queue.get_nowait()
    assert evt.type == EventType.SESSION_SKIPPED
    assert evt.data["reason"] == "daily_budget_exhausted"
    event_bus.unsubscribe(sub_id)


async def test_posture_changed_event():
    sub_id, queue = event_bus.subscribe(tenant_id="default")
    event_bus.publish(
        Event(
            type=EventType.POSTURE_CHANGED,
            tenant_id="default",
            data={"declared": "defensive", "effective": "defensive"},
        )
    )
    evt = queue.get_nowait()
    assert evt.type == EventType.POSTURE_CHANGED
    assert evt.data["declared"] == "defensive"
    event_bus.unsubscribe(sub_id)

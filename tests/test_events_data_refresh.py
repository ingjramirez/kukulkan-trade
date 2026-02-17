"""Tests for data refresh event types (positions, snapshots, intraday, budget)."""

from __future__ import annotations

import pytest

from src.events.event_bus import Event, EventType, event_bus


@pytest.fixture(autouse=True)
def _clean_bus():
    event_bus.clear()
    yield
    event_bus.clear()


async def test_positions_updated_event():
    sub_id, queue = event_bus.subscribe(tenant_id="default")
    event_bus.publish(
        Event(
            type=EventType.POSITIONS_UPDATED,
            tenant_id="default",
            data={"trades_executed": 3},
        )
    )
    evt = queue.get_nowait()
    assert evt.type == EventType.POSITIONS_UPDATED
    assert evt.data["trades_executed"] == 3
    event_bus.unsubscribe(sub_id)


async def test_portfolio_snapshot_event():
    sub_id, queue = event_bus.subscribe(tenant_id="default")
    event_bus.publish(
        Event(
            type=EventType.PORTFOLIO_SNAPSHOT,
            tenant_id="default",
            data={"portfolio": "A", "date": "2026-02-17"},
        )
    )
    evt = queue.get_nowait()
    assert evt.type == EventType.PORTFOLIO_SNAPSHOT
    assert evt.data["portfolio"] == "A"
    event_bus.unsubscribe(sub_id)


async def test_intraday_update_event_carries_values():
    """INTRADAY_UPDATE includes actual portfolio values (exception to signals-only design)."""
    sub_id, queue = event_bus.subscribe(tenant_id="default")
    event_bus.publish(
        Event(
            type=EventType.INTRADAY_UPDATE,
            tenant_id="default",
            data={"portfolio": "B", "equity": 50123.45, "cash": 4876.55},
        )
    )
    evt = queue.get_nowait()
    assert evt.type == EventType.INTRADAY_UPDATE
    assert evt.data["equity"] == 50123.45
    assert evt.data["cash"] == 4876.55
    event_bus.unsubscribe(sub_id)


async def test_budget_updated_event():
    sub_id, queue = event_bus.subscribe(tenant_id="default")
    event_bus.publish(
        Event(
            type=EventType.BUDGET_UPDATED,
            tenant_id="default",
            data={"cost_usd": 0.42, "session_label": "morning"},
        )
    )
    evt = queue.get_nowait()
    assert evt.type == EventType.BUDGET_UPDATED
    assert evt.data["cost_usd"] == 0.42
    event_bus.unsubscribe(sub_id)


async def test_multiple_data_events_in_sequence():
    """Multiple data refresh events arrive in order."""
    sub_id, queue = event_bus.subscribe(tenant_id="default")
    event_bus.publish(Event(type=EventType.POSITIONS_UPDATED, tenant_id="default", data={"trades_executed": 2}))
    event_bus.publish(Event(type=EventType.PORTFOLIO_SNAPSHOT, tenant_id="default", data={"portfolio": "A"}))
    event_bus.publish(Event(type=EventType.BUDGET_UPDATED, tenant_id="default", data={"cost_usd": 0.1}))

    assert queue.qsize() == 3
    types = [queue.get_nowait().type for _ in range(3)]
    assert types == [EventType.POSITIONS_UPDATED, EventType.PORTFOLIO_SNAPSHOT, EventType.BUDGET_UPDATED]
    event_bus.unsubscribe(sub_id)


async def test_data_events_tenant_scoped():
    """Data events for t1 don't reach t2 subscriber."""
    sub_id, queue = event_bus.subscribe(tenant_id="t2")
    event_bus.publish(Event(type=EventType.POSITIONS_UPDATED, tenant_id="t1", data={}))
    event_bus.publish(Event(type=EventType.BUDGET_UPDATED, tenant_id="t1", data={}))
    assert queue.empty()
    event_bus.unsubscribe(sub_id)


async def test_data_events_in_history():
    """Data refresh events are stored in history for catch-up."""
    event_bus.publish(Event(type=EventType.POSITIONS_UPDATED, tenant_id="default", data={"n": 1}))
    event_bus.publish(Event(type=EventType.PORTFOLIO_SNAPSHOT, tenant_id="default", data={"n": 2}))
    history = event_bus.get_recent(tenant_id="default")
    assert len(history) == 2
    assert history[0].type == EventType.POSITIONS_UPDATED
    assert history[1].type == EventType.PORTFOLIO_SNAPSHOT
